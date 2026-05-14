import os
import json
import cv2
import logging
from typing import Any

from football_ai.config.schema import SystemConfig
from football_ai.events.event_extractor import EventExtractor
from football_ai.commentary.script_generator import ScriptGenerator
from football_ai.audio.tts_engine import EdgeVietnameseTTS
from football_ai.audio.audio_mixer import CommentaryAudioMixer

class CommentaryPipeline:
    """
    Coordinates the automated commentary workflow.
    Translates track data into Vietnamese commentary text and speech clips,
    mixing them with source video composites synchronously.
    """
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        # Basic logger console setup if root logger not set
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

    def run(self):
        output_dir = self.config.video.output_dir
        os.makedirs(output_dir, exist_ok=True)

        tracks_path = os.path.join(output_dir, "analytics_tracks.json")
        video_in_path = os.path.join(output_dir, "analytics_video.mp4")

        print("\n================================================")
        print("      RUNNING COMMENTARY PIPELINE")
        print("================================================")

        # 1. Check prerequisites
        if not os.path.exists(tracks_path):
            print(f"[!] CRITICAL ERROR: Track file missing at '{tracks_path}'")
            print("Please run the analytics pipeline FIRST to generate tracks.")
            print("Command: python -m football_ai.cli --mode analytics")
            return

        # 2. Read tracks
        self.logger.info(f"Loading analytics track log: {tracks_path}")
        try:
            with open(tracks_path, 'r', encoding='utf-8') as f:
                tracks_data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to parse track JSON file: {e}")
            return

        # 3. Retrieve metadata from video or fallback
        fps = 30.0
        duration_ms = 0
        
        video_found = os.path.exists(video_in_path)
        if video_found:
            cap = cv2.VideoCapture(video_in_path)
            if cap.isOpened():
                v_fps = cap.get(cv2.CAP_PROP_FPS)
                f_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if v_fps > 0:
                    fps = float(v_fps)
                if f_count > 0 and fps > 0:
                    duration_ms = int((f_count / fps) * 1000)
                cap.release()
            self.logger.info(f"Detected input video dimensions: {fps} FPS, {duration_ms} ms duration.")
        else:
            self.logger.warning(f"Analytics video composite '{video_in_path}' missing. Attempting to estimate duration from tracking logs.")

        # Fallback for duration calculation
        if duration_ms <= 0 and tracks_data:
            max_f = max((t["frame_index"] for t in tracks_data), default=0)
            duration_ms = int(((max_f + 1) / fps) * 1000)
            self.logger.info(f"Estimated playback duration from tracking logs: {duration_ms} ms at default {fps} FPS.")

        # 4. Extract Events
        print("\n>>> Step 1: Extracting events from tracking records...")
        extractor = EventExtractor(fps=fps)
        events = extractor.extract(tracks_data)
        
        events_output = os.path.join(output_dir, "events.json")
        serialized_events = [e.to_dict() for e in events]
        try:
            with open(events_output, 'w', encoding='utf-8') as f:
                json.dump(serialized_events, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Successfully archived {len(events)} events into: {events_output}")
        except Exception as e:
            self.logger.error(f"Failed to write events JSON: {e}")

        # 5. Generate Commentary Script
        print(">>> Step 2: Generating Vietnamese commentary script...")
        gen_cfg = self.config.commentary
        script_gen = ScriptGenerator(
            min_gap_seconds=gen_cfg.min_gap_seconds,
            max_events_per_minute=gen_cfg.max_events_per_minute,
            min_priority=gen_cfg.min_priority
        )
        script_items = script_gen.generate(events)
        
        script_output = os.path.join(output_dir, "commentary_script.json")
        try:
            with open(script_output, 'w', encoding='utf-8') as f:
                json.dump(script_items, f, ensure_ascii=False, indent=2)
            spoken_cnt = sum(1 for i in script_items if i["should_speak"])
            self.logger.info(f"Script compilation completed ({spoken_cnt}/{len(script_items)} voiced). Saved to: {script_output}")
        except Exception as e:
            self.logger.error(f"Failed to save script JSON: {e}")

        # 6. Audio Generation Flow
        audio_dir = os.path.join(output_dir, "commentary_audio")
        full_mp3_path = os.path.join(output_dir, "commentary_full.mp3")
        
        if self.config.tts.enabled:
            print("\n>>> Step 3: Invoking Edge-TTS Vietnamese speech synthesis...")
            os.makedirs(audio_dir, exist_ok=True)
            
            tts_cfg = self.config.tts
            tts_engine = EdgeVietnameseTTS(
                voice=tts_cfg.voice,
                rate=tts_cfg.rate,
                volume=tts_cfg.volume,
                pitch=tts_cfg.pitch
            )
            
            synthesized_count = 0
            for idx, item in enumerate(script_items):
                if not item["should_speak"]:
                    continue
                
                ev_id = item["event_id"]
                out_path = os.path.join(audio_dir, f"{ev_id}.mp3")
                
                try:
                    print(f"  [{idx+1}/{len(script_items)}] Synthesizing audio for {ev_id}: \"{item['text']}\"")
                    tts_engine.synthesize(
                        text=item["text"],
                        output_path=out_path,
                        emotion=item["emotion"],
                        speed=item["speed"]
                    )
                    synthesized_count += 1
                except Exception as e:
                    # DO NOT CRASH ON TTS FAILURES. Just report warning.
                    print(f"[!] WARNING: Failed synthesizing {ev_id}: {e}")
            
            print(f"\n>>> Generated {synthesized_count} individual event audio clips.")
            
            # Mix into master audio file
            print(">>> Step 4: Mixing clips onto master timeline track...")
            audio_cfg = self.config.audio
            mixer = CommentaryAudioMixer(commentary_volume_db=audio_cfg.commentary_volume_db)
            mixer.build_timeline(
                commentary_items=script_items,
                duration_ms=duration_ms,
                audio_dir=audio_dir,
                output_path=full_mp3_path
            )
            
            # 7. Video-Audio Multiplexing (Muxing)
            video_out_final = os.path.join(output_dir, "video_with_commentary.mp4")
            if video_found:
                print(">>> Step 5: Packaging audio timeline into final video composite...")
                mux_success = mixer.mux_audio_video(
                    video_path=video_in_path,
                    audio_path=full_mp3_path,
                    output_path=video_out_final
                )
                if mux_success:
                    print(f"\n>>> COMPLETION SUCCESS!")
                    print(f"Saved composite video to: {video_out_final}")
                else:
                    print("\n>>> Finished text and audio creation.")
                    print(f"Saved master audio timeline to: {full_mp3_path}")
                    print("Video packing omitted due to packaging warning.")
            else:
                print("\n>>> Finished text and audio creation.")
                print(f"Saved master audio timeline to: {full_mp3_path}")
                print("Source video composite missing; skipping packaging.")
        else:
            print("\n>>> Text compilation finished (TTS speech synthesis option disabled in configuration).")
        
        print("------------------------------------------------")
        print("Finished automated commentary module.")
