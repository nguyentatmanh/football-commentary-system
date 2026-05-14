import os
import shutil
import subprocess
import logging
from typing import List, Dict, Any
from pydub import AudioSegment

class CommentaryAudioMixer:
    """
    Uses pydub to composite event audio clips onto a continuous timelines,
    and invokes ffmpeg subprocesses to mux output tracks into final videos.
    """
    def __init__(
        self,
        commentary_volume_db: float = 0.0
    ):
        self.commentary_volume_db = commentary_volume_db
        self.logger = logging.getLogger(__name__)

    def build_timeline(
        self,
        commentary_items: List[Dict[str, Any]],
        duration_ms: int,
        audio_dir: str,
        output_path: str
    ) -> None:
        """
        Combines discrete audio files for flagged events onto a master timeline.
        Exports the finalized audio stream to disk.
        """
        if not commentary_items:
            self.logger.warning("No commentary items provided. Generating silent timeline.")
            # Minimum 1000ms if duration is tiny
            safe_duration = max(1000, duration_ms)
            silent = AudioSegment.silent(duration=safe_duration)
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            silent.export(output_path, format="mp3")
            return

        # Enforce a safe non-zero duration limit
        safe_duration = max(1000, duration_ms)
        
        # Check if timeline duration needs to expand past video length to accommodate late speech
        max_end_ms = 0
        for item in commentary_items:
            if not item.get("should_speak", False):
                continue
            ev_id = item["event_id"]
            ev_path = os.path.join(audio_dir, f"{ev_id}.mp3")
            if os.path.exists(ev_path):
                try:
                    audio_seg = AudioSegment.from_file(ev_path)
                    clip_end_ms = int(item["start_time"] * 1000) + len(audio_seg)
                    if clip_end_ms > max_end_ms:
                        max_end_ms = clip_end_ms
                except Exception:
                    pass
                    
        timeline_duration = max(safe_duration, max_end_ms)
        
        # Create blank silent base
        master_timeline = AudioSegment.silent(duration=timeline_duration)

        overlays_applied = 0
        for item in commentary_items:
            if not item.get("should_speak", False):
                continue

            ev_id = item["event_id"]
            ev_path = os.path.join(audio_dir, f"{ev_id}.mp3")
            
            if not os.path.exists(ev_path):
                self.logger.warning(f"Audio file missing for event '{ev_id}' at path: {ev_path}")
                continue

            try:
                audio_seg = AudioSegment.from_file(ev_path)
                
                # Apply volume adjustments if configured
                if self.commentary_volume_db != 0.0:
                    audio_seg = audio_seg.apply_gain(self.commentary_volume_db)
                    
                position_ms = int(item["start_time"] * 1000)
                
                master_timeline = master_timeline.overlay(audio_seg, position=position_ms)
                overlays_applied += 1
            except Exception as e:
                self.logger.error(f"Failed to read or overlay {ev_id} audio: {e}")

        # Ensure parent dirs
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        master_timeline.export(output_path, format="mp3")
        self.logger.info(f"Successfully exported mixed audio timeline. Applied {overlays_applied} clips.")

    def mux_audio_video(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """
        Combines MP4 video stream and MP3 audio track into finished video file.
        Validates the presence of local ffmpeg executable gracefully.
        """
        if not os.path.exists(video_path):
            self.logger.warning(f"Input video missing at '{video_path}'. Cannot perform muxing.")
            return False

        if not os.path.exists(audio_path):
            self.logger.warning(f"Input audio timeline missing at '{audio_path}'. Cannot mux.")
            return False

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            self.logger.warning("\n" + "="*60 +
                                "\nWARNING: 'ffmpeg' is NOT installed or not present in PATH." +
                                "\nSkipping audio-video composite creation." +
                                "\nTo generate video composite files, please install ffmpeg first." +
                                "\n" + "="*60)
            return False

        # Call FFmpeg subprocess
        # Maps the first video input track and second audio input track
        # Safe fallback audio codec: aac
        cmd = [
            ffmpeg_bin,
            "-y",                   # Overwrite existing
            "-i", video_path,       # Input 0: Video
            "-i", audio_path,       # Input 1: Audio
            "-c:v", "copy",         # Direct copy video stream (fast)
            "-c:a", "aac",          # Convert to standard AAC
            "-shortest",            # Stop at shorter stream length
            "-map", "0:v:0",        # Select first video
            "-map", "1:a:0",        # Select first audio
            output_path
        ]
        
        try:
            self.logger.info("Muxing audio and video via FFmpeg...")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.logger.info(f"Completed video composite muxing: {output_path}")
            return True
        except subprocess.CalledProcessError as cpe:
            self.logger.error(f"FFmpeg process failed with return code {cpe.returncode}")
            self.logger.debug(f"FFmpeg STDERR: {cpe.stderr.decode('utf-8', errors='ignore')}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to run FFmpeg command: {e}")
            return False
