"""
Text-to-Speech Commentary Manager.

This module manages the asynchronous queuing and processing of match events into
synthesized speech audio files, and handles the FFmpeg mixing required to
integrate the audio overlay smoothly into the broadcast pipeline.
"""

import os
import subprocess
import tempfile
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class TextToSpeechManager:
    """
    Handles generation of voice commentary and multiplexes it into an audio timeline.
    """

    def __init__(self, language: str = "en"):
        """
        Initializes the TTS engine.

        Args:
            language (str): The language code for speech synthesis.
        """
        self.language = language
        self.events_queue = []

    def queue_event(self, event: Dict[str, Any]):
        """
        Pushes a new broadcast event into the processing queue.
        """
        self.events_queue.append(event)

    def _synthesize_clip(self, text: str, output_wav: str) -> float:
        """
        Synthesizes speech using edge-tts and returns the duration in seconds.
        Converts the output to a standard WAV format using FFmpeg.
        """
        temp_mp3 = output_wav.replace(".wav", ".mp3")
        
        # We use a great British male sports voice: en-GB-RyanNeural
        subprocess.run(
            ["edge-tts", "--text", text, "--voice", "en-GB-RyanNeural", "--write-media", temp_mp3],
            capture_output=True,
            check=True
        )

        # Convert to WAV
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_mp3, output_wav],
            capture_output=True,
            check=True
        )
        os.remove(temp_mp3)

        # Probe duration
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", output_wav
            ],
            capture_output=True,
            text=True
        )
        try:
            return float(probe.stdout.strip())
        except ValueError:
            return 2.0

    def build_broadcast_audio(self, total_duration_secs: float, output_path: str):
        """
        Compiles the queued events into a single audio timeline.
        
        Args:
            total_duration_secs (float): Duration of the base silent track.
            output_path (str): File path to save the mixed audio.
        """
        logger.info(f"Generating TTS broadcast mix for {len(self.events_queue)} events...")
        
        with tempfile.TemporaryDirectory() as staging_dir:
            base_track = os.path.join(staging_dir, "silence_base.wav")
            
            # Generate silence template
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", 
                 "-t", str(total_duration_secs), base_track],
                capture_output=True, check=True
            )

            generated_clips = []
            cursor_time = 0.0

            for idx, event in enumerate(self.events_queue):
                timestamp = event["timestamp"]
                text_content = event["text"]

                # Prevent overlapping clips
                if timestamp < cursor_time:
                    continue

                clip_path = os.path.join(staging_dir, f"segment_{idx}.wav")
                try:
                    duration = self._synthesize_clip(text_content, clip_path)
                    generated_clips.append((timestamp, clip_path))
                    cursor_time = timestamp + duration
                    logger.debug(f"Synthesized: [{timestamp}s] {text_content}")
                except Exception as ex:
                    logger.error(f"TTS Synthesis Failed for '{text_content}': {ex}")

            if not generated_clips:
                subprocess.run(["cp", base_track, output_path], check=True)
                return

            # Construct FFmpeg mix filter graph
            input_args = ["-i", base_track]
            for _, path in generated_clips:
                input_args.extend(["-i", path])

            filter_complex = ""
            mix_tags = "[0:a]"

            for idx, (ts, _) in enumerate(generated_clips):
                # Apply a -0.8s offset to sync the commentary with the visual action peak
                adjusted_ts = max(0.0, ts - 0.8)
                delay_ms = int(adjusted_ts * 1000)
                filter_complex += f"[{idx+1}:a]adelay={delay_ms}|{delay_ms}[delay{idx}];"
                mix_tags += f"[delay{idx}]"

            # Combine delayed tracks over the base silence track
            total_inputs = len(generated_clips) + 1
            filter_complex += f"{mix_tags}amix=inputs={total_inputs}:duration=first:dropout_transition=0:normalize=0[mixed_out]"

            subprocess.run(
                ["ffmpeg", "-y"] + input_args + ["-filter_complex", filter_complex, "-map", "[mixed_out]", output_path],
                capture_output=True,
                check=True
            )

    def attach_audio_to_video(self, video_path: str, audio_path: str, output_path: str):
        """
        Multiplexes the synthesized audio track into the final output video.
        """
        logger.info("Merging synthesized broadcast audio with video...")
        
        result = subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path, "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
            "-shortest", output_path
        ], capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Failed to merge audio and video: {result.stderr}")
            raise RuntimeError("FFmpeg multiplexing encountered an error.")
