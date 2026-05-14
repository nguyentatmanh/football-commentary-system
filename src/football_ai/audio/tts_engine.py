import os
import asyncio
import re
import logging
import edge_tts

class EdgeVietnameseTTS:
    """
    Wraps the edge-tts backend for generating Vietnamese speech samples.
    Performs rates tuning based on contextual speed parameters.
    """
    def __init__(
        self,
        voice: str = "vi-VN-NamMinhNeural",
        rate: str = "+10%",
        volume: str = "+0%",
        pitch: str = "+0Hz"
    ):
        self.voice = voice
        self.default_rate = rate
        self.volume = volume
        self.pitch = pitch
        self.logger = logging.getLogger(__name__)

    def _resolve_rate(self, emotion: str, speed: str) -> str:
        """Dynamically adjusts speed rate based on emotion and requested pace."""
        # Faster pace for excitement/fast speed
        if speed == "fast" or emotion in ["excited", "urgent"]:
            return "+25%"
        elif speed == "slow":
            return "-5%"
        return self.default_rate

    async def _synthesize_async(self, text: str, output_path: str, rate: str):
        """Internal async executor for edge-tts communicate API."""
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=rate,
            volume=self.volume,
            pitch=self.pitch
        )
        await communicate.save(output_path)

    def synthesize(self, text: str, output_path: str, emotion: str = "neutral", speed: str = "normal") -> None:
        """
        Synthesizes input text and saves output mp3 synchronously.
        Creates parent directories automatically.
        """
        if not text:
            self.logger.warning("Empty text submitted to TTS engine. Skipping synthesis.")
            return

        # Create parent output directories
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        rate = self._resolve_rate(emotion, speed)
        
        try:
            # Run asyncio loop synchronously
            asyncio.run(self._synthesize_async(text, output_path, rate))
        except Exception as e:
            self.logger.error(f"Edge-TTS synthesis failed for file '{output_path}': {e}")
            raise RuntimeError(f"TTS Synthesis failure: {e}") from e
