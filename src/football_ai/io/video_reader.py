import os
import cv2
import numpy as np
from typing import Generator, Tuple

class VideoReader:
    """
    Handles opening and yielding frames sequentially from a video source file.
    """
    def __init__(self, video_path: str):
        if not os.path.exists(video_path):
            raise FileNotFoundError(
                f"Video file missing at: {video_path}. "
                "Please download dependencies or provide valid video source."
            )
            
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Yield frame indices and frame arrays."""
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1
            
    def release(self):
        """Close OpenCV handles."""
        if self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
