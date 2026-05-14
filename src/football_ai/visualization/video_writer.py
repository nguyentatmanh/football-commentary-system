import os
import cv2
import numpy as np

class VideoWriter:
    """
    Saves sequences of NumPy image frames into an encoded video file using OpenCV.
    """
    def __init__(self, output_path: str, fps: float, width: int, height: int):
        """
        Initialize OpenCV video writer container.
        """
        # Ensure target folder exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self.output_path = output_path
        # mp4v codec is highly robust across systems
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        self.writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )
        if not self.writer.isOpened():
            raise IOError(f"Failed to open VideoWriter for path: {output_path}")

    def write_frame(self, frame: np.ndarray):
        """Write an annotated NumPy BGR frame to disk."""
        self.writer.write(frame)

    def release(self):
        """Finalize video encoding and close file descriptor."""
        if self.writer.isOpened():
            self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
