from abc import ABC, abstractmethod
from typing import List
import numpy as np
from football_ai.detection.detection_result import DetectionResult

class Detector(ABC):
    """
    Abstract Interface for Object Detection.
    Allows plugging in different model backends (e.g., YOLOv8, Faster R-CNN).
    """
    @abstractmethod
    def detect_frame(self, frame: np.ndarray, frame_index: int) -> List[DetectionResult]:
        """
        Run detection on a single video frame.

        Args:
            frame: Image matrix (numpy BGR array).
            frame_index: Index of the active frame in the video stream.

        Returns:
            A list of DetectionResult instances.
        """
        pass
