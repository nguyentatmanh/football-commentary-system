import os
import numpy as np
import supervision as sv
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from ultralytics import YOLO
from football_ai.config.schema import DetectionConfig

@dataclass
class PitchKeypointResult:
    frame_index: int
    keypoints_xy: Optional[List[List[float]]] = None # Shape: (32, 2)
    confidence: Optional[List[float]] = None        # Shape: (32,)
    success: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class PitchKeypointDetector:
    """
    Handles loading the specialized YOLO keypoint model and extracting
    predefined keypoint coordinates from video frames.
    """
    def __init__(self, config: DetectionConfig, device: str = "cpu"):
        if not os.path.exists(config.pitch_model_path):
            raise FileNotFoundError(
                f"Pitch keypoint model weights missing at: {config.pitch_model_path}. "
                "Please ensure assets are downloaded."
            )
            
        self.model = YOLO(config.pitch_model_path)
        self.model.to(device)
        self.conf_threshold = config.confidence

    def detect(self, frame: np.ndarray, frame_index: int) -> PitchKeypointResult:
        """
        Execute pose detection and parse the pitch boundary landmarks.
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        if not results:
            return PitchKeypointResult(frame_index=frame_index, success=False)
            
        result = results[0]
        # In YOLO pose, if keypoints are found
        if not hasattr(result, 'keypoints') or result.keypoints is None or len(result.keypoints) == 0:
            return PitchKeypointResult(frame_index=frame_index, success=False)
            
        # Wrap with supervision or extract directly from ultralytics
        # Ultralytics keypoints tensors: result.keypoints.xy shape (num_detections, num_points, 2)
        # Usually we only have 1 pitch detection, we take index 0.
        xy = result.keypoints.xy[0].cpu().numpy() # (32, 2)
        conf = result.keypoints.conf[0].cpu().numpy() if result.keypoints.conf is not None else None

        # Validation check: do we have any non-zero detected points?
        # Valid keypoints are non-zero.
        valid_mask = (xy[:, 0] > 1.0) & (xy[:, 1] > 1.0)
        num_detected = np.sum(valid_mask)

        # A homography estimation requires at least 4 points
        if num_detected < 4:
            return PitchKeypointResult(
                frame_index=frame_index,
                keypoints_xy=xy.tolist(),
                confidence=conf.tolist() if conf is not None else None,
                success=False
            )

        return PitchKeypointResult(
            frame_index=frame_index,
            keypoints_xy=xy.tolist(),
            confidence=conf.tolist() if conf is not None else None,
            success=True
        )
