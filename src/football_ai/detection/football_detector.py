import os
from typing import List, Dict
import numpy as np
from ultralytics import YOLO
from football_ai.detection.detector import Detector
from football_ai.detection.detection_result import DetectionResult
from football_ai.config.schema import DetectionConfig

class FootballDetector(Detector):
    """
    Implementation of Detector using Ultralytics YOLOv8 weights trained on football assets.
    """
    
    # Roboflow Sports Default Class Mapping for the combined model
    CLASS_MAP = {
        0: "ball",
        1: "goalkeeper",
        2: "player",
        3: "referee"
    }

    def __init__(self, config: DetectionConfig, device: str = "cpu"):
        """
        Initialize YOLOv8 detector.
        
        Args:
            config: DetectionConfig parameter settings.
            device: Torch execution device (cpu, cuda, or mps).
        """
        if not os.path.exists(config.player_model_path):
            raise FileNotFoundError(
                f"Model weights missing at: {config.player_model_path}. "
                "Please run 'python scripts/download_assets.py' first."
            )
            
        self.model = YOLO(config.player_model_path)
        self.model.to(device)
        self.confidence = config.confidence
        self.iou = config.iou
        self.imgsz = config.imgsz

    def detect_frame(self, frame: np.ndarray, frame_index: int) -> List[DetectionResult]:
        """
        Detect soccer players, goalkeepers, referee, and balls in a video frame.
        """
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False
        )
        
        detections = []
        if len(results) == 0:
            return detections
            
        result = results[0]
        boxes = result.boxes
        
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist() # Get [x1, y1, x2, y2]
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            
            # Normalize role/class name
            class_name = self.CLASS_MAP.get(cls_id, "unknown")
            role = class_name # For primary detections role = initial class assignment
            
            det = DetectionResult(
                frame_index=frame_index,
                class_id=cls_id,
                class_name=class_name,
                confidence=conf,
                bbox_xyxy=xyxy,
                role=role
            )
            detections.append(det)
            
        return detections
