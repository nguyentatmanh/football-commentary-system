import json
import os
from typing import List, Dict, Any
from football_ai.detection.detection_result import DetectionResult

class ResultWriter:
    """
    Provides functionality to dump model and pipeline state results to JSON files.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_json(self, filename: str, data: Any):
        """Dumps data into JSON in the output directory."""
        file_path = os.path.join(self.output_dir, filename)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def save_detections(self, detections: List[DetectionResult], filename: str = "detections.json"):
        """Serializes list of DetectionResults to a JSON list."""
        serialized_data = [det.to_dict() for det in detections]
        self.write_json(filename, serialized_data)
