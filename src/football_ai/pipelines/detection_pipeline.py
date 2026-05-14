import os
import sys
from tqdm import tqdm
from football_ai.config.schema import SystemConfig
from football_ai.io.video_reader import VideoReader
from football_ai.io.result_writer import ResultWriter
from football_ai.detection.football_detector import FootballDetector

class DetectionPipeline:
    """
    Simple orchestrator that reads video frames, runs YOLO detection,
    and saves frame-by-frame raw detection dumps to a JSON file.
    """
    def __init__(self, config: SystemConfig):
        self.config = config
        self.detector = FootballDetector(config.detection, device=config.device)
        self.writer = ResultWriter(config.video.output_dir)

    def run(self, max_frames: int = -1):
        """
        Execute the detection pass.
        
        Args:
            max_frames: Optional limit to process only N first frames. 
                        If -1, process entire video.
        """
        print(f"Opening input video: {self.config.video.input_path}")
        reader = VideoReader(self.config.video.input_path)
        
        all_detections = []
        total_frames_to_run = reader.total_frames
        if max_frames > 0:
            total_frames_to_run = min(reader.total_frames, max_frames)
            
        print(f"Running detection for {total_frames_to_run} frames...")
        
        try:
            with tqdm(total=total_frames_to_run, desc="Detections") as pbar:
                for idx, frame in reader.read_frames():
                    if 0 < max_frames <= idx:
                        break
                        
                    frame_detections = self.detector.detect_frame(frame, frame_index=idx)
                    all_detections.extend(frame_detections)
                    pbar.update(1)
        finally:
            reader.release()
            
        output_path = os.path.join(self.config.video.output_dir, "detections.json")
        print(f"\nFinished detection. Total instances: {len(all_detections)}")
        self.writer.save_detections(all_detections)
        print(f"Saved json reports to: {output_path}")
