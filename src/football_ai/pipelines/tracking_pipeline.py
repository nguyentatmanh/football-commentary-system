import os
from tqdm import tqdm
from football_ai.config.schema import SystemConfig
from football_ai.io.video_reader import VideoReader
from football_ai.io.result_writer import ResultWriter
from football_ai.detection.football_detector import FootballDetector
from football_ai.tracking.tracker import Tracker
from football_ai.visualization.annotator import Annotator
from football_ai.visualization.video_writer import VideoWriter

class TrackingPipeline:
    """
    Runs unified inference, multi-object association tracking, and annotated 
    video synthesis. Exits with a tracks.json summary and annotated_video.mp4.
    """
    def __init__(self, config: SystemConfig):
        self.config = config
        self.detector = FootballDetector(config.detection, device=config.device)
        self.tracker = Tracker(config.tracking)
        self.annotator = Annotator()
        self.result_writer = ResultWriter(config.video.output_dir)

    def run(self, max_frames: int = -1):
        """
        Execute tracking and video generation pass.
        """
        print(f"Opening video sequence: {self.config.video.input_path}")
        reader = VideoReader(self.config.video.input_path)
        
        # Prepare video writer
        video_out_path = os.path.join(self.config.video.output_dir, "annotated_video.mp4")
        writer = VideoWriter(
            output_path=video_out_path,
            fps=reader.fps,
            width=reader.width,
            height=reader.height
        )

        all_tracks = []
        total_frames_to_run = reader.total_frames
        if max_frames > 0:
            total_frames_to_run = min(reader.total_frames, max_frames)
            
        print(f"Running tracker and visualizer for {total_frames_to_run} frames...")
        
        unique_track_ids = set()
        
        try:
            with tqdm(total=total_frames_to_run, desc="Tracking") as pbar:
                for idx, frame in reader.read_frames():
                    if 0 < max_frames <= idx:
                        break
                        
                    # 1. Object Detection
                    detections = self.detector.detect_frame(frame, frame_index=idx)
                    
                    # 2. Multi-Object Tracking
                    tracks = self.tracker.update(detections, frame_index=idx)
                    
                    # Store identifiers
                    for t in tracks:
                        all_tracks.append(t)
                        if t.track_id != -1: # Exclude constant ball ID from unique counter
                            unique_track_ids.add(t.track_id)
                            
                    # 3. Video Frame Annotation
                    annotated = self.annotator.annotate_frame(frame, tracks)
                    
                    # 4. Video Encoding
                    writer.write_frame(annotated)
                    pbar.update(1)
        finally:
            reader.release()
            writer.release()

        # 5. Output Serializations
        serialized_tracks = [t.to_dict() for t in all_tracks]
        self.result_writer.write_json("tracks.json", serialized_tracks)
        
        print(f"\nFinished tracking pipeline.")
        print(f"Total tracked instances: {len(all_tracks)}")
        print(f"Total unique human tracks: {len(unique_track_ids)}")
        print(f"Saved annotated video to: {video_out_path}")
        print(f"Saved track database to: {os.path.join(self.config.video.output_dir, 'tracks.json')}")
