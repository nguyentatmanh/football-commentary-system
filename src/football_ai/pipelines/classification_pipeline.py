import os
import cv2
import numpy as np
from tqdm import tqdm
from typing import List
import supervision as sv

from football_ai.config.schema import SystemConfig
from football_ai.io.video_reader import VideoReader
from football_ai.io.result_writer import ResultWriter
from football_ai.detection.football_detector import FootballDetector
from football_ai.tracking.tracker import Tracker
from football_ai.classification.team_classifier import TeamClassifier
from football_ai.classification.referee_classifier import RefereeClassifier
from football_ai.classification.goalkeeper_classifier import GoalkeeperClassifier
from football_ai.visualization.annotator import Annotator
from football_ai.visualization.video_writer import VideoWriter
from football_ai.classification.role_smoother import RoleSmoother
from football_ai.classification.role_override import RoleOverrideApplier
from football_ai.classification.track_debug_exporter import TrackDebugExporter

class ClassificationPipeline:
    """
    A robust two-pass orchestrator:
    1. Strides through the video to gather player image crops and fit the TeamClassifier.
    2. Runs full sequential Inference + Tracking + Team/Ref/Goalkeeper classification + Synthesis.
    """
    def __init__(self, config: SystemConfig):
        self.config = config
        self.detector = FootballDetector(config.detection, device=config.device)
        self.tracker = Tracker(config.tracking)
        
        self.team_classifier = TeamClassifier(device=config.device)
        self.referee_classifier = RefereeClassifier()
        self.goalkeeper_classifier = GoalkeeperClassifier()
        self.role_smoother = RoleSmoother(config.role_smoothing)
        self.role_override_applier = RoleOverrideApplier(config.role_overrides)
        self.track_debug_exporter = TrackDebugExporter(enabled=config.debug_tracks)
        
        self.annotator = Annotator()
        self.result_writer = ResultWriter(config.video.output_dir)

    def _get_player_crop(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Crop an image based on [x1, y1, x2, y2] bounding box coordinate constraint."""
        x1, y1, x2, y2 = map(int, bbox)
        # Enforce image bounds
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return frame[y1:y2, x1:x2]

    def run(self, max_frames: int = -1):
        # PASS 1: Strided collection of player crops for Fitting
        print("\n>>> PASS 1: Gathering representative player crops for team fitting...")
        reader = VideoReader(self.config.video.input_path)
        
        stride = self.config.classification.stride
        fitting_crops = []
        
        # Limit fit processing to max_frames constraint if provided
        max_fitting_idx = reader.total_frames
        if max_frames > 0:
            max_fitting_idx = min(reader.total_frames, max_frames)

        try:
            # We skip frames according to stride configuration
            for idx, frame in tqdm(reader.read_frames(), desc="Harvesting Crops", total=reader.total_frames):
                if idx >= max_fitting_idx:
                    break
                if idx % stride != 0:
                    continue
                    
                detections = self.detector.detect_frame(frame, frame_index=idx)
                # Isolate players
                for det in detections:
                    if det.role == "player":
                        crop = self._get_player_crop(frame, det.bbox_xyxy)
                        if crop.size > 0:
                            fitting_crops.append(crop)
        finally:
            reader.release()

        if not fitting_crops:
            raise ValueError("Failed to find any player crops to train the team classifier! Is the threshold too high?")
            
        # Fit the Team Classifier
        self.team_classifier.fit(fitting_crops)

        # PASS 2: Full Sequential Processing
        print("\n>>> PASS 2: Sequentially classifying and rendering annotated output...")
        reader = VideoReader(self.config.video.input_path)
        video_out_path = os.path.join(self.config.video.output_dir, "classified_video.mp4")
        writer = VideoWriter(
            output_path=video_out_path,
            fps=reader.fps,
            width=reader.width,
            height=reader.height
        )

        all_tracks = []
        total_frames_to_run = max_fitting_idx
        
        try:
            with tqdm(total=total_frames_to_run, desc="Classifying") as pbar:
                for idx, frame in reader.read_frames():
                    if 0 < max_frames <= idx:
                        break
                        
                    # 1. Detection
                    detections = self.detector.detect_frame(frame, frame_index=idx)
                    
                    # 2. Multi-Object Tracking
                    tracks = self.tracker.update(detections, frame_index=idx)
                    
                    # 3. Optional Color Sampling & Role Smoothing
                    current_crops = {}
                    for t in tracks:
                        if t.role != "ball":
                            crop = self._get_player_crop(frame, t.bbox_xyxy)
                            if crop.size > 0:
                                current_crops[t.track_id] = crop
                                # Accumulate high confidence referee color data
                                if t.role == "referee":
                                    self.role_smoother.collect_referee_color_sample(crop, t.confidence)
                                    
                    tracks = self.role_smoother.smooth_roles(tracks, crops=current_crops)
                    # Apply Initial Manual Overrides
                    tracks = self.role_override_applier.apply_overrides(tracks)
                    
                    tracks = self.referee_classifier.normalize_referees(tracks)
                    
                    # 4. Team Classification for active players
                    player_tracks = [t for t in tracks if t.role == "player"]
                    if player_tracks:
                        player_crops = [self._get_player_crop(frame, t.bbox_xyxy) for t in player_tracks]
                        # Ensure no empty crops crash predictor
                        valid_indices = [i for i, cr in enumerate(player_crops) if cr.size > 0]
                        
                        if valid_indices:
                            valid_crops = [player_crops[i] for i in valid_indices]
                            pred_teams = self.team_classifier.predict(valid_crops)
                            
                            # Assign predictions back
                            for j, v_idx in enumerate(valid_indices):
                                player_tracks[v_idx].team_id = pred_teams[j]
                    
                    # 5. Goalkeeper Resolution (Spatial Centroid Proximity)
                    tracks = self.goalkeeper_classifier.resolve_team_ids(tracks)
                    
                    # Final Manual Override Safety Pass
                    tracks = self.role_override_applier.apply_overrides(tracks)
                    
                    # Buffer data into debug visual exporter
                    self.track_debug_exporter.ingest_frame_crops(tracks, current_crops)
                    
                    # 6. Render Overlays
                    annotated = self.annotator.annotate_frame(frame, tracks)
                    
                    # Write frame and log results
                    writer.write_frame(annotated)
                    all_tracks.extend(tracks)
                    pbar.update(1)
        finally:
            reader.release()
            writer.release()

        # Write output reports
        serialized = [t.to_dict() for t in all_tracks]
        self.result_writer.write_json("classified_tracks.json", serialized)

        # 7. Role Smoothing Diagnostic Console and Persistence
        self.role_smoother.print_summary_and_save(self.config.video.output_dir)
        
        # 8. Export Visual Audit Sheets & Summaries
        self.track_debug_exporter.export(self.config.video.output_dir)

        print(f"\n>>> Finished classification pipeline.")
        print(f"Saved annotated classified video to: {video_out_path}")
        print(f"Saved track database to: {os.path.join(self.config.video.output_dir, 'classified_tracks.json')}")
