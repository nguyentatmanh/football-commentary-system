import os
import cv2
import numpy as np
from tqdm import tqdm
from typing import List

from football_ai.config.schema import SystemConfig
from football_ai.io.video_reader import VideoReader
from football_ai.io.result_writer import ResultWriter
from football_ai.detection.football_detector import FootballDetector
from football_ai.tracking.tracker import Tracker
from football_ai.classification.team_classifier import TeamClassifier
from football_ai.classification.referee_classifier import RefereeClassifier
from football_ai.classification.goalkeeper_classifier import GoalkeeperClassifier
from football_ai.field_mapping.pitch_keypoints import PitchKeypointDetector
from football_ai.field_mapping.radar_mapper import RadarMapper
from football_ai.visualization.annotator import Annotator
from football_ai.visualization.video_writer import VideoWriter
from football_ai.visualization.radar_visualizer import RadarVisualizer
from football_ai.classification.role_smoother import RoleSmoother
from football_ai.classification.role_override import RoleOverrideApplier
from football_ai.classification.track_debug_exporter import TrackDebugExporter

class FieldMappingPipeline:
    """
    The ultimate spatial orchestrator:
    1. Harvests strided crops to train team classifiers.
    2. Sequentially Tracks, Classifies, Estimates homographies, and Maps coordinates.
    3. Outputs mapped_tracks.json, standalone radar_video.mp4, and overlaid mapped_video.mp4.
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
        
        self.pitch_detector = PitchKeypointDetector(config.detection, device=config.device)
        self.radar_mapper = RadarMapper(max_homography_staleness=15)
        
        self.annotator = Annotator()
        self.radar_viz = RadarVisualizer()
        self.result_writer = ResultWriter(config.video.output_dir)

    def _get_player_crop(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return frame[y1:y2, x1:x2]

    def _overlay_radar(self, frame: np.ndarray, radar: np.ndarray, target_width: int = 350) -> np.ndarray:
        """Overlays the radar graphic elegantly into the bottom-right corner."""
        out = frame.copy()
        
        # Maintain aspect ratio
        rh, rw, _ = radar.shape
        scale = target_width / rw
        target_height = int(rh * scale)
        
        resized_radar = cv2.resize(radar, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        # Calculate corner indices with slight margin padding
        fh, fw, _ = out.shape
        margin = 30
        
        y1 = fh - target_height - margin
        y2 = fh - margin
        x1 = fw - target_width - margin
        x2 = fw - margin
        
        if y1 >= 0 and x1 >= 0:
            # Sophisticated alpha blend overlay
            alpha = 0.85
            sub = out[y1:y2, x1:x2]
            blended = cv2.addWeighted(resized_radar, alpha, sub, 1 - alpha, 0)
            
            # Thin white border around radar
            cv2.rectangle(blended, (0,0), (target_width-1, target_height-1), (200, 200, 200), 1)
            
            out[y1:y2, x1:x2] = blended
            
        return out

    def run(self, max_frames: int = -1):
        # PASS 1: Harvest Player Crops for Teams
        print("\n>>> PASS 1: Harvesting player crops for spatial classification...")
        reader = VideoReader(self.config.video.input_path)
        stride = self.config.classification.stride
        fitting_crops = []
        
        max_fitting_idx = reader.total_frames
        if max_frames > 0:
            max_fitting_idx = min(reader.total_frames, max_frames)

        try:
            for idx, frame in tqdm(reader.read_frames(), desc="Harvesting Crops", total=reader.total_frames):
                if idx >= max_fitting_idx:
                    break
                if idx % stride != 0:
                    continue
                detections = self.detector.detect_frame(frame, frame_index=idx)
                for det in detections:
                    if det.role == "player":
                        crop = self._get_player_crop(frame, det.bbox_xyxy)
                        if crop.size > 0:
                            fitting_crops.append(crop)
        finally:
            reader.release()

        if not fitting_crops:
            raise ValueError("No fitting crops harvested. Check detection thresholds.")
            
        self.team_classifier.fit(fitting_crops)

        # PASS 2: Sequential Mapping + Synthesis
        print("\n>>> PASS 2: Running end-to-end mapping, tracking, and synthesis...")
        reader = VideoReader(self.config.video.input_path)
        
        # Setup video writers
        radar_video_path = os.path.join(self.config.video.output_dir, "radar_video.mp4")
        mapped_video_path = os.path.join(self.config.video.output_dir, "mapped_video.mp4")
        
        # Draw dummy radar to determine radar frame resolution
        dummy_radar = self.radar_viz.generate_radar_image([])
        rh, rw, _ = dummy_radar.shape
        
        radar_writer = VideoWriter(radar_video_path, reader.fps, rw, rh)
        mapped_writer = VideoWriter(mapped_video_path, reader.fps, reader.width, reader.height)

        all_tracks = []
        total_mapped_count = 0
        total_frames_to_run = max_fitting_idx
        
        try:
            with tqdm(total=total_frames_to_run, desc="Mapping & Synthesizing") as pbar:
                for idx, frame in reader.read_frames():
                    if 0 < max_frames <= idx:
                        break
                        
                    # 1. Core Detection & Tracking
                    detections = self.detector.detect_frame(frame, frame_index=idx)
                    tracks = self.tracker.update(detections, frame_index=idx)
                    
                    # 2. Role Smoothing & Normalizations
                    current_crops = {}
                    for t in tracks:
                        if t.role != "ball":
                            crop = self._get_player_crop(frame, t.bbox_xyxy)
                            if crop.size > 0:
                                current_crops[t.track_id] = crop
                                if t.role == "referee":
                                    self.role_smoother.collect_referee_color_sample(crop, t.confidence)
                                    
                    tracks = self.role_smoother.smooth_roles(tracks, crops=current_crops)
                    # Initial overrides
                    tracks = self.role_override_applier.apply_overrides(tracks)
                    
                    tracks = self.referee_classifier.normalize_referees(tracks)
                    
                    player_tracks = [t for t in tracks if t.role == "player"]
                    if player_tracks:
                        player_crops = [self._get_player_crop(frame, t.bbox_xyxy) for t in player_tracks]
                        valid_idxs = [i for i, cr in enumerate(player_crops) if cr.size > 0]
                        if valid_idxs:
                            preds = self.team_classifier.predict([player_crops[i] for i in valid_idxs])
                            for j, v_idx in enumerate(valid_idxs):
                                player_tracks[v_idx].team_id = preds[j]
                                
                    tracks = self.goalkeeper_classifier.resolve_team_ids(tracks)
                    
                    # Final overrides safety
                    tracks = self.role_override_applier.apply_overrides(tracks)
                    
                    # Ingest crops for diagnostics
                    self.track_debug_exporter.ingest_frame_crops(tracks, current_crops)
                    
                    # 3. Pitch Landmarking & Projection Estimation
                    kps_res = self.pitch_detector.detect(frame, frame_index=idx)
                    self.radar_mapper.update_pitch_projection(kps_res)
                    
                    # 4. Coordinates Projection
                    tracks = self.radar_mapper.map_tracks(tracks)
                    
                    # Count mapped states
                    for t in tracks:
                        if t.pitch_xy is not None:
                            total_mapped_count += 1
                            
                    all_tracks.extend(tracks)
                    
                    # 5. Visual synthesis
                    # Tracked overlay frame
                    annotated_main = self.annotator.annotate_frame(frame, tracks)
                    # Top-down radar frame
                    radar_img = self.radar_viz.generate_radar_image(tracks)
                    # Superimpose mini-radar overlay onto main
                    final_composite = self._overlay_radar(annotated_main, radar_img)
                    
                    # Save frames to respective writers
                    radar_writer.write_frame(radar_img)
                    mapped_writer.write_frame(final_composite)
                    pbar.update(1)
        finally:
            reader.release()
            radar_writer.release()
            mapped_writer.release()

        # Export logs
        serialized = [t.to_dict() for t in all_tracks]
        self.result_writer.write_json("mapped_tracks.json", serialized)

        # Diagnostic Reports
        self.role_smoother.print_summary_and_save(self.config.video.output_dir)
        self.track_debug_exporter.export(self.config.video.output_dir)

        print(f"\n>>> Finished spatial mapping pipeline.")
        print(f"Total tracked instances: {len(all_tracks)}")
        print(f"Successfully mapped to 2D pitch_xy: {total_mapped_count} states")
        print(f"Saved 2D radar standalone to: {radar_video_path}")
        print(f"Saved premium composite to: {mapped_video_path}")
        print(f"Saved spatial coordinates log to: {os.path.join(self.config.video.output_dir, 'mapped_tracks.json')}")
