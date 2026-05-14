import os
import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Dict

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

# New Analytics imports
from football_ai.analytics.movement_analyzer import MovementAnalyzer
from football_ai.analytics.heatmap_generator import HeatmapGenerator
from football_ai.analytics.possession_analyzer import PossessionAnalyzer
from football_ai.classification.role_smoother import RoleSmoother
from football_ai.classification.role_override import RoleOverrideApplier
from football_ai.classification.track_debug_exporter import TrackDebugExporter

class AnalyticsPipeline:
    """
    Comprehensive analytics runner:
    1. Runs spatial-mapping video analytics suite.
    2. Extracts trajectories, speed distributions, and distance counts.
    3. Formulates proximity ball possession timeline stats.
    4. Renders occupancy density heatmaps to PNGs.
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
        
        # Analytics engines will be initialized dynamically once reader parameters are read

    def _get_player_crop(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return frame[y1:y2, x1:x2]

    def _overlay_composite(self, frame: np.ndarray, radar: np.ndarray, possession_text: str = "") -> np.ndarray:
        """Overlays mini-radar overlay elegantly in the corner and injects status tags."""
        out = frame.copy()
        
        # Aspect-correct overlay
        target_width = 350
        rh, rw, _ = radar.shape
        scale = target_width / rw
        target_height = int(rh * scale)
        
        resized_radar = cv2.resize(radar, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        fh, fw, _ = out.shape
        margin = 30
        
        y1 = fh - target_height - margin
        y2 = fh - margin
        x1 = fw - target_width - margin
        x2 = fw - margin
        
        if y1 >= 0 and x1 >= 0:
            alpha = 0.85
            sub = out[y1:y2, x1:x2]
            blended = cv2.addWeighted(resized_radar, alpha, sub, 1 - alpha, 0)
            cv2.rectangle(blended, (0,0), (target_width-1, target_height-1), (200, 200, 200), 1)
            out[y1:y2, x1:x2] = blended
            
        # Add possession tag at top
        if possession_text:
            cv2.putText(
                out, 
                possession_text, 
                (50, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (0, 223, 252), # Cyan highlight
                3, 
                cv2.LINE_AA
            )
            
        return out

    def run(self, max_frames: int = -1):
        # Initialize Reader first to extract precise FPS
        reader = VideoReader(self.config.video.input_path)
        fps = float(reader.fps)
        reader.release()

        # Instantiate analytic components with correct FPS
        movement_analyzer = MovementAnalyzer(fps=fps)
        possession_analyzer = PossessionAnalyzer(grab_radius_m=2.5)
        heatmap_gen = HeatmapGenerator()

        # PASS 1: Fit Team Classifier
        print("\n>>> PASS 1: Harvesting player crops for spatial classification...")
        reader = VideoReader(self.config.video.input_path)
        stride = self.config.classification.stride
        fitting_crops = []
        max_fitting_idx = min(reader.total_frames, max_frames) if max_frames > 0 else reader.total_frames

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
            raise ValueError("No crops harvested. Ensure weights are active and video has players.")
            
        self.team_classifier.fit(fitting_crops)

        # PASS 2: Complete Analysis Run
        print("\n>>> PASS 2: Running end-to-end analysis processing...")
        reader = VideoReader(self.config.video.input_path)
        
        video_out_path = os.path.join(self.config.video.output_dir, "analytics_video.mp4")
        video_writer = VideoWriter(video_out_path, reader.fps, reader.width, reader.height)

        all_tracks = []
        
        # Dynamic grouped trajectory trackers for batch heatmap processing at pipeline finalization
        hmap_groups: Dict[str, List[List[float]]] = {
            "Team A": [],
            "Team B": [],
            "Referee": [],
            "Ball": [],
            "All Players": []
        }

        try:
            with tqdm(total=max_fitting_idx, desc="Processing Suite") as pbar:
                for idx, frame in reader.read_frames():
                    if 0 < max_frames <= idx:
                        break
                        
                    # 1. Pipeline execution
                    detections = self.detector.detect_frame(frame, frame_index=idx)
                    tracks = self.tracker.update(detections, frame_index=idx)
                    
                    # 2. Optional Color Sampling & Role Smoothing
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
                    
                    # Final safety overrides
                    tracks = self.role_override_applier.apply_overrides(tracks)
                    
                    # Visual buffer diagnostics
                    self.track_debug_exporter.ingest_frame_crops(tracks, current_crops)
                    
                    kps_res = self.pitch_detector.detect(frame, frame_index=idx)
                    self.radar_mapper.update_pitch_projection(kps_res)
                    
                    # 2. Pitch Projection
                    tracks = self.radar_mapper.map_tracks(tracks)
                    
                    # 3. Analytics Ingestion
                    movement_analyzer.process_frame_tracks(tracks)
                    possession_analyzer.process_frame(tracks, idx)
                    
                    # Populate heatmap arrays
                    for t in tracks:
                        if t.pitch_xy is not None:
                            pt = t.pitch_xy
                            if t.role == "ball":
                                hmap_groups["Ball"].append(pt)
                            elif t.role == "referee":
                                hmap_groups["Referee"].append(pt)
                            elif t.role == "player" or t.role == "goalkeeper":
                                hmap_groups["All Players"].append(pt)
                                if t.team_id == 0:
                                    hmap_groups["Team A"].append(pt)
                                elif t.team_id == 1:
                                    hmap_groups["Team B"].append(pt)

                    all_tracks.extend(tracks)
                    
                    # 4. Visualization composition
                    annotated_main = self.annotator.annotate_frame(frame, tracks)
                    radar_img = self.radar_viz.generate_radar_image(tracks)
                    
                    # Resolve running possession tag for broadcast HUD
                    latest_poss = possession_analyzer.timeline.get(idx, None)
                    poss_txt = ""
                    if latest_poss == 0:
                        poss_txt = "POSSESSION: TEAM A (PINK)"
                    elif latest_poss == 1:
                        poss_txt = "POSSESSION: TEAM B (AZURE)"
                    else:
                        poss_txt = "POSSESSION: CONTESTED / LOOSE"
                        
                    composite = self._overlay_composite(annotated_main, radar_img, poss_txt)
                    video_writer.write_frame(composite)
                    pbar.update(1)
        finally:
            reader.release()
            video_writer.release()

        # Pipeline Finalization
        print("\n>>> Compiling summaries & rendering heatmaps...")
        
        # Save outputs
        move_dict = movement_analyzer.get_summary_dict()
        self.result_writer.write_json("movement_summary.json", move_dict)
        
        poss_dict = possession_analyzer.compute_stats()
        self.result_writer.write_json("possession_summary.json", poss_dict)
        
        serialized_tracks = [t.to_dict() for t in all_tracks]
        self.result_writer.write_json("analytics_tracks.json", serialized_tracks)
        
        # Batch render individual heatmaps for top players (stable IDs with longest distance)
        top_players = [m for m in move_dict if m["role"] in ["player", "goalkeeper"]][:5] # Top 5 active
        for tp in top_players:
            tid = tp["track_id"]
            # Retrieve trajectory from memory object
            mem_stats = movement_analyzer.stats.get(tid)
            if mem_stats and len(mem_stats.trajectory) > 5:
                traj_pts = [coord for fidx, coord in mem_stats.trajectory]
                hmap_groups[f"Player {tid}"] = traj_pts

        # Write PNGs
        heatmap_gen.save_heatmaps(self.config.video.output_dir, hmap_groups)

        # Diagnostic Reports
        self.role_smoother.print_summary_and_save(self.config.video.output_dir)
        self.track_debug_exporter.export(self.config.video.output_dir)
        
        print(f"\n>>> Finished analytics pipeline.")
        print(f"Saved movement summary to: movement_summary.json")
        print(f"Saved possession statistics to: possession_summary.json")
        print(f"Saved interactive analytics log to: analytics_tracks.json")
        print(f"Rendered heatmaps to: {os.path.join(self.config.video.output_dir, 'heatmaps/')}")
        print(f"Saved composite overlay video to: {video_out_path}")
