import numpy as np
from typing import List, Optional, Tuple

import football_ai.utils.paths # Instantiates sports path injection
from sports.configs.soccer import SoccerPitchConfiguration
from football_ai.tracking.track_state import TrackState
from football_ai.field_mapping.homography import HomographyEstimator
from football_ai.field_mapping.pitch_keypoints import PitchKeypointResult

class RadarMapper:
    """
    Maps localized pixel positions into standardized meter coordinates 
    (X: 0 to 105m, Y: 0 to 68m) by applying perspective transformations.
    """
    def __init__(self, max_homography_staleness: int = 15, position_smoothing_factor: float = 0.8):
        # Initialize configured standard 105x68 meter pitch in cm
        self.pitch_config = SoccerPitchConfiguration(length=10500, width=6800)
        self.estimator = HomographyEstimator(smoothing_factor=0.85)
        
        # Staleness caching variables
        self.last_valid_matrix = None
        self.frames_since_valid = 0
        self.max_homography_staleness = max_homography_staleness
        
        # Trajectory smoothing buffer
        self.track_ema = {}
        self.position_smoothing_factor = position_smoothing_factor

    def _get_anchor(self, track: TrackState) -> Tuple[float, float]:
        """Retrieve logical anchor position. Ball: center. Human: bottom center."""
        x1, y1, x2, y2 = track.bbox_xyxy
        if track.role == "ball":
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0
        # Shadow Compensation: pull anchor 5% up from bottom edge
        h = y2 - y1
        return (x1 + x2) / 2.0, y2 - (h * 0.05)

    def update_pitch_projection(self, keypoint_res: PitchKeypointResult) -> bool:
        """
        Updates the homography estimator matrix if valid keypoints are presented.
        Otherwise, degrades stale estimator or falls back to cached historical matrix.
        """
        if keypoint_res.success and keypoint_res.keypoints_xy is not None:
            kps_arr = np.array(keypoint_res.keypoints_xy, dtype=np.float32)
            
            # Confidence Filter: exclude unsure AI guesses
            if keypoint_res.confidence is not None:
                conf_arr = np.array(keypoint_res.confidence)
                conf_mask = conf_arr > 0.5
            else:
                conf_mask = np.ones(len(kps_arr), dtype=bool)

            # Create boolean mask filtering out non-detected zero-anchors (>1 px)
            mask = (kps_arr[:, 0] > 1.0) & (kps_arr[:, 1] > 1.0) & conf_mask
            
            src_points = kps_arr[mask]
            dst_points = np.array(self.pitch_config.vertices, dtype=np.float32)[mask]
            
            if len(src_points) >= 4:
                success = self.estimator.fit(src_points, dst_points)
                if success:
                    self.last_valid_matrix = self.estimator.matrix.copy()
                    self.frames_since_valid = 0
                    return True

        # If we failed, try to reuse the last valid projection if not too old
        if self.last_valid_matrix is not None and self.frames_since_valid < self.max_homography_staleness:
            self.estimator.matrix = self.last_valid_matrix.copy()
            self.frames_since_valid += 1
            return True

        # Evaporate the projection matrices if too stale
        self.estimator.matrix = None
        return False

    def map_tracks(self, tracks: List[TrackState]) -> List[TrackState]:
        """
        Applies the current perspective projection to standard pixel anchors,
        saving normalized meters [x, y] into TrackState.pitch_xy.
        """
        if not self.estimator.is_ready() or not tracks:
            # Clean assignment to None if mapping failed/not ready
            for t in tracks:
                t.pitch_xy = None
            return tracks

        # 1. Gather pixel anchor points
        anchors = np.array([self._get_anchor(t) for t in tracks], dtype=np.float32)
        
        # 2. Bulk project coordinates
        projected = self.estimator.transform_points(anchors)
        
        if projected is None:
            for t in tracks:
                t.pitch_xy = None
            return tracks

        # 3. Scale centimeters down to Meters [x / 100.0]
        meter_points = projected / 100.0
        
        # Cleanup stale tracks from EMA dictionary
        current_track_ids = {t.track_id for t in tracks}
        stale_ids = set(self.track_ema.keys()) - current_track_ids
        for sid in stale_ids:
            del self.track_ema[sid]

        is_drifting = self.frames_since_valid > 5

        for idx, t in enumerate(tracks):
            mx, my = meter_points[idx]
            
            # Anti-Drift Freezing: camera is moving but homography is lost
            if is_drifting and t.track_id in self.track_ema:
                smx, smy = self.track_ema[t.track_id]
                t.pitch_xy = [float(smx), float(smy)]
                continue

            # Out-of-Bounds Clipping (Pitch config is 105x68m, margin +-10m)
            if np.isnan(mx) or np.isinf(mx) or np.isnan(my) or np.isinf(my) or mx < -10 or mx > 115 or my < -10 or my > 78:
                t.pitch_xy = None
                continue

            # Trajectory smoothing logic
            if t.track_id in self.track_ema:
                old_mx, old_my = self.track_ema[t.track_id]
                
                # Ball Teleport Filter: Elevated ball shoots 2D projection away
                dist = np.sqrt((mx - old_mx)**2 + (my - old_my)**2)
                if t.role == "ball" and dist > 5.0:
                    smx, smy = old_mx, old_my
                else:
                    # Blend old and new
                    smx = self.position_smoothing_factor * old_mx + (1.0 - self.position_smoothing_factor) * mx
                    smy = self.position_smoothing_factor * old_my + (1.0 - self.position_smoothing_factor) * my
            else:
                smx, smy = float(mx), float(my)
            
            self.track_ema[t.track_id] = [smx, smy]
            t.pitch_xy = [float(smx), float(smy)]

        return tracks
