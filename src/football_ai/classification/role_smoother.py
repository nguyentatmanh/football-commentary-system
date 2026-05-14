import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from football_ai.tracking.track_state import TrackState
from football_ai.config.schema import RoleSmoothingConfig

class RoleSmoother:
    """
    Enhances tracking role consistency by applying STRICT track-level majority voting
    and temporal locking. Prevents over-classifying players as referees.
    """
    def __init__(self, config: RoleSmoothingConfig):
        self.config = config
        
        # Purely historical state before any smoothing takes place
        # Structure: { 
        #   "raw_roles": [], 
        #   "raw_referee_confidences": [],
        #   "locked_referee": bool,
        #   "final_role": str,
        #   "lock_reason": str
        # }
        self.track_history: Dict[int, Dict[str, Any]] = {}
        
        # Optional Color prototype (HSV color mean) for the referee jersey
        self._referee_hsv_samples: List[np.ndarray] = []
        self.referee_prototype_hsv: Optional[np.ndarray] = None

    def _get_dominant_hsv(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Extracts average HSV color from the center 50% of the crop."""
        if crop is None or crop.size == 0:
            return None
        h, w, _ = crop.shape
        ch1, ch2 = int(h * 0.2), int(h * 0.6)
        cw1, cw2 = int(w * 0.25), int(w * 0.75)
        center_crop = crop[ch1:ch2, cw1:cw2]
        if center_crop.size == 0:
            return None
        hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
        return np.mean(hsv, axis=(0, 1))

    def collect_referee_color_sample(self, crop: np.ndarray, conf: float):
        """Harvests jersey color vectors if enabled."""
        if not self.config.use_color_referee_reid or conf < 0.9:
            return
        color_v = self._get_dominant_hsv(crop)
        if color_v is not None:
            self._referee_hsv_samples.append(color_v)
            self.referee_prototype_hsv = np.mean(self._referee_hsv_samples, axis=0)

    def check_color_reid(self, crop: np.ndarray, threshold: float = 30.0) -> bool:
        if not self.config.use_color_referee_reid or self.referee_prototype_hsv is None:
            return False
        hsv = self._get_dominant_hsv(crop)
        if hsv is None:
            return False
        dist = float(np.linalg.norm(self.referee_prototype_hsv - hsv))
        return dist < threshold

    def update_track_history(self, tracks: List[TrackState]):
        """Records raw incoming tracking configurations prior to modification."""
        if not self.config.enabled:
            return
            
        for t in tracks:
            if t.role == "ball":
                continue
            tid = t.track_id
            if tid not in self.track_history:
                self.track_history[tid] = {
                    "raw_roles": [],
                    "raw_referee_confidences": [],
                    "locked_referee": False,
                    "final_role": t.role,
                    "lock_reason": "Active tracking"
                }
            history = self.track_history[tid]
            
            # Keep record of what the incoming track actually requested
            history["raw_roles"].append(t.role)
            if t.role == "referee":
                history["raw_referee_confidences"].append(t.confidence)

    def smooth_roles(self, tracks: List[TrackState], crops: Dict[int, np.ndarray] = None) -> List[TrackState]:
        """
        Performs multi-tiered strict verification to lock or reject referee designation.
        """
        if not self.config.enabled:
            return tracks
            
        # 1. Track state historical ingestion
        self.update_track_history(tracks)
        
        # 2. Apply logical filtering
        for t in tracks:
            if t.role == "ball":
                continue
            tid = t.track_id
            history = self.track_history.get(tid)
            if not history:
                continue
                
            # Rule A: Pre-locked? Skip computation.
            if history["locked_referee"]:
                self._override_to_referee(t)
                continue
                
            # Rule B: Compile strictly scoped aggregates
            roles = history["raw_roles"]
            ref_frames = roles.count("referee")
            total_frames = len(roles)
            
            ref_ratio = ref_frames / total_frames if total_frames > 0 else 0.0
            
            avg_ref_conf = 0.0
            if history["raw_referee_confidences"]:
                avg_ref_conf = float(np.mean(history["raw_referee_confidences"]))
            
            # Strict triple-check criteria logic
            met_frames = ref_frames >= self.config.referee_min_frames
            met_ratio = ref_ratio >= self.config.referee_min_ratio
            met_conf = avg_ref_conf >= self.config.referee_min_confidence
            
            is_ref = met_frames and met_ratio and met_conf
            lock_msg = ""
            if is_ref:
                lock_msg = (
                    f"Passed verification: {ref_frames} ref frames, "
                    f"ratio={ref_ratio:.2f}, avg_conf={avg_ref_conf:.2f}"
                )
            
            # Optional color overrides disabled by default
            if not is_ref and crops and tid in crops and self.config.use_color_referee_reid:
                # Safer Re-ID gate: Only fallback to color if track has basic referee signals
                has_signal = ref_frames >= 2 or (ref_ratio >= 0.15 and avg_ref_conf >= 0.45)
                if has_signal and self.check_color_reid(crops[tid]):
                    is_ref = True
                    lock_msg = "Locked via Color Prototype fallback"

            if is_ref:
                if self.config.lock_referee_role:
                    history["locked_referee"] = True
                    history["lock_reason"] = lock_msg
                self._override_to_referee(t)
                history["final_role"] = "referee"
            else:
                # Rejection logging updates
                if ref_frames > 0:
                    rejection_reasons = []
                    if not met_frames: rejection_reasons.append(f"frames ({ref_frames}<{self.config.referee_min_frames})")
                    if not met_ratio: rejection_reasons.append(f"ratio ({ref_ratio:.2f}<{self.config.referee_min_ratio})")
                    if not met_conf: rejection_reasons.append(f"conf ({avg_ref_conf:.2f}<{self.config.referee_min_confidence})")
                    history["lock_reason"] = "Rejected due to: " + ", ".join(rejection_reasons)
                history["final_role"] = t.role # Keeps player or gk designations
                
        return tracks

    def _override_to_referee(self, track: TrackState):
        track.role = "referee"
        track.class_name = "referee"
        track.team_id = None

    def generate_diagnostic_report(self) -> Dict[str, Any]:
        """
        Compiles complete debug state for all evaluated track IDs.
        """
        report = {}
        for tid, hist in self.track_history.items():
            roles = hist["raw_roles"]
            ref_count = roles.count("referee")
            total = len(roles)
            ratio = ref_count / total if total > 0 else 0.0
            
            ref_confs = hist["raw_referee_confidences"]
            avg_ref_conf = float(np.mean(ref_confs)) if ref_confs else 0.0
            
            # Build original count summary mapping role->count
            orig_counts = {}
            for r in set(roles):
                orig_counts[r] = roles.count(r)
                
            report[str(tid)] = {
                "total_frames": total,
                "original_roles_count": orig_counts,
                "smoothed_final_role": hist["final_role"],
                "referee_ratio": round(ratio, 3),
                "avg_referee_confidence": round(avg_ref_conf, 3),
                "is_locked_as_referee": hist["locked_referee"],
                "reason": hist["lock_reason"]
            }
        return report

    def print_summary_and_save(self, output_dir: str):
        """
        Prints descriptive diagnostic telemetry and dumps role_smoothing_report.json to the output dir.
        """
        if not self.config.enabled:
            return
            
        import os
        import json
        
        report = self.generate_diagnostic_report()
        
        # Save JSON
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "role_smoothing_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            
        # Collect aggregation metrics
        total_tracks = len(report)
        referee_tracks = [tid for tid, d in report.items() if d["is_locked_as_referee"]]
        total_ref_frames = sum(d["original_roles_count"].get("referee", 0) for d in report.values())
        
        print("\n========================================")
        print("   REFEREE ROLE SMOOTHING DIAGNOSTICS   ")
        print("========================================")
        print(f"Total Tracks Analyzed:     {total_tracks}")
        print(f"Locked Referee Tracks:     {len(referee_tracks)} tracks")
        print(f"Initial Referee Frames:    {total_ref_frames} frames")
        print(f"Locked Referee Track IDs:  {sorted([int(x) for x in referee_tracks])}")
        
        # Sort by ratio descending
        sorted_by_ratio = sorted(report.items(), key=lambda x: x[1]["referee_ratio"], reverse=True)
        
        print("\n--- Top 10 Tracks by Referee Ratio ---")
        for tid, d in sorted_by_ratio[:10]:
            lock_tag = "[LOCKED]" if d["is_locked_as_referee"] else "[REJECTED]"
            print(f"Track #{tid:2}: Ratio={d['referee_ratio']:.3f} | AvgConf={d['avg_referee_confidence']:.2f} | Frames={d['total_frames']:3} {lock_tag}")
            
        # Rejected tracks with some referee evidence
        rejected_ref_tracks = [
            (tid, d) for tid, d in sorted_by_ratio 
            if not d["is_locked_as_referee"] and d["original_roles_count"].get("referee", 0) > 0
        ]
        if rejected_ref_tracks:
            print("\n--- Tracks Rejected from Referee Role ---")
            for tid, d in rejected_ref_tracks[:10]:
                print(f"Track #{tid:2}: {d['reason']}")
                
        # 5. Safety Warning Threshold (max 4 referee IDs)
        if len(referee_tracks) > 4:
            print("\n" + "!" * 70)
            print(f"WARNING: More than {len(referee_tracks)} referee tracks detected. Please inspect track_contact_sheets.")
            print("!" * 70)
            
        print(f"\nSaved detailed diagnostics to: {report_path}\n")

