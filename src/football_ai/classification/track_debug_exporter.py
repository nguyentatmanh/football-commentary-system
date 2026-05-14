import os
import cv2
import json
import numpy as np
from typing import List, Dict, Any
from football_ai.tracking.track_state import TrackState

class TrackDebugExporter:
    """
    Harvests sample visual crops and compiles telemetry datasets and grid
    contact sheets for human audit and confirmation of track roles.
    """
    def __init__(self, enabled: bool = True, max_crops: int = 16):
        self.enabled = enabled
        self.max_crops = max_crops
        # Memory store: {track_id: {"crops": [], "roles": [], "confs": [], "team_ids": []}}
        self.data: Dict[int, Dict[str, Any]] = {}

    def ingest_frame_crops(self, tracks: List[TrackState], crops: Dict[int, np.ndarray]):
        """Collects state indicators and visual crops for dynamic active tracks."""
        if not self.enabled:
            return
            
        for t in tracks:
            tid = t.track_id
            if tid not in self.data:
                self.data[tid] = {
                    "crops": [],
                    "original_roles": [],
                    "final_roles": [],
                    "team_ids": [],
                    "confidences": []
                }
            
            rec = self.data[tid]
            rec["final_roles"].append(t.role)
            rec["team_ids"].append(t.team_id)
            rec["confidences"].append(t.confidence)
            
            # Keep track of original_roles via class_name (as dynamic tracker updates role)
            rec["original_roles"].append(t.class_name)
            
            # Buffer crop with reservoir/stride logic so we don't overflow memory
            if crops and tid in crops:
                crop = crops[tid]
                if crop is not None and crop.size > 0:
                    # Append but keep at most 50 crops stored, we subsample to 16 at saving stage
                    if len(rec["crops"]) < 50:
                        # Copy crop so it isn't dependent on large parent memory buffers
                        rec["crops"].append(crop.copy())

    def _create_grid(self, crops: List[np.ndarray], cols: int = 4, thumb_size: tuple = (96, 192)) -> np.ndarray:
        """Stitches images together in a regular spacing grid."""
        if not crops:
            # Draw simple gray rectangle if empty
            return np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8) + 127
            
        # Sample up to target size evenly
        if len(crops) > self.max_crops:
            indices = np.linspace(0, len(crops) - 1, self.max_crops, dtype=int)
            samples = [crops[i] for i in indices]
        else:
            samples = crops
            
        # Pad to fill last row completely
        rows = int(np.ceil(len(samples) / cols))
        total_slots = rows * cols
        
        # Resize elements to uniform sizes
        resized = [cv2.resize(c, thumb_size) for c in samples]
        
        # Fill remainder slots with pure black
        while len(resized) < total_slots:
            resized.append(np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8))
            
        grid_rows = []
        for r in range(rows):
            row_imgs = resized[r * cols : (r + 1) * cols]
            grid_rows.append(np.hstack(row_imgs))
            
        return np.vstack(grid_rows)

    def export(self, output_dir: str):
        """Generates JSON stats report and visual jpg sheets under directory."""
        if not self.enabled or not self.data:
            return
            
        sheets_dir = os.path.join(output_dir, "track_contact_sheets")
        os.makedirs(sheets_dir, exist_ok=True)
        
        summary_data = {}
        
        for tid, rec in self.data.items():
            total = len(rec["final_roles"])
            if total == 0:
                continue
                
            avg_conf = float(np.mean(rec["confidences"]))
            
            # Count distributions helper
            def count_dict(items):
                d = {}
                for i in items:
                    k = str(i)
                    d[k] = d.get(k, 0) + 1
                return d
                
            orig_counts = count_dict(rec["original_roles"])
            final_counts = count_dict(rec["final_roles"])
            team_counts = count_dict(rec["team_ids"])
            
            summary_data[str(tid)] = {
                "total_frames": total,
                "avg_confidence": round(avg_conf, 3),
                "original_role_distribution": orig_counts,
                "final_role_distribution": final_counts,
                "team_id_distribution": team_counts,
            }
            
            # 2. Visual Contact Sheet Rendering
            # Combine and draw metadata text bar above
            grid_img = self._create_grid(rec["crops"], cols=4)
            gh, gw, _ = grid_img.shape
            
            header = np.zeros((60, gw, 3), dtype=np.uint8) + 40  # Dark header
            
            # Text metrics strings
            final_role_str = ", ".join([f"{k}:{v}" for k, v in final_counts.items()])
            header_text = f"Track #{tid} | Frames: {total} | Roles: {final_role_str}"
            team_str = f"Teams: " + ", ".join([f"{k}:{v}" for k, v in team_counts.items()])
            
            cv2.putText(header, header_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(header, team_str, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            
            full_sheet = np.vstack([header, grid_img])
            
            out_sheet_path = os.path.join(sheets_dir, f"track_{tid}.jpg")
            cv2.imwrite(out_sheet_path, full_sheet)
            
        # Export summary metrics JSON
        summary_path = os.path.join(output_dir, "track_role_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
            
        print(f"\n[Debug Exporter] Created {len(summary_data)} visual contact sheets.")
        print(f"[Debug Exporter] Saved summary to: {summary_path}")
        print(f"[Debug Exporter] Saved sheets to: {sheets_dir}\n")
