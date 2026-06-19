import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from football_ai.tracking.track_state import TrackState

@dataclass
class MovementStats:
    track_id: int
    role: str
    team_id: int | None
    total_distance_m: float = 0.0
    average_speed_mps: float = 0.0
    max_speed_mps: float = 0.0
    trajectory: List[Tuple[int, List[float]]] = field(default_factory=list) # List of (frame_idx, [x, y])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "role": self.role,
            "team_id": self.team_id,
            "total_distance_m": round(self.total_distance_m, 2),
            "average_speed_mps": round(self.average_speed_mps, 2),
            "max_speed_mps": round(self.max_speed_mps, 2),
            # Trajectory excluded from summary json to keep output small but accessible in memory
        }

class MovementAnalyzer:
    """
    Analyzes spatial trajectories of players and the ball across time.
    Estimates distances, filters physics-defying speeds, and tracks maximums.
    """
    def __init__(self, fps: float = 30.0, max_player_speed: float = 12.0, max_ball_speed: float = 40.0):
        self.fps = fps
        self.max_player_speed = max_player_speed
        self.max_ball_speed = max_ball_speed
        
        # Storage index by track_id
        self.stats: Dict[int, MovementStats] = {}
        
        # Temporary memory tracking speed deltas for final average calculations
        self._speed_samples: Dict[int, List[float]] = {}

    def process_frame_tracks(self, tracks: List[TrackState]):
        """
        Appends incoming track positions and computes delta metrics on the fly.
        """
        for t in tracks:
            if t.pitch_xy is None:
                continue
                
            tid = t.track_id
            
            # Initialize state if track is brand new
            if tid not in self.stats:
                self.stats[tid] = MovementStats(
                    track_id=tid,
                    role=t.role,
                    team_id=t.team_id
                )
                self._speed_samples[tid] = []

            # Áp dụng bộ lọc EMA (Exponential Moving Average) để khử nhiễu vi dao động của bounding box
            mstats = self.stats[tid]
            alpha = 0.15  # Hệ số làm mượt (càng nhỏ càng mượt, giảm nhiễu)
            
            if not mstats.trajectory:
                smoothed_xy = list(t.pitch_xy)
            else:
                _, prev_smoothed_xy = mstats.trajectory[-1]
                smoothed_xy = [
                    alpha * t.pitch_xy[0] + (1 - alpha) * prev_smoothed_xy[0],
                    alpha * t.pitch_xy[1] + (1 - alpha) * prev_smoothed_xy[1]
                ]

            # Lưu tọa độ đã được làm mượt vào quỹ đạo
            mstats.trajectory.append((t.frame_index, smoothed_xy))

            # Nếu có ít nhất 2 điểm, tính toán độ dịch chuyển và vận tốc
            if len(mstats.trajectory) >= 2:
                prev_frame, prev_xy = mstats.trajectory[-2]
                curr_frame, curr_xy = mstats.trajectory[-1]
                
                # 1. Tính khoảng thời gian giữa 2 khung hình
                frame_diff = curr_frame - prev_frame
                if frame_diff <= 0:
                    continue
                time_delta = frame_diff / self.fps
                
                # 2. Tính độ dịch chuyển (m) dựa trên tọa độ đã làm mượt
                dist = float(np.linalg.norm(np.array(curr_xy) - np.array(prev_xy)))
                speed = dist / time_delta
                
                # 3. Loại bỏ các bước nhảy phi vật lý (nhiễu nhảy ID hoặc lỗi tracking)
                threshold = self.max_ball_speed if t.role == "ball" else self.max_player_speed
                if speed > threshold:
                    mstats.trajectory.pop()
                    continue
                    
                # 4. Triệt tiêu nhiễu dao động nhỏ khi đứng yên (dead-zone filter)
                # Nếu dịch chuyển nhỏ hơn 2cm (0.02m) trong 1 frame, coi như đứng yên
                if dist < 0.02:
                    dist = 0.0
                    speed = 0.0
                    
                # 5. Cộng dồn quãng đường và cập nhật tốc độ cực đại
                mstats.total_distance_m += dist
                mstats.max_speed_mps = max(mstats.max_speed_mps, speed)
                
                self._speed_samples[tid].append(speed)

    def finalize_stats(self) -> Dict[int, MovementStats]:
        """
        Computes aggregate statistics (average speeds) and returns final results.
        """
        for tid, speed_list in self._speed_samples.items():
            if speed_list:
                self.stats[tid].average_speed_mps = float(np.mean(speed_list))
            else:
                self.stats[tid].average_speed_mps = 0.0
                
        return self.stats

    def get_summary_dict(self) -> List[Dict[str, Any]]:
        """Yields sorted serializable list of analysis objects."""
        finalized = self.finalize_stats()
        # Sort by role and distance
        return [s.to_dict() for s in sorted(finalized.values(), key=lambda x: x.total_distance_m, reverse=True)]
