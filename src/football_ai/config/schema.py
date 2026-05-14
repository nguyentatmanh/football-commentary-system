from dataclasses import dataclass, field
from typing import Optional

@dataclass
class VideoConfig:
    input_path: str = "data/raw/sample.mp4"
    output_dir: str = "data/outputs/demo"

@dataclass
class DetectionConfig:
    player_model_path: str = "data/models/football-player-detection.pt"
    pitch_model_path: str = "data/models/football-pitch-detection.pt"
    ball_model_path: str = "data/models/football-ball-detection.pt"
    confidence: float = 0.3
    iou: float = 0.5
    imgsz: int = 1280

@dataclass
class TrackingConfig:
    min_consecutive_frames: int = 3

@dataclass
class ClassificationConfig:
    stride: int = 60

@dataclass
class RoleSmoothingConfig:
    enabled: bool = True
    referee_min_frames: int = 8
    referee_min_ratio: float = 0.55
    referee_min_confidence: float = 0.55
    lock_referee_role: bool = True
    use_color_referee_reid: bool = False
@dataclass
class RoleOverridesConfig:
    enabled: bool = True
    referee_track_ids: list[int] = field(default_factory=list)
    player_track_ids: list[int] = field(default_factory=list)
    goalkeeper_track_ids: list[int] = field(default_factory=list)
    ball_track_ids: list[int] = field(default_factory=list)
@dataclass
class AnalyticsConfig:
    radar_opacity: float = 0.5
    heatmap_opacity: float = 0.6

@dataclass
class CommentaryConfig:
    enabled: bool = True
    llm_fallback: bool = False

@dataclass
class SystemConfig:
    device: str = "cpu"
    video: VideoConfig = field(default_factory=VideoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    role_smoothing: RoleSmoothingConfig = field(default_factory=RoleSmoothingConfig)
    role_overrides: RoleOverridesConfig = field(default_factory=RoleOverridesConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    commentary: CommentaryConfig = field(default_factory=CommentaryConfig)
    debug_tracks: bool = True
