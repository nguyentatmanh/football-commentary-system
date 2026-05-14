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
    language: str = "vi"
    mode: str = "rule_based"
    min_gap_seconds: float = 4.0
    max_events_per_minute: int = 6
    min_priority: int = 5
    llm_fallback: bool = False

@dataclass
class TTSConfig:
    enabled: bool = True
    provider: str = "edge"
    voice: str = "vi-VN-NamMinhNeural"
    rate: str = "+10%"
    volume: str = "+0%"
    pitch: str = "+0Hz"

@dataclass
class AudioConfig:
    enabled: bool = True
    mix_with_original: bool = False
    commentary_volume_db: int = 0
    background_volume_db: int = -12

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
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    debug_tracks: bool = True
