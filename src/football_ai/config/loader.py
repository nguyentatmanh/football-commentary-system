import os
import yaml
from typing import Any, Dict
from football_ai.config.schema import (
    SystemConfig,
    VideoConfig,
    DetectionConfig,
    TrackingConfig,
    ClassificationConfig,
    RoleSmoothingConfig,
    RoleOverridesConfig,
    AnalyticsConfig,
    CommentaryConfig
)

def load_config(config_path: str) -> SystemConfig:
    """Load yaml config file and merge with dataclass structure."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f) or {}
    
    def get_nested(d: Dict, key: str, cls: Any) -> Any:
        sub = d.get(key, {})
        if not isinstance(sub, dict):
            return cls()
        return cls(**{k: v for k, v in sub.items() if k in cls.__dataclass_fields__})

    return SystemConfig(
        device=data.get("device", "cpu"),
        video=get_nested(data, "video", VideoConfig),
        detection=get_nested(data, "detection", DetectionConfig),
        tracking=get_nested(data, "tracking", TrackingConfig),
        classification=get_nested(data, "classification", ClassificationConfig),
        role_smoothing=get_nested(data, "role_smoothing", RoleSmoothingConfig),
        role_overrides=get_nested(data, "role_overrides", RoleOverridesConfig),
        analytics=get_nested(data, "analytics", AnalyticsConfig),
        commentary=get_nested(data, "commentary", CommentaryConfig),
        debug_tracks=data.get("debug_tracks", True)
    )

def merge_config_overrides(config: SystemConfig, overrides: Dict[str, Any]) -> SystemConfig:
    """Merge runtime CLI args override parameters."""
    if overrides.get("device"):
        config.device = overrides["device"]
    if overrides.get("input_path"):
        config.video.input_path = overrides["input_path"]
    if overrides.get("output_dir"):
        config.video.output_dir = overrides["output_dir"]
    return config
