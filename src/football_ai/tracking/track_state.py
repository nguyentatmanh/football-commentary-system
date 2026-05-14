from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

@dataclass
class TrackState:
    frame_index: int
    track_id: int
    class_id: Optional[int]
    class_name: str
    role: str
    confidence: float
    bbox_xyxy: List[float]
    team_id: Optional[int] = None
    pitch_xy: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize data class to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrackState':
        """Deserialize dictionary to data class."""
        return cls(
            frame_index=data["frame_index"],
            track_id=data["track_id"],
            class_id=data.get("class_id"),
            class_name=data["class_name"],
            role=data["role"],
            confidence=data["confidence"],
            bbox_xyxy=data["bbox_xyxy"],
            team_id=data.get("team_id"),
            pitch_xy=data.get("pitch_xy")
        )
