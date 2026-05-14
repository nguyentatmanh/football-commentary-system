from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

@dataclass
class DetectionResult:
    frame_index: int
    class_id: Optional[int]
    class_name: str
    confidence: float
    bbox_xyxy: List[float] # [x1, y1, x2, y2]
    track_id: Optional[int] = None
    team_id: Optional[int] = None
    role: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize data class to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DetectionResult':
        """Deserialize dictionary to data class."""
        return cls(
            frame_index=data["frame_index"],
            class_id=data.get("class_id"),
            class_name=data["class_name"],
            confidence=data["confidence"],
            bbox_xyxy=data["bbox_xyxy"],
            track_id=data.get("track_id"),
            team_id=data.get("team_id"),
            role=data.get("role", "unknown")
        )
