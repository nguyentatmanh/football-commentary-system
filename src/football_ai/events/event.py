from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict

@dataclass
class FootballEvent:
    event_id: str
    type: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    team_id: Optional[int]
    player_id: Optional[int]
    confidence: float
    priority: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the event to a dictionary."""
        return asdict(self)
