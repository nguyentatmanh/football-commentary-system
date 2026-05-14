from typing import List
from football_ai.tracking.track_state import TrackState
from football_ai.config.schema import RoleOverridesConfig

class RoleOverrideApplier:
    """
    Enforces hardcoded role assignments for specific track IDs to allow
    deterministic human override in practical test cases.
    """
    def __init__(self, config: RoleOverridesConfig):
        self.config = config
        # Cache sets for faster lookups
        self.referee_ids = set(config.referee_track_ids) if config.referee_track_ids else set()
        self.player_ids = set(config.player_track_ids) if config.player_track_ids else set()
        self.gk_ids = set(config.goalkeeper_track_ids) if config.goalkeeper_track_ids else set()
        self.ball_ids = set(config.ball_track_ids) if config.ball_track_ids else set()

    def apply_overrides(self, tracks: List[TrackState]) -> List[TrackState]:
        """Apply hard overrides if active."""
        if not self.config.enabled:
            return tracks
            
        for t in tracks:
            tid = t.track_id
            
            if tid in self.referee_ids:
                t.role = "referee"
                t.class_name = "referee"
                t.team_id = None
            elif tid in self.player_ids:
                t.role = "player"
                t.class_name = "player"
                # Keeps team_id intact or lets classifier compute it
            elif tid in self.gk_ids:
                t.role = "goalkeeper"
                t.class_name = "goalkeeper"
            elif tid in self.ball_ids:
                t.role = "ball"
                t.class_name = "ball"
                t.team_id = None
                
        return tracks
