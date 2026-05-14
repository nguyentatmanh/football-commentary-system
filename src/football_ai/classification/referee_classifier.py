from typing import List
from football_ai.tracking.track_state import TrackState

class RefereeClassifier:
    """
    Handles normalizing referee identifiers, ensuring they are not
    misclassified into Team A or Team B.
    """
    def process_state(self, state: TrackState) -> TrackState:
        """Normalizes track state to ensure referee invariants hold."""
        if state.role == "referee" or state.class_name == "referee":
            state.role = "referee"
            state.team_id = None
        return state

    def normalize_referees(self, tracks: List[TrackState]) -> List[TrackState]:
        """Apply referee normalization constraints across a batch of tracks."""
        for track in tracks:
            self.process_state(track)
        return tracks
