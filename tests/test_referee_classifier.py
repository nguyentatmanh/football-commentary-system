import pytest
from football_ai.classification.referee_classifier import RefereeClassifier
from football_ai.tracking.track_state import TrackState

def test_referee_normalization():
    """Verifies that referee classifier strips out team_id and forces role label."""
    classifier = RefereeClassifier()
    
    ref = TrackState(
        frame_index=0, track_id=1, class_id=3, class_name="referee",
        role="some_other_role", confidence=0.9, bbox_xyxy=[0,0,10,10], team_id=0 # Accidentally given Team 0
    )
    
    processed = classifier.process_state(ref)
    assert processed.role == "referee"
    assert processed.team_id is None
