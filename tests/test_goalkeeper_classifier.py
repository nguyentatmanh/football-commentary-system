import pytest
from football_ai.classification.goalkeeper_classifier import GoalkeeperClassifier
from football_ai.tracking.track_state import TrackState

def test_goalkeeper_proximity_assignment():
    """
    Creates geometric centroids for Team 0 (left side) and Team 1 (right side), 
    and verifies goalkeeper moves towards nearest bucket.
    """
    classifier = GoalkeeperClassifier()

    # 1. Setup players for Team 0 around x=100, y=100
    team_0_player = TrackState(
        frame_index=0, track_id=1, class_id=2, class_name="player",
        role="player", confidence=0.9, bbox_xyxy=[90.0, 90.0, 110.0, 110.0], team_id=0
    )
    
    # 2. Setup players for Team 1 around x=900, y=900
    team_1_player = TrackState(
        frame_index=0, track_id=2, class_id=2, class_name="player",
        role="player", confidence=0.9, bbox_xyxy=[890.0, 890.0, 910.0, 910.0], team_id=1
    )
    
    # 3. Setup goalkeeper near Team 0 (at x=150, y=150)
    gk_near_0 = TrackState(
        frame_index=0, track_id=3, class_id=1, class_name="goalkeeper",
        role="goalkeeper", confidence=0.9, bbox_xyxy=[140.0, 140.0, 160.0, 160.0], team_id=None
    )
    
    # 4. Setup goalkeeper near Team 1 (at x=850, y=850)
    gk_near_1 = TrackState(
        frame_index=0, track_id=4, class_id=1, class_name="goalkeeper",
        role="goalkeeper", confidence=0.9, bbox_xyxy=[840.0, 840.0, 860.0, 860.0], team_id=None
    )

    tracks = [team_0_player, team_1_player, gk_near_0, gk_near_1]
    resolved = classifier.resolve_team_ids(tracks)

    # GK near team 0 (x=150) should map to Team 0 (centroid x=100)
    assert resolved[2].track_id == 3
    assert resolved[2].team_id == 0
    
    # GK near team 1 (x=850) should map to Team 1 (centroid x=900)
    assert resolved[3].track_id == 4
    assert resolved[3].team_id == 1

def test_goalkeeper_empty_cases():
    """Ensures classifier doesn't crash and outputs None if missing evidence."""
    classifier = GoalkeeperClassifier()
    
    # Setup goalkeeper with no players around
    gk = TrackState(
        frame_index=0, track_id=3, class_id=1, class_name="goalkeeper",
        role="goalkeeper", confidence=0.9, bbox_xyxy=[140.0, 140.0, 160.0, 160.0], team_id=None
    )
    
    resolved = classifier.resolve_team_ids([gk])
    assert resolved[0].team_id is None
