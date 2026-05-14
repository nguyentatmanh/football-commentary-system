import pytest
from football_ai.classification.role_override import RoleOverrideApplier
from football_ai.config.schema import RoleOverridesConfig
from football_ai.tracking.track_state import TrackState

@pytest.fixture
def mock_override_config():
    return RoleOverridesConfig(
        enabled=True,
        referee_track_ids=[5],
        player_track_ids=[10],
        goalkeeper_track_ids=[15],
        ball_track_ids=[20]
    )

def _make_track(tid: int, role: str, team_id=None) -> TrackState:
    return TrackState(
        frame_index=0,
        track_id=tid,
        class_id=1,
        class_name=role,
        role=role,
        confidence=0.9,
        bbox_xyxy=[0, 0, 100, 100],
        team_id=team_id
    )

def test_role_override_applies_forced_roles(mock_override_config):
    applier = RoleOverrideApplier(mock_override_config)
    
    t5 = _make_track(5, "player", team_id=0)
    t10 = _make_track(10, "referee", team_id=None)
    t15 = _make_track(15, "player", team_id=1)
    t20 = _make_track(20, "player", team_id=None)
    
    tracks = [t5, t10, t15, t20]
    applier.apply_overrides(tracks)
    
    # Assertions
    assert t5.role == "referee"
    assert t5.team_id is None
    
    assert t10.role == "player"
    
    assert t15.role == "goalkeeper"
    
    assert t20.role == "ball"
    assert t20.team_id is None

def test_disabled_override(mock_override_config):
    mock_override_config.enabled = False
    applier = RoleOverrideApplier(mock_override_config)
    
    t5 = _make_track(5, "player", team_id=0)
    applier.apply_overrides([t5])
    
    # Should NOT change since it's disabled
    assert t5.role == "player"
    assert t5.team_id == 0
