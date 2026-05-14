import pytest
import numpy as np
from football_ai.classification.role_smoother import RoleSmoother
from football_ai.config.schema import RoleSmoothingConfig
from football_ai.tracking.track_state import TrackState

@pytest.fixture
def strict_config():
    return RoleSmoothingConfig(
        enabled=True,
        referee_min_frames=8,
        referee_min_ratio=0.55,
        referee_min_confidence=0.55,
        lock_referee_role=True,
        use_color_referee_reid=False
    )

def _make_track(frame_idx: int, tid: int, role: str, conf: float, team_id=None) -> TrackState:
    return TrackState(
        frame_index=frame_idx,
        track_id=tid,
        class_id=2,
        class_name=role,
        role=role,
        confidence=conf,
        bbox_xyxy=[0,0,10,10],
        team_id=team_id
    )

def test_role_smoother_eighty_percent_valid(strict_config):
    """A track with 8/10 (80%) referee detections should become referee."""
    smoother = RoleSmoother(strict_config)
    tid = 100
    
    # 8 referee frames, 2 player frames
    for f in range(8):
        smoother.smooth_roles([_make_track(f, tid, "referee", 0.8)])
    for f in range(8, 10):
        smoother.smooth_roles([_make_track(f, tid, "player", 0.85)])
        
    # 8/10 = 80% > 55%
    # Met 8 frames constraint
    # Met confidence
    report = smoother.generate_diagnostic_report()
    assert report[str(tid)]["is_locked_as_referee"] is True
    assert report[str(tid)]["smoothed_final_role"] == "referee"

def test_role_smoother_insufficient_ratio(strict_config):
    """A track with 8 referee frames but 30 total frames (26% ratio) fails ratio threshold."""
    smoother = RoleSmoother(strict_config)
    tid = 200
    
    # 30 frames total: Interleave Player-Player-Referee so ratio sits around 33%
    # 8 referee frames scattered, 22 player frames scattered
    frames_sequence = ["player", "player", "referee"] * 8 + ["player"] * 6
    
    for idx, r in enumerate(frames_sequence):
        smoother.smooth_roles([_make_track(idx, tid, r, 0.85)])
        
    report = smoother.generate_diagnostic_report()
    # 8/30 = 0.266 < 0.55
    assert report[str(tid)]["is_locked_as_referee"] is False
    assert report[str(tid)]["smoothed_final_role"] == "player"
    assert "ratio" in report[str(tid)]["reason"]

def test_role_smoother_insufficient_frames(strict_config):
    """A track with 100% ratio but only 2 frames fails min_frames count."""
    smoother = RoleSmoother(strict_config)
    tid = 300
    
    # 2 referee frames only
    for f in range(2):
        smoother.smooth_roles([_make_track(f, tid, "referee", 0.95)])
        
    report = smoother.generate_diagnostic_report()
    assert report[str(tid)]["is_locked_as_referee"] is False
    assert report[str(tid)]["smoothed_final_role"] == "referee" # keeps raw for now but NOT locked!
    assert "frames" in report[str(tid)]["reason"]

def test_role_smoother_noisy_player_unaffected(strict_config):
    """A true player track with small, noisy referee glitches remains a player."""
    smoother = RoleSmoother(strict_config)
    tid = 400
    
    # 20 frames total: 2 ref noise glitches, 18 players
    for f in range(18):
        smoother.smooth_roles([_make_track(f, tid, "player", 0.85)])
    for f in range(18, 20):
        smoother.smooth_roles([_make_track(f, tid, "referee", 0.45)])
        
    report = smoother.generate_diagnostic_report()
    assert report[str(tid)]["is_locked_as_referee"] is False
    assert report[str(tid)]["smoothed_final_role"] == "referee" # the raw check, but let's test final overlay
    
    # Ensure when we run smooth_roles again on a player, it STAYS player!
    t_final = _make_track(20, tid, "player", 0.9)
    smoother.smooth_roles([t_final])
    assert t_final.role == "player"

def test_role_smoother_team_clearing(strict_config):
    """Locked referee tracks always have team_id = None."""
    smoother = RoleSmoother(strict_config)
    tid = 500
    
    # 8 strong referee frames -> locks
    for f in range(8):
        smoother.smooth_roles([_make_track(f, tid, "referee", 0.8)])
        
    # Simulate incoming frame where YOLO predicts "player" with an accidental team assignment
    t_bad = _make_track(8, tid, "player", 0.9, team_id=0)
    
    smoother.smooth_roles([t_bad])
    assert t_bad.role == "referee"
    assert t_bad.team_id is None
