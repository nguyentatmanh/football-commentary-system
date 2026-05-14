import pytest
from football_ai.analytics.movement_analyzer import MovementAnalyzer
from football_ai.tracking.track_state import TrackState

def test_movement_analyzer_basic_metrics():
    """Validates linear distance accumulation and velocity averages."""
    # FPS=10 for clean math. Max player speed = 15.0 m/s
    analyzer = MovementAnalyzer(fps=10.0, max_player_speed=15.0)
    
    # Create sequence of linear movements for 1 player
    # Frame 0: x=0, y=0
    # Frame 10: x=10, y=0 (Time = 1s, Distance = 10m, Speed = 10m/s)
    # Frame 20: x=10, y=10 (Time = 1s, Distance = 10m, Speed = 10m/s)
    t0 = TrackState(frame_index=0, track_id=1, class_id=2, role="player", class_name="player", confidence=0.9, bbox_xyxy=[0,0,10,10], pitch_xy=[0.0, 0.0])
    t1 = TrackState(frame_index=10, track_id=1, class_id=2, role="player", class_name="player", confidence=0.9, bbox_xyxy=[0,0,10,10], pitch_xy=[10.0, 0.0])
    t2 = TrackState(frame_index=20, track_id=1, class_id=2, role="player", class_name="player", confidence=0.9, bbox_xyxy=[0,0,10,10], pitch_xy=[10.0, 10.0])
    
    analyzer.process_frame_tracks([t0])
    analyzer.process_frame_tracks([t1])
    analyzer.process_frame_tracks([t2])
    
    res = analyzer.finalize_stats()
    p1 = res[1]
    
    assert pytest.approx(p1.total_distance_m, 1e-2) == 20.0
    assert pytest.approx(p1.average_speed_mps, 1e-2) == 10.0
    assert pytest.approx(p1.max_speed_mps, 1e-2) == 10.0

def test_movement_analyzer_speed_clamping():
    """Ensures physics-defying coordinate jumps are filtered."""
    analyzer = MovementAnalyzer(fps=10.0, max_player_speed=12.0)
    
    # Frame 0 -> Frame 10 (1 second delta)
    t0 = TrackState(frame_index=0, track_id=1, class_id=2, role="player", class_name="player", confidence=0.9, bbox_xyxy=[0,0,10,10], pitch_xy=[0.0, 0.0])
    # Frame 10 -> x=100m (Speed = 100m/s! WAY higher than 12.0)
    t1 = TrackState(frame_index=10, track_id=1, class_id=2, role="player", class_name="player", confidence=0.9, bbox_xyxy=[0,0,10,10], pitch_xy=[100.0, 0.0])
    
    analyzer.process_frame_tracks([t0])
    analyzer.process_frame_tracks([t1])
    
    res = analyzer.finalize_stats()
    p1 = res[1]
    
    # The illegal jump should have been popped! Total distance remains 0.
    assert p1.total_distance_m == 0.0
    assert len(p1.trajectory) == 1 # Only first anchor retained
