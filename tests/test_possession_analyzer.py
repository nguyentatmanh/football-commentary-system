import pytest
from football_ai.analytics.possession_analyzer import PossessionAnalyzer
from football_ai.tracking.track_state import TrackState

def test_possession_radius_proximity():
    """Validates active possession assignment based on absolute proximity constraints."""
    # Radius limit = 2.0 meters
    analyzer = PossessionAnalyzer(grab_radius_m=2.0)
    
    # 1. Place Ball at center [50, 30]
    ball = TrackState(frame_index=0, track_id=-1, class_id=0, role="ball", class_name="ball", confidence=0.9, bbox_xyxy=[0,0,0,0], pitch_xy=[50.0, 30.0])
    
    # 2. Place Team A Player at [51.5, 30] (distance = 1.5 meters -> inside radius)
    player_a = TrackState(frame_index=0, track_id=10, class_id=2, role="player", class_name="player", team_id=0, confidence=0.9, bbox_xyxy=[0,0,0,0], pitch_xy=[51.5, 30.0])
    
    # 3. Place Team B Player at [60, 30] (distance = 10.0 meters -> outside)
    player_b = TrackState(frame_index=0, track_id=20, class_id=2, role="player", class_name="player", team_id=1, confidence=0.9, bbox_xyxy=[0,0,0,0], pitch_xy=[60.0, 30.0])
    
    analyzer.process_frame([ball, player_a, player_b], frame_index=0)
    
    # Should successfully assign to Team 0
    assert analyzer.timeline[0] == 0

def test_possession_out_of_bounds():
    """Guarantees possession results in None if players are too far from ball."""
    analyzer = PossessionAnalyzer(grab_radius_m=2.0)
    
    ball = TrackState(frame_index=0, track_id=-1, class_id=0, role="ball", class_name="ball", confidence=0.9, bbox_xyxy=[0,0,0,0], pitch_xy=[50.0, 30.0])
    # Distance = 2.5 meters (exceeds 2.0 limit)
    player_a = TrackState(frame_index=0, track_id=10, class_id=2, role="player", class_name="player", team_id=0, confidence=0.9, bbox_xyxy=[0,0,0,0], pitch_xy=[52.5, 30.0])
    
    analyzer.process_frame([ball, player_a], frame_index=0)
    assert analyzer.timeline[0] is None
    
    # Test computation stats
    stats = analyzer.compute_stats()
    assert stats["possession_distribution"]["none_percent"] == 100.0
