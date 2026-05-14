import pytest
from football_ai.tracking.track_state import TrackState
from football_ai.tracking.tracker import Tracker
from football_ai.detection.detection_result import DetectionResult
from football_ai.config.schema import TrackingConfig

def test_track_state_serialization():
    """Verifies dict conversion loop for TrackState."""
    state = TrackState(
        frame_index=5,
        track_id=9,
        class_id=2,
        class_name="player",
        role="player",
        confidence=0.95,
        bbox_xyxy=[50.0, 60.0, 150.0, 260.0],
        team_id=None,
        pitch_xy=[10.5, 20.0]
    )

    serialized = state.to_dict()
    assert serialized["frame_index"] == 5
    assert serialized["track_id"] == 9
    assert serialized["pitch_xy"] == [10.5, 20.0]

    deserialized = TrackState.from_dict(serialized)
    assert deserialized.frame_index == 5
    assert deserialized.track_id == 9
    assert deserialized.bbox_xyxy == [50.0, 60.0, 150.0, 260.0]
    assert deserialized.pitch_xy == [10.5, 20.0]

def test_tracker_with_fake_detections():
    """Tests the high-level interface logic of our Tracker using mock objects."""
    config = TrackingConfig(min_consecutive_frames=1)
    tracker = Tracker(config)

    # Fake frame detections containing human players and a ball
    detections = [
        DetectionResult(frame_index=0, class_id=2, class_name="player", confidence=0.9, bbox_xyxy=[10,10,50,50], role="player"),
        DetectionResult(frame_index=0, class_id=0, class_name="ball", confidence=0.8, bbox_xyxy=[100,100,120,120], role="ball")
    ]

    tracks = tracker.update(detections, frame_index=0)
    
    # We expect the ball to definitely be present (since tracking is direct)
    # The humans might or might not be returned by ByteTrack on first frame depending on its configurations
    # Let's check if we parsed successfully.
    assert len(tracks) > 0
    roles = [t.role for t in tracks]
    assert "ball" in roles

    # Verify ball track id logic
    ball_track = next(t for t in tracks if t.role == "ball")
    assert ball_track.track_id == -1
    assert ball_track.bbox_xyxy == [100, 100, 120, 120]
