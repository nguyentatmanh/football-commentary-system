import pytest
from football_ai.detection.detection_result import DetectionResult

def test_detection_result_serialization():
    """Test that DetectionResult transforms to and from dictionary correctly."""
    det = DetectionResult(
        frame_index=42,
        class_id=2,
        class_name="player",
        confidence=0.85,
        bbox_xyxy=[10.0, 20.0, 100.0, 200.0],
        track_id=5,
        team_id=0,
        role="player"
    )

    # To Dict
    serialized = det.to_dict()
    assert serialized["frame_index"] == 42
    assert serialized["class_id"] == 2
    assert serialized["class_name"] == "player"
    assert serialized["confidence"] == 0.85
    assert serialized["bbox_xyxy"] == [10.0, 20.0, 100.0, 200.0]
    assert serialized["track_id"] == 5
    assert serialized["team_id"] == 0
    assert serialized["role"] == "player"

    # From Dict
    deserialized = DetectionResult.from_dict(serialized)
    assert deserialized.frame_index == 42
    assert deserialized.class_id == 2
    assert deserialized.confidence == 0.85
    assert deserialized.bbox_xyxy == [10.0, 20.0, 100.0, 200.0]
    assert deserialized.track_id == 5
    assert deserialized.team_id == 0
    assert deserialized.role == "player"

def test_detection_result_defaults():
    """Test defaults for optional params."""
    det = DetectionResult(
        frame_index=0,
        class_id=None,
        class_name="unknown",
        confidence=0.0,
        bbox_xyxy=[0,0,0,0]
    )
    assert det.track_id is None
    assert det.team_id is None
    assert det.role == "unknown"
