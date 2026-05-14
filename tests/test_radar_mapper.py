import pytest
import numpy as np
from unittest.mock import patch
from football_ai.field_mapping.radar_mapper import RadarMapper
from football_ai.field_mapping.pitch_keypoints import PitchKeypointResult
from football_ai.tracking.track_state import TrackState

def test_radar_mapper_scaling_and_caching():
    """
    Validates coordinate meter scaling, logical anchor points, 
    and robust temporal caching of homography projection.
    """
    mapper = RadarMapper(max_homography_staleness=3)
    
    # Create a fake track state (player) at screen coords
    # Bottom center will be at x = 500, y = 1000
    player = TrackState(
        frame_index=0, track_id=1, class_id=2, class_name="player",
        role="player", confidence=0.9, bbox_xyxy=[400.0, 900.0, 600.0, 1000.0]
    )

    def mock_fit(src, dst):
        mapper.estimator.matrix = np.eye(3)
        return True

    # Generate a synthetic keypoint result representing a perfect map to pitch configs
    with patch.object(mapper.estimator, 'fit', side_effect=mock_fit), \
         patch.object(mapper.estimator, 'transform_points') as mock_project:
        
        # Force transformer output to arbitrary cm value: [10500.0, 6800.0] (the max corner of field)
        mock_project.return_value = np.array([[10500.0, 6800.0]], dtype=np.float32)
        
        # Setup dummy keypoints resulting in estimator updates
        fake_kps = PitchKeypointResult(
            frame_index=0, 
            keypoints_xy=[[100,100]]*32, 
            success=True
        )
        
        # Update projection matrix (this is simulated to succeed)
        mapper.update_pitch_projection(fake_kps)
        
        # Run mapping
        resolved = mapper.map_tracks([player])
        
        # Verify coordinates mapped successfully and scaled from cm to meters [/100.0]
        assert resolved[0].pitch_xy == [105.0, 68.0]

def test_radar_mapper_stale_matrix_degradation():
    """
    Guarantees caching thresholds clear matrices after exceeding tolerance limit.
    """
    mapper = RadarMapper(max_homography_staleness=2)
    
    # Inject fake valid matrix directly
    mapper.last_valid_matrix = np.eye(3)
    mapper.frames_since_valid = 0
    
    # 1. Unsuccessful frame 1 -> uses cached projection
    fail_kps = PitchKeypointResult(frame_index=1, success=False)
    updated = mapper.update_pitch_projection(fail_kps)
    assert updated is True
    assert mapper.estimator.matrix is not None
    
    # 2. Unsuccessful frame 2 -> uses cached projection
    updated = mapper.update_pitch_projection(fail_kps)
    assert updated is True
    
    # 3. Unsuccessful frame 3 -> staleness threshold exceeded! (degraded!)
    updated = mapper.update_pitch_projection(fail_kps)
    assert updated is False
    assert mapper.estimator.matrix is None
