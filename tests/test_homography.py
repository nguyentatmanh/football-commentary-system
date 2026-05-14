import pytest
import numpy as np
from football_ai.field_mapping.homography import HomographyEstimator

def test_homography_successful_transform():
    """Tests ideal homography mapping of a standard square unit to pixel space."""
    estimator = HomographyEstimator()
    
    # 1. Setup points for a unit square to a scaled translated rectangle
    # Source: (0,0), (1,0), (1,1), (0,1)
    src = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float32)
    
    # Target: scaled by 100, shifted by (50, 50)
    dst = np.array([
        [50.0, 50.0],
        [150.0, 50.0],
        [150.0, 150.0],
        [50.0, 150.0]
    ], dtype=np.float32)
    
    success = estimator.fit(src, dst)
    assert success is True
    assert estimator.is_ready() is True
    
    # 2. Project the center point (0.5, 0.5) -> should map to (100.0, 100.0)
    center = np.array([[0.5, 0.5]], dtype=np.float32)
    res = estimator.transform_points(center)
    
    assert res is not None
    assert res.shape == (1, 2)
    np.testing.assert_allclose(res[0], [100.0, 100.0], atol=1e-5)

def test_homography_failed_cases():
    """Checks error checking invariants in homography calculator."""
    estimator = HomographyEstimator()
    
    # Case A: Not enough points (<4)
    src = np.array([[0,0],[1,1],[2,2]], dtype=np.float32)
    dst = np.array([[0,0],[1,1],[2,2]], dtype=np.float32)
    
    assert estimator.fit(src, dst) is False
    assert estimator.is_ready() is False
    
    # Case B: Size mismatches
    src = np.array([[0,0],[1,1],[2,2],[3,3]], dtype=np.float32)
    dst = np.array([[0,0],[1,1],[2,2]], dtype=np.float32)
    assert estimator.fit(src, dst) is False
    
    # Case C: Unready transform
    assert estimator.transform_points(np.array([[1,1]])) is None
