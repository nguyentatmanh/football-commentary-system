import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from football_ai.classification.team_classifier import TeamClassifier

def test_team_classifier_fit_predict_loop():
    """
    Tests the high-level TeamClassifier interface by mocking out 
    the underlying heavy SigLIP/Roboflow dependency.
    """
    with patch('football_ai.classification.team_classifier.RoboflowTeamClassifier') as MockRoboflow:
        # Configure the mock classifier
        mock_instance = MockRoboflow.return_value
        mock_instance.predict.return_value = np.array([0, 1])

        classifier = TeamClassifier()
        
        # Generate dummy image arrays
        fake_crops = [
            np.zeros((50, 50, 3), dtype=np.uint8),
            np.ones((50, 50, 3), dtype=np.uint8)
        ]
        
        # Should fail predict if not fitted
        with pytest.raises(ValueError):
            classifier.predict(fake_crops)
            
        # Fit
        classifier.fit(fake_crops)
        assert classifier._is_fitted is True
        
        # Predict batch
        results = classifier.predict(fake_crops)
        assert len(results) == 2
        assert results == [0, 1]
        
        # Predict one
        mock_instance.predict.return_value = np.array([1])
        res_one = classifier.predict_one(fake_crops[1])
        assert res_one == 1
