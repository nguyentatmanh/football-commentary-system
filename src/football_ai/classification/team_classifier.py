import numpy as np
from typing import List, Optional
import football_ai.utils.paths # Automatically sets up pathing
# pyrefly: ignore [missing-import]
from sports.common.team import TeamClassifier as RoboflowTeamClassifier

class TeamClassifier:
    """
    Classifies tracked players into Team 0 / Team 1 based on Roboflow's
    SigLIP vision feature extraction and UMAP/KMeans clustering.
    """
    def __init__(self, device: str = "cpu", batch_size: int = 32):
        self.device = device
        # Instantiate lazy/eager depending on device usage
        self._classifier = RoboflowTeamClassifier(device=device, batch_size=batch_size)
        self._is_fitted = False

    def fit(self, player_crops: List[np.ndarray]) -> None:
        """
        Train the clustering classifier model using a collection of representative player crops.
        """
        if not player_crops:
            print("[!] Warning: fit() called with empty player crops list.")
            return
            
        print(f"Fitting team classifier with {len(player_crops)} player image crops...")
        self._classifier.fit(player_crops)
        self._is_fitted = True

    def predict(self, player_crops: List[np.ndarray]) -> List[int]:
        """
        Predict numerical team identifiers (0 or 1) for a batch of player image crops.
        """
        if not self._is_fitted:
            raise ValueError("TeamClassifier has not been fitted! Please call fit() with sample crops first.")
            
        if not player_crops:
            return []
            
        # Roboflow classifier returns a numpy array of labels (usually 0 and 1)
        labels = self._classifier.predict(player_crops)
        return [int(lbl) for lbl in labels]

    def predict_one(self, crop: np.ndarray) -> Optional[int]:
        """
        Predict team identifier for a single player image crop.
        """
        preds = self.predict([crop])
        return preds[0] if preds else None
