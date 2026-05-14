import cv2
import numpy as np
from typing import Optional

class HomographyEstimator:
    """
    Encapsulates perspective transformations between screen-space pixel 
    coordinates and localized pitch-space coordinates.
    """
    def __init__(self):
        self.matrix = None

    def fit(self, source_points: np.ndarray, target_points: np.ndarray) -> bool:
        """
        Compute perspective projection matrix based on anchor pairs.
        Returns True if successful, False otherwise.
        """
        if source_points.shape != target_points.shape:
            return False
        if len(source_points) < 4:
            return False

        try:
            src = source_points.astype(np.float32)
            dst = target_points.astype(np.float32)
            m, mask = cv2.findHomography(src, dst)
            if m is not None:
                self.matrix = m
                return True
        except Exception as e:
            print(f"[!] Homography computation failed: {e}")
            
        self.matrix = None
        return False

    def is_ready(self) -> bool:
        """Checks if homography transform has been calculated."""
        return self.matrix is not None

    def transform_points(self, points: np.ndarray) -> Optional[np.ndarray]:
        """
        Project 2D image coordinates into the estimated projection plane.
        
        Args:
            points: NumPy array of shape (N, 2)
        Returns:
            NumPy array of transformed coordinates or None.
        """
        if not self.is_ready() or points.size == 0:
            return None

        try:
            if len(points.shape) == 1:
                points = points.reshape(1, 2)
            if points.shape[1] != 2:
                return None

            # Reshape for cv2.perspectiveTransform constraint
            reshaped = points.reshape(-1, 1, 2).astype(np.float32)
            transformed = cv2.perspectiveTransform(reshaped, self.matrix)
            return transformed.reshape(-1, 2).astype(np.float32)
        except Exception as e:
            print(f"[!] Point projection failure: {e}")
            return None
