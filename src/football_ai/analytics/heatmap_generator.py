import os
import cv2
import numpy as np
import supervision as sv
from typing import List, Tuple

import football_ai.utils.paths
from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import draw_pitch

class HeatmapGenerator:
    """
    Generates smoothed 2D spatial occupancy heatmaps blended over 
    a standardized football pitch background image.
    """
    def __init__(self, width_m: float = 105.0, height_m: float = 68.0, scale: float = 0.1):
        self.width_cm = int(width_m * 100)
        self.height_cm = int(height_m * 100)
        self.scale = scale
        self.padding = 40
        
        self.pitch_config = SoccerPitchConfiguration(length=self.width_cm, width=self.height_cm)
        
        # Draw premium turf background base for blending
        self.pitch_base = draw_pitch(
            config=self.pitch_config,
            padding=self.padding,
            scale=self.scale,
            background_color=sv.Color(30, 50, 30), # elegant dark green
            line_color=sv.Color(150, 150, 150),   # subtle grey
            line_thickness=2
        )
        
        self.img_h, self.img_w, _ = self.pitch_base.shape

    def _meters_to_pixel(self, mx: float, my: float) -> Tuple[int, int]:
        """Resolves meter positions into static canvas grid coordinates."""
        cx = int(mx * 100.0 * self.scale) + self.padding
        cy = int(my * 100.0 * self.scale) + self.padding
        return cx, cy

    def generate_heatmap(self, trajectory_points: List[List[float]], sigma: int = 15) -> np.ndarray:
        """
        Aggregates 2D positions, smears densities using Gaussian blur, 
        applies Jet colormap gradient, and blends dynamically onto base turf.
        """
        if not trajectory_points:
            return self.pitch_base.copy()

        # 1. Initialize density accumulator array
        accumulator = np.zeros((self.img_h, self.img_w), dtype=np.float32)

        # 2. Increment points
        for pt in trajectory_points:
            if len(pt) < 2:
                continue
            px, py = self._meters_to_pixel(pt[0], pt[1])
            
            # Clamp boundaries to prevent out-of-bounds index crash
            if 0 <= px < self.img_w and 0 <= py < self.img_h:
                accumulator[py, px] += 1.0

        # 3. Smear density using heavy Gaussian kernel for buttery visualization
        if np.sum(accumulator) > 0:
            # Adjust kernel based on trajectory volumes
            ksize = max(3, int(6 * sigma) | 1) # Enforce odd
            blurred = cv2.GaussianBlur(accumulator, (ksize, ksize), sigma)
            
            # Normalize 0 to 1.0
            max_val = np.max(blurred)
            if max_val > 0:
                normalized = blurred / max_val
            else:
                normalized = blurred
        else:
            normalized = accumulator

        # 4. Map density gradient (Jet)
        # Scale to 8-bit grayscale [0-255]
        gray_heatmap = (normalized * 255.0).astype(np.uint8)
        
        # Jet colormap: Blue (cold) -> Green -> Yellow -> Red (hot)
        color_heatmap = cv2.applyColorMap(gray_heatmap, cv2.COLORMAP_JET)
        
        # 5. Composite blend
        # We mask out cold background regions (near-zero) so we don't paint the whole field blue.
        # Threshold: only show heat above 5% intensity
        alpha_mask = (normalized > 0.05).astype(np.float32)
        # Smooth transition using the normalized intensities themselves
        alpha_mask = np.clip(normalized * 1.5, 0.0, 0.85) # Cap at 85% opacity
        
        # Broaden mask shape to RGB
        alpha_3ch = np.dstack([alpha_mask] * 3)
        
        # Perform Alpha Composition Blend
        blended = (color_heatmap * alpha_3ch + self.pitch_base * (1.0 - alpha_3ch)).astype(np.uint8)
        
        return blended

    def save_heatmaps(self, output_dir: str, grouped_points: dict):
        """
        Helper performing batch rendering of heatmaps and writing PNG formats.
        Args:
            output_dir: Base folder to save heatmaps subfolder
            grouped_points: dict of name -> list of [x,y]
        """
        hmap_dir = os.path.join(output_dir, "heatmaps")
        os.makedirs(hmap_dir, exist_ok=True)
        
        for name, pts in grouped_points.items():
            hmap_img = self.generate_heatmap(pts)
            filename = f"{name.lower().replace(' ', '_')}_heatmap.png"
            out_path = os.path.join(hmap_dir, filename)
            
            # Draw text label identifying the heatmap type in the corner
            cv2.putText(
                hmap_img, 
                f"OCCUPANCY: {name.upper()}", 
                (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.75, 
                (255, 255, 255), 
                2, 
                cv2.LINE_AA
            )
            
            cv2.imwrite(out_path, hmap_img)
