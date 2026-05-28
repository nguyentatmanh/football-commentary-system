import cv2
import numpy as np
from typing import List, Tuple
import supervision as sv

import football_ai.utils.paths
from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from football_ai.tracking.track_state import TrackState

class RadarVisualizer:
    """
    Generates a 2D Top-Down Radar perspective image mapping players onto 
    a standardized football field diagram.
    """
    def __init__(self, width_m: float = 105.0, height_m: float = 68.0, scale: float = 0.1):
        # Match the same dimension config as mapping (stored in cm)
        self.pitch_config = SoccerPitchConfiguration(
            length=int(width_m * 100), 
            width=int(height_m * 100)
        )
        self.scale = scale
        self.padding = 40
        
        # Draw base pitch diagram (reused to minimize redrawing time)
        self.base_pitch_diagram = draw_pitch(
            config=self.pitch_config,
            padding=self.padding,
            scale=self.scale,
            background_color=sv.Color(30, 50, 30), # Deep elegant dark-green turf
            line_color=sv.Color(200, 200, 200),    # Silver-grey clean boundaries
            line_thickness=2
        )

    def _get_bgr_colors(self) -> dict:
        """Synchronize colors from central Annotator scheme."""
        return {
            "team_0": (147, 20, 255),     # Radiant Pink/Magenta (Team A)
            "team_1": (255, 191, 0),      # Ocean Azure/Sky Blue (Team B)
            "goalkeeper": (0, 140, 255),  # Energetic Orange
            "referee": (100, 255, 100),   # Lime Green
            "ball": (0, 223, 252),        # Bright Gold/Yellow
        }

    def _meters_to_pixels(self, pt_m: List[float]) -> Tuple[int, int]:
        """Convert meter coordinates to canvas pixel coordinates."""
        cm_x, cm_y = pt_m[0] * 100.0, pt_m[1] * 100.0
        px = int(cm_x * self.scale) + self.padding
        py = int(cm_y * self.scale) + self.padding
        return px, py

    def generate_radar_image(self, tracks: List[TrackState]) -> np.ndarray:
        """
        Renders a top-down view populated by mapped player states.
        """
        # Copy the pristine base turf
        img = self.base_pitch_diagram.copy()
        
        # Separate entities to draw with customized styles/ordering
        colors = self._get_bgr_colors()
        
        # Human categories
        t0_pts = []
        t1_pts = []
        gk_pts = []
        ref_pts = []
        ball_pts = []
        
        for t in tracks:
            if t.pitch_xy is None:
                continue
                
            # IMPORTANT: Scale meters back to centimeters for Roboflow drawer expectations
            scaled_pt = [t.pitch_xy[0] * 100.0, t.pitch_xy[1] * 100.0]
            
            if t.role == "ball":
                ball_pts.append(scaled_pt)
            elif t.role == "referee":
                ref_pts.append(scaled_pt)
            elif t.role == "goalkeeper":
                gk_pts.append(scaled_pt)
            elif t.role == "player":
                if t.team_id == 0:
                    t0_pts.append(scaled_pt)
                elif t.team_id == 1:
                    t1_pts.append(scaled_pt)

        # Draw points sequentially (layer order: Referees -> Team A -> Team B -> Goalkeepers -> Ball)
        if ref_pts:
            img = draw_points_on_pitch(
                config=self.pitch_config, xy=np.array(ref_pts),
                face_color=sv.Color(*colors["referee"]),
                edge_color=sv.Color.WHITE,
                radius=8, thickness=1,
                padding=self.padding, scale=self.scale, pitch=img
            )

        if t0_pts:
            img = draw_points_on_pitch(
                config=self.pitch_config, xy=np.array(t0_pts),
                face_color=sv.Color(*colors["team_0"]),
                edge_color=sv.Color.WHITE,
                radius=9, thickness=1,
                padding=self.padding, scale=self.scale, pitch=img
            )

        if t1_pts:
            img = draw_points_on_pitch(
                config=self.pitch_config, xy=np.array(t1_pts),
                face_color=sv.Color(*colors["team_1"]),
                edge_color=sv.Color.WHITE,
                radius=9, thickness=1,
                padding=self.padding, scale=self.scale, pitch=img
            )

        if gk_pts:
            # Double border accent for goalkeepers
            img = draw_points_on_pitch(
                config=self.pitch_config, xy=np.array(gk_pts),
                face_color=sv.Color(*colors["goalkeeper"]),
                edge_color=sv.Color.WHITE,
                radius=11, thickness=2,
                padding=self.padding, scale=self.scale, pitch=img
            )

        if ball_pts:
            # Glowy gold ring for the ball
            img = draw_points_on_pitch(
                config=self.pitch_config, xy=np.array(ball_pts),
                face_color=sv.Color(*colors["ball"]),
                edge_color=sv.Color.WHITE,
                radius=6, thickness=2,
                padding=self.padding, scale=self.scale, pitch=img
            )
            
        return img
