import cv2
import numpy as np
from typing import List, Tuple, Dict
from football_ai.tracking.track_state import TrackState

class Annotator:
    """
    Handles drawing tracking annotations (ellipses, boxes, markers, labels) 
    onto video frames using modern aesthetic colors.
    """
    # Define BGR Color Palette
    COLORS = {
        "team_0": (147, 20, 255),     # Radiant Pink/Magenta (Team A)
        "team_1": (255, 191, 0),      # Ocean Azure/Sky Blue (Team B)
        "player": (200, 200, 200),    # Neutral Grey (fallback)
        "goalkeeper": (0, 140, 255),  # Energetic Orange
        "referee": (100, 255, 100),   # Lime Green
        "ball": (0, 223, 252),        # Bright Gold/Yellow
        "text": (255, 255, 255),      # Pure White
        "label_bg": (40, 40, 40)      # Dark Charcoal
    }

    def __init__(self, line_thickness: int = 2, font_scale: float = 0.5):
        self.line_thickness = line_thickness
        self.font_scale = font_scale

    def _get_entity_color(self, track: TrackState) -> Tuple[int, int, int]:
        """Resolve entity BGR color based on role and team affiliation."""
        if track.role == "ball":
            return self.COLORS["ball"]
        if track.role == "referee":
            return self.COLORS["referee"]
        
        # Handle team specific assignments
        if track.team_id == 0:
            return self.COLORS["team_0"]
        if track.team_id == 1:
            return self.COLORS["team_1"]
            
        # Fallback
        return self.COLORS.get(track.role, (180, 180, 180))

    def draw_label(self, img: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int]):
        """Draws a text label with a solid background rectangle."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        
        # Get text size for background box
        (tw, th), baseline = cv2.getTextSize(text, font, self.font_scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(
            img,
            (x, y - th - 6),
            (x + tw + 6, y + baseline),
            self.COLORS["label_bg"],
            -1
        )
        # Draw boundary highlight
        cv2.rectangle(
            img,
            (x, y - th - 6),
            (x + tw + 6, y + baseline),
            color,
            1
        )
        # Draw white text inside
        cv2.putText(
            img,
            text,
            (x + 3, y - 2),
            font,
            self.font_scale,
            self.COLORS["text"],
            thickness,
            cv2.LINE_AA
        )

    def annotate_frame(self, frame: np.ndarray, tracks: List[TrackState]) -> np.ndarray:
        """
        Render tracked entities onto a copy of the frame.
        """
        out_frame = frame.copy()

        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox_xyxy)
            role = track.role
            color = self._get_entity_color(track)

            # 1. Draw Ball Annotation
            if role == "ball":
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                # Draw outer glowing ring
                cv2.circle(out_frame, (cx, cy), 12, (255, 255, 255), 1, cv2.LINE_AA)
                # Draw core fill
                cv2.circle(out_frame, (cx, cy), 6, color, -1, cv2.LINE_AA)
                # Optional triangle pointer above ball
                pts = np.array([
                    [cx, cy - 15],
                    [cx - 7, cy - 27],
                    [cx + 7, cy - 27]
                ])
                cv2.fillPoly(out_frame, [pts], color)
                continue

            # 2. Draw Human Annotations (Players, Goalkeepers, Referees)
            if role == "player":
                cx = (x1 + x2) // 2
                width = (x2 - x1) // 2
                height = int(width * 0.35) # Flattened ellipse
                
                # Draw filled transparent-like ellipse base by blending
                ellipse_overlay = out_frame.copy()
                cv2.ellipse(
                    ellipse_overlay, 
                    (cx, y2), 
                    (width, height), 
                    0, 0, 360, 
                    color, 
                    -1, 
                    cv2.LINE_AA
                )
                cv2.addWeighted(ellipse_overlay, 0.4, out_frame, 0.6, 0, out_frame)
                
                # Outer solid ring
                cv2.ellipse(
                    out_frame, 
                    (cx, y2), 
                    (width, height), 
                    0, 0, 360, 
                    color, 
                    self.line_thickness, 
                    cv2.LINE_AA
                )
            else:
                # Draw bounding box for Referee / Goalkeeper
                # Keep goalkeeper visually distinct even if assigned to team color
                if role == "goalkeeper":
                    # Double ring boundary or thicker rectangle
                    cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, self.line_thickness + 1, cv2.LINE_AA)
                    # Internal accent border
                    cv2.rectangle(out_frame, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, self.line_thickness, cv2.LINE_AA)

            # 3. Draw Labels (ID, Team, and Role)
            label_str = f"#{track.track_id}"
            
            if role == "player":
                if track.team_id == 0:
                    label_str += " T-A"
                elif track.team_id == 1:
                    label_str += " T-B"
            elif role == "referee":
                label_str += " REF"
            elif role == "goalkeeper":
                if track.team_id == 0:
                    label_str += " GK-A"
                elif track.team_id == 1:
                    label_str += " GK-B"
                else:
                    label_str += " GK"
                
            # Place label at the top of head
            self.draw_label(out_frame, label_str, x1, y1, color)

        return out_frame
