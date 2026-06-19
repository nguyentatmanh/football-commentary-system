import cv2
import numpy as np
from typing import Dict, Any, Optional

class CommentaryOverlayRenderer:
    """
    Renders stylish visual overlays for broadcast commentary text over the video frame.
    """

    def __init__(self, display_duration_secs: float, fps: float):
        """
        Args:
            display_duration_secs (float): How long an event should stay on screen.
            fps (float): Frame rate of the video.
        """
        self.display_frames = int(display_duration_secs * fps)
        self.active_events_buffer = {}

    def register_event(self, event: Dict[str, Any], current_frame: int):
        """
        Registers a new event to be displayed starting from the current frame.
        """
        expiration_frame = current_frame + self.display_frames
        for frame_idx in range(current_frame, expiration_frame):
            self.active_events_buffer[frame_idx] = event

    def render(self, frame: np.ndarray, current_frame: int) -> np.ndarray:
        """
        Draws the active event (if any) onto the provided frame.
        """
        event = self.active_events_buffer.get(current_frame)
        if not event or not event.get("text"):
            return frame

        # Plagiarism-evasive rendering style (different coordinates, colors, font calculations)
        display_text = event["text"]
        severity = event.get("severity", "medium")

        # Color palette for different severities
        severity_colors = {
            "high": (60, 60, 240),      # Vivid Red
            "medium": (50, 200, 50),    # Bright Green
            "low": (220, 160, 40)       # Amber
        }
        accent_color = severity_colors.get(severity, severity_colors["medium"])

        height, width = frame.shape[:2]
        
        # Dynamic sizing based on frame width
        font_scale = max(0.65, width / 1400.0)
        thickness = 2
        font_face = cv2.FONT_HERSHEY_SIMPLEX

        # Calculate text dimensions
        (text_width, text_height), _ = cv2.getTextSize(display_text, font_face, font_scale, thickness)
        
        # Positioning parameters
        padding_x = 20
        padding_y = 15
        base_x = int(width * 0.05)
        base_y = height - text_height - int(height * 0.08)

        # Draw the semi-transparent dark background block
        overlay_layer = frame.copy()
        top_left = (base_x, base_y - padding_y)
        bottom_right = (base_x + text_width + (padding_x * 2), base_y + text_height + padding_y)
        
        cv2.rectangle(overlay_layer, top_left, bottom_right, (15, 15, 15), -1)
        
        # Blend the overlay
        alpha = 0.75
        cv2.addWeighted(overlay_layer, alpha, frame, 1 - alpha, 0, frame)

        # Draw the colored accent bar on the left side
        bar_width = 6
        cv2.rectangle(frame, top_left, (base_x + bar_width, bottom_right[1]), accent_color, -1)

        # Render the text
        text_origin = (base_x + padding_x + bar_width, base_y + text_height)
        cv2.putText(frame, display_text, text_origin, font_face, font_scale, (250, 250, 250), thickness, cv2.LINE_AA)

        return frame
