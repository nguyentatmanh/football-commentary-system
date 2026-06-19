"""
Match Event Detection Engine.

This module provides the logic to interpret raw tracking data and bounding boxes,
extracting significant gameplay events such as ball possession changes, passes,
and shots on goal. All algorithms rely on dynamic distance calculations and 
trajectory history to provide a robust event feed.
"""

import math
from collections import deque
from typing import List, Dict, Tuple, Optional, Any

class MatchEventDetector:
    """
    Analyzes track data across consecutive frames to emit high-level match events.
    """

    def __init__(self, fps: float, config: Any):
        """
        Initializes the MatchEventDetector.

        Args:
            fps (float): Frames per second of the video stream.
            config (Any): The match_events configuration object.
        """
        self.fps = fps
        self.config = config

        self.ball_trajectory = deque(maxlen=self.config.ball_trajectory_history_len)
        self.player_trajectories = {}

        self.current_possessor_id = None
        self.ball_speed = 0.0
        self.frame_index = 0

        self.cooldown_frames = self.config.event_cooldown_frames
        self.last_event_emitted_frame = -999

    def _calculate_distance(self, pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
        """Helper to calculate Euclidean distance."""
        return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

    def _determine_possession(self, players: List[Any], ball_position: Tuple[float, float]) -> Optional[int]:
        """
        Determines which player (if any) currently has possession of the ball based on
        dynamic bounding box height ratios.
        """
        if not ball_position:
            return None

        closest_player_id = None
        min_distance = float('inf')

        bx, by = ball_position

        for player in players:
            # Assuming player object has bounding box attributes
            x1, y1, x2, y2 = player.bbox_xyxy
            px = (x1 + x2) / 2.0
            
            # Distance calculated from player's feet
            feet_y = y2
            bbox_height = y2 - y1

            dist = self._calculate_distance((bx, by), (px, feet_y))
            
            # Dynamic proximity threshold based on player's apparent scale
            dynamic_threshold = bbox_height * self.config.possession_proximity_ratio

            if dist < dynamic_threshold and dist < min_distance:
                min_distance = dist
                closest_player_id = player.track_id

        return closest_player_id

    def process_frame(self, tracks: List[Any]) -> Optional[Dict[str, Any]]:
        """
        Ingests the current frame's tracks and evaluates for state changes.

        Args:
            tracks (List[Any]): Active track objects for the current frame.

        Returns:
            Optional[Dict[str, Any]]: An event dictionary if a notable event occurred, else None.
        """
        self.frame_index += 1
        
        players = []
        ball_position = None

        for t in tracks:
            if t.role == "ball":
                x1, y1, x2, y2 = t.bbox_xyxy
                ball_position = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                self.ball_trajectory.append(ball_position)
            elif t.role in ["player", "goalkeeper", "referee"]:
                if t.role != "referee":
                    players.append(t)
                
                # Update player trajectory history
                if t.track_id not in self.player_trajectories:
                    self.player_trajectories[t.track_id] = deque(maxlen=self.config.ball_trajectory_history_len)
                x1, y1, x2, y2 = t.bbox_xyxy
                self.player_trajectories[t.track_id].append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

        # Calculate current ball velocity vector magnitude
        if len(self.ball_trajectory) >= 2:
            dx = self.ball_trajectory[-1][0] - self.ball_trajectory[-2][0]
            dy = self.ball_trajectory[-1][1] - self.ball_trajectory[-2][1]
            self.ball_speed = math.hypot(dx, dy)
        else:
            self.ball_speed = 0.0

        active_possessor = self._determine_possession(players, ball_position)
        
        emitted_event = None
        frames_since_last_event = self.frame_index - self.last_event_emitted_frame

        if frames_since_last_event > self.cooldown_frames:
            timestamp_sec = round(self.frame_index / self.fps, 2)
            
            # State Transition: Interception or Turnover
            if active_possessor is not None and self.current_possessor_id is not None and active_possessor != self.current_possessor_id:
                emitted_event = {
                    "timestamp": timestamp_sec,
                    "text": f"Player {active_possessor} steals possession from Player {self.current_possessor_id}!",
                    "severity": "high"
                }
                self.last_event_emitted_frame = self.frame_index
            
            # State Transition: Gaining loose ball
            elif active_possessor is not None and self.current_possessor_id is None:
                emitted_event = {
                    "timestamp": timestamp_sec,
                    "text": f"Player {active_possessor} assumes control of the ball.",
                    "severity": "low"
                }
                self.last_event_emitted_frame = self.frame_index

            # State Transition: Pass or Shot detection
            elif self.current_possessor_id is not None and active_possessor is None:
                if self.ball_speed > self.config.pass_velocity_threshold:
                    if self.ball_speed > self.config.pass_velocity_threshold * self.config.shot_velocity_threshold_multiplier:
                        emitted_event = {
                            "timestamp": timestamp_sec,
                            "text": f"Player {self.current_possessor_id} unleashes a powerful shot!",
                            "severity": "high"
                        }
                    else:
                        emitted_event = {
                            "timestamp": timestamp_sec,
                            "text": f"Player {self.current_possessor_id} delivers a progressive pass.",
                            "severity": "medium"
                        }
                    self.current_possessor_id = None
                    self.last_event_emitted_frame = self.frame_index

        if active_possessor is not None:
            self.current_possessor_id = active_possessor

        return emitted_event
