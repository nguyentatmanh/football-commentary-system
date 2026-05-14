import numpy as np
import supervision as sv
from typing import List
from football_ai.detection.detection_result import DetectionResult
from football_ai.tracking.track_state import TrackState
from football_ai.config.schema import TrackingConfig

class Tracker:
    """
    Wraps supervision's ByteTrack to provide consistent tracking identifiers
    across video sequences.
    """
    def __init__(self, config: TrackingConfig):
        # Instantiate ByteTrack
        self.tracker = sv.ByteTrack(
            minimum_consecutive_frames=config.min_consecutive_frames
        )

    def update(self, detections: List[DetectionResult], frame_index: int) -> List[TrackState]:
        """
        Processes a list of raw DetectionResults and produces a list of TrackState.
        Splits out the ball tracking to avoid noisy tracker associations.
        """
        if not detections:
            return []

        # Separate the ball from human players/referees
        trackable_detections = []
        ball_detections = []
        
        for det in detections:
            if det.role == "ball":
                ball_detections.append(det)
            else:
                trackable_detections.append(det)

        track_states = []

        # 1. Track the human actors using ByteTrack
        if trackable_detections:
            xyxy = np.array([det.bbox_xyxy for det in trackable_detections])
            confidence = np.array([det.confidence for det in trackable_detections])
            class_id = np.array([det.class_id for det in trackable_detections])

            sv_detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id
            )
            
            # ByteTrack update
            tracked_sv = self.tracker.update_with_detections(sv_detections)
            
            # Note: supervision tracker returns a subset of detections that are currently active.
            # We reconstruct classes and roles based on the output class_id mapping.
            # Because tracker_id is guaranteed to be provided by update_with_detections:
            if tracked_sv.tracker_id is not None:
                # Create mapping from class_id to human-readable role for trackable objects
                # Standard mapping as implemented in FootballDetector:
                # 1 -> goalkeeper, 2 -> player, 3 -> referee
                role_map = {1: "goalkeeper", 2: "player", 3: "referee"}
                
                for i in range(len(tracked_sv)):
                    cls_val = int(tracked_sv.class_id[i])
                    t_id = int(tracked_sv.tracker_id[i])
                    state = TrackState(
                        frame_index=frame_index,
                        track_id=t_id,
                        class_id=cls_val,
                        class_name=role_map.get(cls_val, "unknown"),
                        role=role_map.get(cls_val, "unknown"),
                        confidence=float(tracked_sv.confidence[i]),
                        bbox_xyxy=tracked_sv.xyxy[i].tolist()
                    )
                    track_states.append(state)

        # 2. Process the ball separately
        # Usually there is only one active ball. If multiple, we take the highest confidence one.
        # We assign constant track_id = -1 to the ball, ensuring it stays identifiable.
        if ball_detections:
            # Get highest confidence ball
            best_ball = max(ball_detections, key=lambda d: d.confidence)
            ball_state = TrackState(
                frame_index=frame_index,
                track_id=-1, # Constant ball identifier
                class_id=best_ball.class_id,
                class_name="ball",
                role="ball",
                confidence=best_ball.confidence,
                bbox_xyxy=best_ball.bbox_xyxy
            )
            track_states.append(ball_state)

        return track_states
