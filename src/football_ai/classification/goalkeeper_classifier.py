import numpy as np
from typing import List, Optional, Tuple
from football_ai.tracking.track_state import TrackState

class GoalkeeperClassifier:
    """
    Resolves goalkeeper team membership by measuring geometric proximity
    between the goalkeeper and the average position of other players assigned to Team 0 and 1.
    """
    
    def _get_bottom_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Calculates the bottom-center anchor of a bounding box."""
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, y2

    def resolve_team_ids(self, tracks: List[TrackState]) -> List[TrackState]:
        """
        Examine the frame's tracks, compute centroids for Team 0 and Team 1,
        and assign goalkeepers to the nearest team.
        """
        # 1. Separate players with assigned teams and goalkeepers without assigned teams
        team_0_positions = []
        team_1_positions = []
        goalkeepers = []

        for track in tracks:
            if track.role == "player" and track.team_id is not None:
                pos = self._get_bottom_center(track.bbox_xyxy)
                if track.team_id == 0:
                    team_0_positions.append(pos)
                elif track.team_id == 1:
                    team_1_positions.append(pos)
            elif track.role == "goalkeeper":
                goalkeepers.append(track)

        if not goalkeepers:
            return tracks

        # If we don't have representation for BOTH teams, we can't compare distance meaningfully.
        # Leave them as None.
        if not team_0_positions or not team_1_positions:
            for gk in goalkeepers:
                gk.team_id = None
            return tracks

        # 2. Calculate centroids
        centroid_0 = np.mean(team_0_positions, axis=0)
        centroid_1 = np.mean(team_1_positions, axis=0)

        # 3. Resolve distances for each goalkeeper
        for gk in goalkeepers:
            gk_pos = np.array(self._get_bottom_center(gk.bbox_xyxy))
            
            dist_0 = np.linalg.norm(gk_pos - centroid_0)
            dist_1 = np.linalg.norm(gk_pos - centroid_1)
            
            # Assign to nearest centroid
            assigned_team = 0 if dist_0 < dist_1 else 1
            gk.team_id = assigned_team
            
        return tracks
