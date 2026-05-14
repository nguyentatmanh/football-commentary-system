import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from football_ai.events.event import FootballEvent

class EventExtractor:
    """
    Extracts standard FootballEvents from analytics tracking records.
    Features robust handling for missing data and deterministic event IDs.
    """
    def __init__(self, grab_radius_m: float = 2.5, fps: float = 30.0):
        self.grab_radius_m = grab_radius_m
        self.fps = fps
        self.event_counter = 1

    def _create_event(self, event_type: str, start_frame: int, end_frame: int,
                      team_id: int | None, player_id: int | None, 
                      priority: int, confidence: float = 1.0, metadata: Dict[str, Any] = None) -> FootballEvent:
        event_id = f"event_{self.event_counter:04d}"
        self.event_counter += 1
        return FootballEvent(
            event_id=event_id,
            type=event_type,
            start_frame=start_frame,
            end_frame=end_frame,
            start_time=start_frame / self.fps,
            end_time=end_frame / self.fps,
            team_id=team_id,
            player_id=player_id,
            confidence=confidence,
            priority=priority,
            metadata=metadata or {}
        )

    def extract(self, tracking_data: List[Dict[str, Any]]) -> List[FootballEvent]:
        if not tracking_data:
            return []

        # 1. Group tracks by frame
        frames_dict = defaultdict(list)
        max_frame = 0
        for t in tracking_data:
            f_idx = t["frame_index"]
            frames_dict[f_idx].append(t)
            if f_idx > max_frame:
                max_frame = f_idx

        # Sorted list of frames present
        sorted_frames = sorted(frames_dict.keys())
        if not sorted_frames:
            return []

        # 2. Compute raw timeline for possession and ball positions
        # timeline[frame] = team_id or None
        raw_possession: Dict[int, int | None] = {}
        ball_pos: Dict[int, Tuple[float, float] | None] = {}
        
        for f_idx in sorted_frames:
            tracks = frames_dict[f_idx]
            ball = next((t for t in tracks if t["role"] == "ball" and t.get("pitch_xy") is not None), None)
            
            if ball:
                b_xy = ball["pitch_xy"]
                ball_pos[f_idx] = (b_xy[0], b_xy[1])
            else:
                ball_pos[f_idx] = None
                raw_possession[f_idx] = None
                continue
                
            # Find closest player
            b_np = np.array(b_xy)
            players = [t for t in tracks if t["role"] in ["player", "goalkeeper"] and t.get("pitch_xy") is not None and t.get("team_id") is not None]
            
            min_d = float("inf")
            closest_team = None
            for p in players:
                p_np = np.array(p["pitch_xy"])
                dist = float(np.linalg.norm(p_np - b_np))
                if dist < min_d:
                    min_d = dist
                    closest_team = p["team_id"]
            
            if min_d <= self.grab_radius_m:
                raw_possession[f_idx] = closest_team
            else:
                raw_possession[f_idx] = None

        # 3. Compute smoothed stable possession
        # Require N frames to change stable state
        stable_possession: Dict[int, int | None] = {}
        current_stable = None
        consecutive_frames = 0
        
        # Set up parameters
        STABLE_FRAMES = int(0.3 * self.fps)  # 9 frames at 30fps
        LOOSE_FRAMES = int(1.0 * self.fps)   # 30 frames at 30fps
        
            
        # Let's rewrite the smoothing cleanly
        smoothed = []
        win = int(0.4 * self.fps) # 12 frames window
        for i in range(len(sorted_frames)):
            start = max(0, i - win // 2)
            end = min(len(sorted_frames), i + win // 2 + 1)
            sub = [raw_possession.get(sorted_frames[j]) for j in range(start, end)]
            counts_0 = sub.count(0)
            counts_1 = sub.count(1)
            counts_none = sub.count(None)
            
            if counts_0 > len(sub) / 2:
                smoothed.append(0)
            elif counts_1 > len(sub) / 2:
                smoothed.append(1)
            else:
                smoothed.append(None)
        
        stable_timeline = {sorted_frames[i]: smoothed[i] for i in range(len(sorted_frames))}

        events: List[FootballEvent] = []

        # Event extraction cooldown maps to avoid spamming
        cooldowns = defaultdict(float)

        def is_on_cooldown(key: str, time: float) -> bool:
            return time < cooldowns[key]

        def set_cooldown(key: str, time: float, duration: float):
            cooldowns[key] = time + duration

        # 4. DETECT EVENT: possession_change & loose_ball
        prev_stable = None
        loose_start_frame = None
        
        for i, f_idx in enumerate(sorted_frames):
            curr_stable = stable_timeline[f_idx]
            curr_time = f_idx / self.fps
            
            # Change event
            if prev_stable is not None and curr_stable is not None and prev_stable != curr_stable:
                # Transition between teams
                if not is_on_cooldown("possession_change", curr_time):
                    events.append(self._create_event(
                        event_type="possession_change",
                        start_frame=f_idx,
                        end_frame=f_idx + int(self.fps),
                        team_id=curr_stable,
                        player_id=None,
                        priority=6,
                        metadata={"from_team": prev_stable, "to_team": curr_stable}
                    ))
                    set_cooldown("possession_change", curr_time, 5.0)
            
            # Track loose ball
            if curr_stable is None:
                if loose_start_frame is None:
                    loose_start_frame = f_idx
            else:
                if loose_start_frame is not None:
                    duration_f = f_idx - loose_start_frame
                    duration_s = duration_f / self.fps
                    if duration_s >= 1.5:
                        # Loose ball event occurred
                        if not is_on_cooldown("loose_ball", loose_start_frame / self.fps):
                            events.append(self._create_event(
                                event_type="loose_ball",
                                start_frame=loose_start_frame,
                                end_frame=f_idx,
                                team_id=None,
                                player_id=None,
                                priority=5,
                                metadata={"duration_seconds": duration_s}
                            ))
                            set_cooldown("loose_ball", loose_start_frame / self.fps, 8.0)
                    loose_start_frame = None
            
            prev_stable = curr_stable

        # 5. DETECT EVENT: deep_entry
        # Ball transitions from middle zone to final thirds
        prev_ball_x = None
        for f_idx in sorted_frames:
            curr_pos = ball_pos.get(f_idx)
            curr_time = f_idx / self.fps
            if curr_pos is None:
                prev_ball_x = None
                continue
            
            curr_x = curr_pos[0]
            curr_team = stable_timeline.get(f_idx)
            
            if prev_ball_x is not None and curr_team is not None:
                # Entering left zone (X < 35) or right zone (X > 70)
                entered_left = (prev_ball_x >= 35.0 and curr_x < 35.0)
                entered_right = (prev_ball_x <= 70.0 and curr_x > 70.0)
                
                if (entered_left or entered_right) and not is_on_cooldown("deep_entry", curr_time):
                    events.append(self._create_event(
                        event_type="deep_entry",
                        start_frame=f_idx,
                        end_frame=f_idx + int(1.5 * self.fps),
                        team_id=curr_team,
                        player_id=None,
                        priority=8,
                        metadata={"zone": "left_third" if entered_left else "right_third", "x_coord": curr_x}
                    ))
                    set_cooldown("deep_entry", curr_time, 10.0)
            
            prev_ball_x = curr_x

        # 6. DETECT EVENT: high_speed_run
        # Track historical coordinates for each player
        player_paths = defaultdict(list) # player_id -> list of (frame_index, pitch_xy)
        
        for f_idx in sorted_frames:
            tracks = frames_dict[f_idx]
            for t in tracks:
                if t["role"] in ["player", "goalkeeper"] and t.get("pitch_xy") is not None:
                    pid = t["track_id"]
                    player_paths[pid].append((f_idx, t["pitch_xy"]))

        for pid, path in player_paths.items():
            # We calculate speed across a delta of 10 frames (~0.33s)
            lookback = 10
            for idx in range(lookback, len(path)):
                f_prev, pos_prev = path[idx - lookback]
                f_curr, pos_curr = path[idx]
                
                dt = (f_curr - f_prev) / self.fps
                if dt <= 0:
                    continue
                
                dist = float(np.linalg.norm(np.array(pos_curr) - np.array(pos_prev)))
                speed_mps = dist / dt
                
                curr_time = f_curr / self.fps
                
                if speed_mps > 7.0 and not is_on_cooldown(f"high_speed_{pid}", curr_time):
                    # Find player's team
                    p_tracks = [t for t in frames_dict[f_curr] if t["track_id"] == pid]
                    team_id = p_tracks[0].get("team_id") if p_tracks else None
                    
                    events.append(self._create_event(
                        event_type="high_speed_run",
                        start_frame=f_prev,
                        end_frame=f_curr,
                        team_id=team_id,
                        player_id=pid,
                        priority=7,
                        metadata={"speed_mps": speed_mps}
                    ))
                    # Don't trigger again for this player for 12 seconds
                    set_cooldown(f"high_speed_{pid}", curr_time, 12.0)

        # 7. DETECT EVENT: attack_build_up
        # Possession holds for >= 5s and ball advances >= 15m
        # Walk through the stable possession timeline and detect long stretches
        if len(sorted_frames) > 0:
            build_up_start_idx = None
            current_build_up_team = None
            
            BUILD_UP_FRAMES = int(5.0 * self.fps)
            
            for i, f_idx in enumerate(sorted_frames):
                curr_team = stable_timeline[f_idx]
                
                if curr_team != current_build_up_team or curr_team is None:
                    # Reset and check if previous stretch was valid build-up
                    if current_build_up_team is not None and build_up_start_idx is not None:
                        duration_f = sorted_frames[i-1] - sorted_frames[build_up_start_idx]
                        if duration_f >= BUILD_UP_FRAMES:
                            # Check ball movement
                            s_pos = ball_pos.get(sorted_frames[build_up_start_idx])
                            e_pos = ball_pos.get(sorted_frames[i-1])
                            if s_pos and e_pos:
                                dx = abs(e_pos[0] - s_pos[0])
                                if dx >= 15.0:
                                    st_time = sorted_frames[build_up_start_idx] / self.fps
                                    if not is_on_cooldown("attack_build_up", st_time):
                                        events.append(self._create_event(
                                            event_type="attack_build_up",
                                            start_frame=sorted_frames[build_up_start_idx],
                                            end_frame=sorted_frames[i-1],
                                            team_id=current_build_up_team,
                                            player_id=None,
                                            priority=6,
                                            metadata={"distance_x": dx}
                                        ))
                                        set_cooldown("attack_build_up", st_time, 15.0)
                    
                    current_build_up_team = curr_team
                    build_up_start_idx = i if curr_team is not None else None

        # Sort all events by start_frame
        events.sort(key=lambda x: x.start_frame)
        
        # Re-index IDs to be fully deterministic and sequentially ordered
        for i, ev in enumerate(events):
            ev.event_id = f"event_{i+1:04d}"
            
        return events
