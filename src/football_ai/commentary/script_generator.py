from typing import List, Dict, Any
import re

from football_ai.events.event import FootballEvent
from football_ai.commentary.templates_vi import get_template, FALLBACK_TEMPLATES

class ScriptGenerator:
    """
    Converts extracted FootballEvents into readable text commentary items.
    Applies density filters and maps dynamic placeholders deterministically.
    """
    def __init__(
        self,
        min_gap_seconds: float = 4.0,
        max_events_per_minute: int = 6,
        min_priority: int = 5
    ):
        self.min_gap_seconds = min_gap_seconds
        self.max_events_per_minute = max_events_per_minute
        self.min_priority = min_priority

    def generate(self, events: List[FootballEvent]) -> List[Dict[str, Any]]:
        if not events:
            return []

        raw_script = []
        for idx, event in enumerate(events):
            # 1. Basic Priority Filter
            if event.priority < self.min_priority:
                continue

            # Resolve dynamic fields
            team_str = None
            if event.team_id == 0:
                team_str = "đội A"
            elif event.team_id == 1:
                team_str = "đội B"

            player_id = event.player_id

            # Resolve Emotion & Speech Pace
            emotion = "neutral"
            speed = "normal"
            if event.type in ["deep_entry"]:
                emotion = "excited"
                speed = "fast"
            elif event.type in ["high_speed_run"]:
                emotion = "excited"
                speed = "fast"
            elif event.type in ["possession_change"]:
                emotion = "neutral"
                speed = "normal"

            # Get deterministic Seed for template based on event sequence index
            try:
                event_index = int(re.search(r'\d+', event.event_id).group())
            except:
                event_index = idx

            text = ""
            # Avoid templates requiring fields if fields missing
            if event.type == "high_speed_run" and player_id is None:
                text = FALLBACK_TEMPLATES["high_speed_run"]
            elif event.type in ["possession_change", "deep_entry", "attack_build_up"] and team_str is None:
                text = FALLBACK_TEMPLATES.get(event.type, "Trận đấu đang tiếp diễn.")
            else:
                template = get_template(event.type, event_index)
                # Inject variables safely
                text = template.format(team=team_str or "", player_id=player_id or "")

            raw_script.append({
                "event_id": event.event_id,
                "start_time": event.start_time,
                "end_time": event.end_time,
                "text": text,
                "emotion": emotion,
                "speed": speed,
                "priority": event.priority,
                "should_speak": True # Will be flagged False if filtered out
            })

        # 2. Sort by start time
        raw_script.sort(key=lambda x: x["start_time"])

        # 3. Apply Temporal Density Filters
        final_script = []
        last_spoken_time = -999.0
        minute_buckets = {} # minute_int -> count of spoken

        for item in raw_script:
            t = item["start_time"]
            minute_bucket = int(t / 60.0)
            
            # Check absolute gap
            gap_ok = (t - last_spoken_time) >= self.min_gap_seconds
            
            # Check minute bucket cap
            current_minute_count = minute_buckets.get(minute_bucket, 0)
            bucket_ok = current_minute_count < self.max_events_per_minute

            if gap_ok and bucket_ok:
                item["should_speak"] = True
                last_spoken_time = t
                minute_buckets[minute_bucket] = current_minute_count + 1
            else:
                item["should_speak"] = False

            final_script.append(item)

        return final_script
