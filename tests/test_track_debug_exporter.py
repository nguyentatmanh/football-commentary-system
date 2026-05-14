import os
import json
import pytest
import numpy as np
import tempfile
from football_ai.classification.track_debug_exporter import TrackDebugExporter
from football_ai.tracking.track_state import TrackState

def _make_track(tid: int, role: str, conf: float) -> TrackState:
    return TrackState(
        frame_index=0,
        track_id=tid,
        class_id=1,
        class_name=role,
        role=role,
        confidence=conf,
        bbox_xyxy=[0, 0, 50, 50],
        team_id=None
    )

def test_track_debug_exporter_generates_files():
    exporter = TrackDebugExporter(enabled=True, max_crops=4)
    
    t1 = _make_track(101, "player", 0.85)
    t2 = _make_track(102, "referee", 0.92)
    
    # Mock crops (simple tiny numpy images)
    dummy_crop = np.zeros((40, 40, 3), dtype=np.uint8)
    
    crops_map = {
        101: dummy_crop,
        102: dummy_crop
    }
    
    # Ingest 3 simulated frames of data
    exporter.ingest_frame_crops([t1, t2], crops_map)
    exporter.ingest_frame_crops([t1, t2], crops_map)
    
    # Use tempdir to test exports
    with tempfile.TemporaryDirectory() as tmpdir:
        exporter.export(tmpdir)
        
        # Verify existence of files
        summary_json = os.path.join(tmpdir, "track_role_summary.json")
        assert os.path.exists(summary_json)
        
        sheets_dir = os.path.join(tmpdir, "track_contact_sheets")
        assert os.path.exists(sheets_dir)
        assert os.path.exists(os.path.join(sheets_dir, "track_101.jpg"))
        assert os.path.exists(os.path.join(sheets_dir, "track_102.jpg"))
        
        # Assert content
        with open(summary_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        assert "101" in data
        assert "102" in data
        assert data["101"]["total_frames"] == 2
        assert data["102"]["original_role_distribution"]["referee"] == 2
        assert round(data["102"]["avg_confidence"], 2) == 0.92
