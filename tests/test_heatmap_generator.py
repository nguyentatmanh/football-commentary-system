import pytest
import numpy as np
import os
import shutil
from unittest.mock import patch
from football_ai.analytics.heatmap_generator import HeatmapGenerator

def test_heatmap_generation_non_empty():
    """Ensures generator populates intensity grid and returns valid blended frames."""
    gen = HeatmapGenerator()
    
    # Generate dummy trajectory (cluster near center)
    dummy_pts = [[52.5, 34.0]] * 50
    
    img = gen.generate_heatmap(dummy_pts)
    
    assert img is not None
    assert img.shape == gen.pitch_base.shape
    
    # Image should NOT be identical to the base pitch diagram
    # because it has been smeared with Heatmap colors (Jet)
    assert not np.array_equal(img, gen.pitch_base)

def test_heatmap_batch_file_creation(tmp_path):
    """Validates file-saving procedures and dynamic label rendering to filesystem."""
    gen = HeatmapGenerator()
    
    # Inject temp output directory
    temp_dir = str(tmp_path)
    
    grouped = {
        "Test Team": [[10,10], [20,20], [30,30]],
        "Test Ball": [[50,50]]
    }
    
    gen.save_heatmaps(temp_dir, grouped)
    
    # Verify files created under target subfolder
    hmap_path = os.path.join(temp_dir, "heatmaps")
    assert os.path.exists(hmap_path)
    assert os.path.exists(os.path.join(hmap_path, "test_team_heatmap.png"))
    assert os.path.exists(os.path.join(hmap_path, "test_ball_heatmap.png"))
