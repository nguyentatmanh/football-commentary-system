# Football Video Commentary & Analysis System

A clean, modular, production-like football video analysis pipeline built on top of Roboflow's Sports logic.

## Features
- Player, Referee, Goalkeeper, and Ball Detection
- ByteTrack Multi-Object Tracking
- Automatic Team Classification using Feature Clustering
- Ground Plane Radar Projection (Homography mapping)
- Speed & Occupancy Movement Heatmaps
- Rule-based Event Extraction (Possession, Shots, Deep entries)
- Automated Event-driven Commentary Text/JSON

## Quick Start

### 1. Setup Environment

On Windows:
```cmd
scripts\setup_env.bat
```

On Linux/macOS:
```bash
chmod +x scripts/*.sh
./scripts/setup.sh
```

### 2. Download Models & Sample Data

```cmd
python scripts/download_assets.py
```

### 3. Run the Full Analysis & Commentary Demo

Run the end-to-end pipeline including movement analytics and Vietnamese speech synthesis commentary:
```cmd
python -m football_ai.cli ^
  --config configs/default.yaml ^
  --input data/raw/sample.mp4 ^
  --output data/outputs/demo ^
  --device cpu ^
  --mode full ^
  --max-frames 300
```

Alternatively, if you have already run analytics once, you can run the commentary pipeline on its own:
```cmd
python -m football_ai.cli ^
  --config configs/default.yaml ^
  --input data/raw/sample.mp4 ^
  --output data/outputs/demo ^
  --device cpu ^
  --mode commentary
```

**Expected Outputs in output directory:**
- `events.json`: List of detected key events.
- `commentary_script.json`: Script of mapped Vietnamese commentary text.
- `commentary_audio/`: Individual mp3 snippets synthesized via edge-tts.
- `commentary_full.mp3`: Full continuous master audio track overlay.
- `video_with_commentary.mp4`: Composite final video output (if ffmpeg installed).


## Documentation
Details are available in the [docs/](docs/) folder.
- [Architecture](docs/ARCHITECTURE.md)
- [Pipeline details](docs/PIPELINE.md)
- [Modules description](docs/MODULES.md)
