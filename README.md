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

### 3. Run the Demo

```cmd
python -m football_ai.cli ^
  --config configs/default.yaml ^
  --input data/raw/sample.mp4 ^
  --output data/outputs/demo ^
  --device cpu
```

## Documentation
Details are available in the [docs/](docs/) folder.
- [Architecture](docs/ARCHITECTURE.md)
- [Pipeline details](docs/PIPELINE.md)
- [Modules description](docs/MODULES.md)
