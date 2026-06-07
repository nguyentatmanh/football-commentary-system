$VIDEO_PATH = "data/raw/sample.mp4"
$MAX_FRAMES = 900
$env:PYTHONPATH = "src"

Write-Host "========================================="
Write-Host " RUNNING 30 SECONDS TEST (CUDA GPU)"
Write-Host "========================================="

# 1. Detection
Write-Host "`n[1/4] Running Detection Module..."
python src/football_ai/cli.py --mode detect --input $VIDEO_PATH --output data/outputs/test30s/detection --device cuda --max-frames $MAX_FRAMES

# 2. Tracking
Write-Host "`n[2/4] Running Tracking Module..."
python src/football_ai/cli.py --mode track --input $VIDEO_PATH --output data/outputs/test30s/tracking --device cuda --max-frames $MAX_FRAMES

# 3. Classification
Write-Host "`n[3/4] Running Classification Module..."
python src/football_ai/cli.py --mode classify --input $VIDEO_PATH --output data/outputs/test30s/classification --device cuda --max-frames $MAX_FRAMES

# 4. Analytics & Heatmaps
Write-Host "`n[4/4] Running Analytics Module (Generates Heatmaps)..."
python src/football_ai/cli.py --mode analytics --input $VIDEO_PATH --output data/outputs/test30s --device cuda --max-frames $MAX_FRAMES

Write-Host "`n========================================="
Write-Host " TEST COMPLETED! CHECK OUTPUT FOLDER"
Write-Host "========================================="
