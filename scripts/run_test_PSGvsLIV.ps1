$env:PYTHONPATH = "src"

$VIDEO_PATH = "data/raw/test_PSGvsLIV.mp4"
$OUTPUT_DIR = "data/outputs/test_PSGvsLIV"

Write-Host "========================================="
Write-Host " RUNNING FULL ANALYTICS PIPELINE END-TO-END"
Write-Host " Input Video: $VIDEO_PATH"
Write-Host " Output Dir: $OUTPUT_DIR"
Write-Host "========================================="

# Mode 'analytics' handles Detection -> Tracking -> Classification -> Field Mapping
# and outputs the final composite video + Heatmaps without saving intermediate module JSONs.
python src/football_ai/cli.py --mode analytics --input $VIDEO_PATH --output $OUTPUT_DIR --device cuda

Write-Host "`n========================================="
Write-Host " TEST COMPLETED! CHECK OUTPUT FOLDER: $OUTPUT_DIR"
Write-Host " Heatmaps and final mapped video are saved here."
Write-Host "========================================="
