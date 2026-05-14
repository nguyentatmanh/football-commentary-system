#!/bin/bash
echo "Starting Demo run using Default Config..."
python3 -m football_ai.cli \
  --config configs/default.yaml \
  --input data/raw/sample.mp4 \
  --output data/outputs/demo \
  --device cpu
