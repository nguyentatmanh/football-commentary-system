#!/bin/bash
set -e

echo "Creating virtual environment .venv..."
python3 -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing project dependencies..."
pip install --upgrade pip
pip install -e .

echo "Setup complete!"
echo "To activate environment, run: source .venv/bin/activate"
