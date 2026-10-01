#!/bin/bash
# Setup script for Turkish ASR project
set -e

echo "🚀 Setting up Virtual Environment (.venv) for Turkish ASR..."
python3 -m venv .venv

echo "🔄 Activating virtual environment..."
source .venv/bin/activate

echo "📦 Upgrading pip..."
pip install --upgrade pip

echo "📦 Installing core dependencies from requirements.txt..."
pip install -r requirements.txt

echo "✅ Environment setup completed successfully!"
echo "To activate manually: source .venv/bin/activate"
