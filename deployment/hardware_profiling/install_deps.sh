#!/bin/bash
# Install dependencies for benchmarking

echo "=== Installing Dependencies ==="
echo "Started: $(date)"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing PyTorch (ARM64 CPU)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "Installing ONNX Runtime..."
pip install onnxruntime

echo "Installing benchmark utilities..."
pip install numpy pillow pandas tabulate psutil

echo "Installing model hub tools..."
pip install timm

echo "=== Dependencies Installed in ~/sigm/venv ==="
pip list > ~/sigm/installed_packages.txt
echo "Finished: $(date)"
