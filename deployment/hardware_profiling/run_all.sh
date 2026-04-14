#!/bin/bash
# Hardware profiling benchmarks (proxy models: MobileNetV2, ResNet18, EfficientNet)
# NOT mSigLIP — this profiles RB3 Gen2 hardware capabilities
# All terminal output is automatically logged with timestamp
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../logs"

cd ~/sigm
source venv/bin/activate

# Setup logging — tee all stdout+stderr to timestamped log file
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/hardware_profiling_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "Qualcomm RB3 Gen2 Hardware Profiling"
echo "(Proxy models — NOT mSigLIP)"
echo "=========================================="
echo "Started: $(date)"
echo "Log file: $LOG_FILE"
echo ""

# Step 1: Collect system info
echo "[1/3] Collecting system information..."
./01_collect_sysinfo.sh

# Step 2: Run benchmarks
echo "[2/3] Running proxy model benchmarks..."
python 03_benchmark.py

# Step 3: Summary
echo "[3/3] Generating summary..."
echo ""
echo "=========================================="
echo "Files generated:"
ls -la *.md *.json *.txt 2>/dev/null
echo "=========================================="
echo "Completed: $(date)"
echo "Log saved: $LOG_FILE"
