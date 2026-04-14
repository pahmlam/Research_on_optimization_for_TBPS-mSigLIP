#!/bin/bash
# Master script to run all benchmarks
cd ~/sigm
source venv/bin/activate

echo "=========================================="
echo "Qualcomm RB3 Gen2 DL Benchmark Suite"
echo "=========================================="
echo "Started: $(date)"
echo ""

# Step 1: Collect system info
echo "[1/3] Collecting system information..."
./01_collect_sysinfo.sh

# Step 2: Run benchmarks
echo "[2/3] Running benchmarks..."
python 03_benchmark.py

# Step 3: Summary
echo "[3/3] Generating summary..."
echo ""
echo "=========================================="
echo "Files generated in ~/sigm:"
ls -la *.md *.json *.txt 2>/dev/null
echo "=========================================="
echo "Completed: $(date)"
