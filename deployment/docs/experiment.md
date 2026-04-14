# Deep Learning Benchmarking Experiment
> **Qualcomm RB3 Gen2 - Step-by-Step Guide**

---

## 1. Objective
Benchmark popular deep learning models on the Qualcomm RB3 Gen2 platform comparing:
* PyTorch CPU (with ARM NEON optimization)
* ONNX Runtime (optimized inference engine)
* Qualcomm SNPE (Hexagon DSP acceleration requires SDK)

---

## 2. Hardware Specification

| Component | Specification |
| :--- | :--- |
| **SoC** | Qualcomm QCS6490 |
| **CPU** | Kryo 670 (4x Cortex-A78 @ 2.7GHz + 4x Cortex-A55 @ 1.9GHz) |
| **GPU** | Adreno 643 |
| **DSP** | Hexagon 770 |
| **NPU** | Qualcomm AI Engine |
| **RAM** | 5.2 GB |
| **OS** | Ubuntu 24.04.3 LTS (aarch64) |

---

## 3. Models Benchmarked

| Model | Parameters | Use Case |
| :--- | :--- | :--- |
| **MobileNetV2** | 3.4M | Lightweight mobile classification |
| **ResNet18** | 11.7M | Standard image classification |
| **EfficientNet-B0** | 5.3M | Efficient accuracy/compute tradeoff |

---

## 4. Step-by-Step Experiment Procedure

### Step 1: Access the Device
```bash
# Option A: Via Cockpit Web Console
# Open browser: https://qcrb3g2-ansalab.j12tee.qzz.io/system/terminal

# Option B: Via SSH
ssh ubuntu@<device-ip>
```

### Step 2: Navigate to Experiment Directory
```bash
cd ~/sigm
```

### Step 3: Environment Setup (First Time Only)
```bash
# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
./02_install_deps.sh
```

### Step 4: Activate Environment
```bash
source venv/bin/activate
```

### Step 5: Collect System Information
```bash
./01_collect_sysinfo.sh
# Output: system_info.md
```

### Step 6: Run Benchmarks
```bash
python 03_benchmark.py
# Outputs:
# benchmark_results.json (raw data)
# benchmark_report.md (formatted report)
```

### Step 7: Review Results
```bash
# View formatted report
cat benchmark_report.md

# View raw JSON data
cat benchmark_results.json
```

### Step 8: Run All at Once (Alternative)
```bash
./run_all.sh
```

---

## 5. Benchmark Methodology

### 5.1 Measurement Protocol
1. **Warmup Phase:** 5 inference runs (results discarded)
2. **Measurement Phase:** 50 inference runs
3. **Statistics Collected:**
   * Average latency (ms)
   * Standard deviation (ms)
   * Min/Max latency (ms)
   * Throughput (images/second)

### 5.2 Input Specification
* **Size:** $224 \times 224 \times 3$ (ImageNet standard)
* **Batch Sizes:** 1, 4, 8
* **Data Type:** FP32

### 5.3 CPU Configuration
```python
# PyTorch
torch.set_num_threads(8)

# ONNX Runtime
sess_options.intra_op_num_threads = 8
```

---

## 6. Qualcomm SDK Integration

### 6.1 Current Status on Device
* **SNPE runtime library:** Found (`/usr/lib/libhta_hexagon_runtime_snpe.so`)
* **SNPE CLI tools:** Not installed
* **Hexagon DSP:** Not accessible (requires SDK)

### 6.2 Installing Qualcomm AI Engine Direct SDK
**1. Download SDK from Qualcomm Developer Network:** `https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk`

**2. Extract and configure:**
```bash
tar -xzf snpe-*.tar.gz -C /opt/
export SNPE_ROOT=/opt/snpe-2.x.x
export PATH=$SNPE_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$SNPE_ROOT/lib/aarch64-ubuntu:$LD_LIBRARY_PATH
```

**3. Add to `~/.bashrc` for persistence:**
```bash
echo 'export SNPE_ROOT=/opt/snpe-2.x.x' >> ~/.bashrc
echo 'export PATH=$SNPE_ROOT/bin:$PATH' >> ~/.bashrc
```

### 6.3 Model Conversion (ONNX to DLC)
```bash
# Convert ONNX model to SNPE DLC format
snpe-onnx-to-dlc \
  --input_network ~/sigm/onnx_models/mobilenetv2_100.onnx \
  --output_path ~/sigm/dlc_models/mobilenetv2.dlc

# Verify conversion
snpe-dlc-info -input_dlc ~/sigm/dlc_models/mobilenetv2.dlc
```

### 6.4 Quantization (INT8)
```bash
# Create calibration data list
ls ~/sigm/calibration_images/*.raw > calibration_list.txt

# Quantize model
snpe-dlc-quantize \
  --input_dlc mobilenetv2.dlc \
  --input_list calibration_list.txt \
  --output_dlc mobilenetv2_int8.dlc \
  --enable_htp
```

### 6.5 Running on Hexagon DSP
```bash
# Create input list
echo "input.raw" > input_list.txt

# Run inference on DSP
snpe-net-run \
  --container mobilenetv2.dlc \
  --input_list input_list.txt \
  --use_dsp \
  --perf_profile high_performance

# Run inference on HTP (Hexagon Tensor Processor)
snpe-net-run \
  --container mobilenetv2_int8.dlc \
  --input_list input_list.txt \
  --use_htp
```

---

## 7. Expected Performance Comparison

### 7.1 Latency (Batch Size 1)
| Model | PyTorch CPU | ONNX Runtime | SNPE CPU | SNPE DSP | SNPE HTP (INT8) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MobileNetV2** | 90 ms | ~70 ms | ~50 ms | ~5 ms | ~2 ms |
| **ResNet18** | 98 ms | ~75 ms | ~60 ms | ~10 ms | ~4 ms |
| **EfficientNet-B0** | 126 ms | ~95 ms | ~70 ms | ~7 ms | ~3 ms |

### 7.2 Throughput (images/second)
| Runtime | MobileNetV2 | ResNet18 | EfficientNet-B0 |
| :--- | :--- | :--- | :--- |
| **PyTorch CPU** | 11 | 10 | 8 |
| **SNPE DSP** | ~200 | ~100 | ~143 |
| **SNPE HTP (INT8)** | ~500 | ~250 | ~333 |

---

## 8. Files Reference

### 8.1 Scripts (`~/sigm/`)
| File | Description | Usage |
| :--- | :--- | :--- |
| `01_collect_sysinfo.sh` | Collect system info | `./01_collect_sysinfo.sh` |
| `02_install_deps.sh` | Install Python deps | `./02_install_deps.sh` |
| `03_benchmark.py` | Run benchmarks | `python 03_benchmark.py` |
| `run_all.sh` | Run everything | `./run_all.sh` |

### 8.2 Output Files
| File | Description |
| :--- | :--- |
| `system_info.md` | Hardware/software documentation |
| `benchmark_results.json` | Raw benchmark data (JSON) |
| `benchmark_report.md` | Formatted report (Markdown) |
| `installed_packages.txt` | Python packages list |
| `benchmark_log.txt` | Execution log |

### 8.3 Directories
| Directory | Description |
| :--- | :--- |
| `venv/` | Python virtual environment |
| `onnx_models/` | Exported ONNX models |
| `dlc_models/` | SNPE DLC models (after conversion) |

---

## 9. Troubleshooting

### 9.1 Memory Issues
```bash
# Reduce batch size
python -c "from 03_benchmark import benchmark_pytorch"
# Modify BATCH_SIZES in script to [1, 2]

# Monitor memory
watch -n 1 free -h
```

### 9.2 Missing Dependencies
```bash
source ~/sigm/venv/bin/activate
pip install <missing-package>
```

### 9.3 ONNX Export Errors
```bash
# Install onnxscript
pip install onnxscript onnx
```
```python
# Use explicit opset version
torch.onnx.export(model, dummy, path, opset_version=17)
```

### 9.4 SNPE Errors
```bash
# Check DSP availability
ls -la /dev/adsprpc*

# Check SNPE libs
ldconfig -p | grep snpe

# Verify SNPE installation
snpe-net-run --version
```

### 9.5 Permission Issues
```bash
# Run with admin access (via Cockpit).
# Click "Turn on administrative access" in web console
```

---

## 10. Additional Experiments

### 10.1 Object Detection (YOLOv8)
```bash
pip install ultralytics

python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx')
"
```

### 10.2 Semantic Segmentation (DeepLabV3)
```bash
python -c "
import torch
model = torch.hub.load('pytorch/vision', 'deeplabv3_mobilenet_v3_large', pretrained=False)
"
```

### 10.3 NLP (DistilBERT)
```bash
pip install transformers

python -c "
from transformers import DistilBertModel
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
"
```

---

## 11. References
* Qualcomm RB3 Gen2 Documentation
* Qualcomm AI Engine Direct SDK
* SNPE User Guide
* PyTorch ARM Optimization
* ONNX Runtime ARM64
* timm Model Zoo

---
*Experiment Guide for Qualcomm RB3 Gen2 Deep Learning Benchmarking | Created: 2026-01-25*