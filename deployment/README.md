# Edge Deployment & Model Compression

Code and documentation for optimizing and deploying the mSigLIP model on edge devices (Qualcomm RB3 Gen2).

## Structure

```
deployment/
├── scripts/                       # mSigLIP deployment pipeline
│   ├── analyze_checkpoint.py      # Shared: Analyze checkpoint (size, RAM, compatibility)
│   ├── inference_test.py          # Shared: Test inference on target device
│   ├── lora_fp16/                 # Step 1: LoRA merge + FP16 export
│   │   └── export.py              #   Merge LoRA → FP16/FP32 state dict
│   └── onnx/                      # Step 2: ONNX conversion
│       └── export.py              #   FP16/FP32 state dict → ONNX
│
├── hardware_profiling/            # RB3 hardware capability testing (proxy models)
│   ├── benchmark.py               # PyTorch CPU vs ONNX Runtime (MobileNetV2, ResNet18)
│   ├── snpe_benchmark.py          # Qualcomm SNPE benchmark (requires DLC models)
│   ├── collect_sysinfo.sh         # Collect system information
│   ├── install_deps.sh            # Install dependencies on device
│   └── run_all.sh                 # Run full hardware profiling suite
│
├── docs/                          # Documentation
│   ├── system.md                  # RB3 Gen2 hardware specifications
│   ├── experiment.md              # Benchmark step-by-step guide
│   └── benchmark-rp.md            # Hardware benchmark results
│
├── logs/                          # Auto-generated logs (timestamped)
├── deploy_utils.py                # Shared utilities (TeeLogger)
└── README.md
```

All scripts in `scripts/` and `hardware_profiling/` auto-log terminal output to `logs/`.

## Deployment Pipeline

```
Training (GPU server)       →  Step 1: FP16 export        →  Step 2: ONNX         →  Deploy (RB3 Gen2)
━━━━━━━━━━━━━━━━━━━            ━━━━━━━━━━━━━━━━━━━           ━━━━━━━━━━━━━         ━━━━━━━━━━━━━━━━━━━
trainer.py                     lora_fp16/export.py            onnx/export.py         inference_test.py
epoch=53.ckpt (1.4 GB)  →     model_fp16.pt (~740 MB)  →    *.onnx            →   ONNX Runtime inference
                               - Strip optimizer states
                               - Merge LoRA into base
                               - FP32 → FP16
```

## Target Device

| Component | Specification |
|-----------|--------------|
| **SoC** | Qualcomm QCS6490 |
| **CPU** | 4x Cortex-A78 @ 2.7GHz + 4x Cortex-A55 @ 1.9GHz |
| **GPU** | Adreno 643 |
| **DSP** | Hexagon 770 |
| **RAM** | 5.2 GB (available ~4 GB) |

## Usage

All scripts auto-log to `deployment/logs/` with timestamps.

### 1. Analyze checkpoint
```bash
python deployment/scripts/analyze_checkpoint.py --ckpt path/to/checkpoint.ckpt
```

### 2. Export to FP16 (merge LoRA + strip optimizer)
```bash
python deployment/scripts/lora_fp16/export.py \
    --ckpt epoch=56-val_score=52.28.ckpt \
    --output-dir exported_model
```

### 3. Convert to ONNX
```bash
python deployment/scripts/onnx/export.py \
    --model-dir exported_model \
    --precision fp32              # fp32 recommended for ONNX stability
```

### 4. Test inference
```bash
python deployment/scripts/inference_test.py \
    --model-dir exported_model \
    --dtype fp16 \
    --dataset-root /path/to/VN3K
```

### 5. Hardware profiling on RB3 (proxy models)
```bash
# SSH to RB3, copy hardware_profiling/ to device
cd ~/sigm
./run_all.sh
```
