# Edge Deployment & Model Compression

Code and documentation for optimizing and deploying the mSigLIP model on edge devices (Qualcomm RB3 Gen2).

## Structure

```
deployment/
├── scripts/                       # mSigLIP deployment pipeline
│   ├── analyze_checkpoint.py      # Shared: Analyze checkpoint (size, RAM, compatibility)
│   ├── inference_test.py          # Shared: Test inference on target device
│   └── lora_fp16/                 # Approach: LoRA merge + FP16 + ONNX
│       └── export.py              #   Export model (merge LoRA → FP16/ONNX)
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
├── utils.py                       # Shared utilities (TeeLogger)
└── README.md
```

All scripts in `scripts/` and `hardware_profiling/` auto-log terminal output to `logs/`.

## Deployment Pipeline

```
Training (GPU server)           →  Export (dev machine)        →  Deploy (RB3 Gen2)
━━━━━━━━━━━━━━━━━━━             ━━━━━━━━━━━━━━━━━━━           ━━━━━━━━━━━━━━━━━━━
trainer.py                         export_inference.py            inference_test.py
epoch=53.ckpt (1.4 GB)      →     model_fp16.pt (~740 MB)   →   torch.load() + inference
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

### 2. Export model (LoRA merge + FP16 + ONNX)
```bash
python deployment/scripts/lora_fp16/export.py \
    --ckpt epoch=53-val_score=51.30.ckpt \
    --output-dir exported_model \
    --format both  # pytorch + onnx
```

### 3. Test inference
```bash
python deployment/scripts/inference_test.py \
    --model-dir exported_model \
    --dtype fp16 \
    --dataset-root /path/to/VN3K
```

### 4. Hardware profiling on RB3 (proxy models)
```bash
# SSH to RB3, copy hardware_profiling/ to device
cd ~/sigm
./run_all.sh
```
