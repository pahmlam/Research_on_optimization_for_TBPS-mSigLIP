# Edge Deployment & Model Compression

Phần này chứa toàn bộ code và tài liệu liên quan đến việc tối ưu hóa và triển khai mô hình mSigLIP lên thiết bị edge (Qualcomm RB3 Gen2).

## Cấu trúc

```
deployment/
├── docs/                          # Tài liệu triển khai
│   ├── system.md                  # Thông số phần cứng RB3 Gen2
│   ├── experiment.md              # Hướng dẫn benchmark step-by-step
│   └── benchmark-rp.md            # Kết quả benchmark (PyTorch, ONNX, SNPE)
│
├── scripts/                       # Scripts export & inference
│   ├── analyze_checkpoint.py      # Phân tích checkpoint (size, RAM, compatibility)
│   ├── export_inference.py        # Export model: merge LoRA → FP16/ONNX
│   └── inference_test.py          # Test inference trên thiết bị đích
│
└── benchmark/                     # Scripts benchmark trên RB3
    ├── collect_sysinfo.sh         # Thu thập thông tin hệ thống
    ├── install_deps.sh            # Cài đặt dependencies
    ├── benchmark.py               # Benchmark PyTorch CPU vs ONNX Runtime
    ├── snpe_benchmark.py          # Benchmark Qualcomm SNPE (cần DLC models)
    └── run_all.sh                 # Chạy toàn bộ benchmark suite
```

## Pipeline triển khai

```
Training (GPU server)           →  Export (dev machine)        →  Deploy (RB3 Gen2)
━━━━━━━━━━━━━━━━━━━             ━━━━━━━━━━━━━━━━━━━           ━━━━━━━━━━━━━━━━━━━
trainer.py                         export_inference.py            inference_test.py
epoch=53.ckpt (1.4 GB)      →     model_fp16.pt (~740 MB)   →   torch.load() + inference
                                   - Bỏ optimizer states
                                   - Merge LoRA vào base
                                   - FP32 → FP16
```

## Thiết bị đích

| Component | Specification |
|-----------|--------------|
| **SoC** | Qualcomm QCS6490 |
| **CPU** | 4x Cortex-A78 @ 2.7GHz + 4x Cortex-A55 @ 1.9GHz |
| **GPU** | Adreno 643 |
| **DSP** | Hexagon 770 |
| **RAM** | 5.2 GB (available ~4 GB) |

## Sử dụng

### 1. Phân tích checkpoint
```bash
python deployment/scripts/analyze_checkpoint.py --ckpt path/to/checkpoint.ckpt
```

### 2. Export model
```bash
python deployment/scripts/export_inference.py \
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

### 4. Benchmark trên RB3
```bash
# SSH vào RB3, copy benchmark/ sang device
cd ~/sigm
./run_all.sh
```
