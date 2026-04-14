# Knowledge Base

> Tài liệu kiến thức tích lũy trong quá trình nghiên cứu và phát triển mSigLIP.
> Mỗi mục ghi lại: **định nghĩa** các khái niệm liên quan, **vì sao** cần làm, **làm gì**, **làm như thế nào**, và **suy nghĩ/cách tiếp cận** của Claude khi giải quyết vấn đề.

---

## Mục lục

1. [Export model trước khi deploy lên RB3](#1-export-model-trước-khi-deploy-lên-rb3)
2. [Cập nhật README — Ongoing Work](#2-cập-nhật-readme--ongoing-work)

---

## 1. Export model trước khi deploy lên RB3

> **Ngày:** 2026-04  
> **Liên quan:** `deployment/scripts/export_inference.py`, `deployment/docs/system.md`

### Định nghĩa

- **Lightning Checkpoint (.ckpt):** File lưu trạng thái đầy đủ của quá trình training — bao gồm model weights (state_dict), optimizer states (Adam momentum + variance), learning rate scheduler, epoch/step counter, và Hydra config. Mục đích: resume training.
- **LoRA (Low-Rank Adaptation):** Kỹ thuật fine-tuning hiệu quả, thêm 2 ma trận nhỏ A (d×r) và B (r×d) vào mỗi attention layer. Output = W·x + B·A·x. Chỉ train A, B (r=32 → ~1.5% tổng params).
- **LoRA Merge:** Cộng sẵn W_merged = W + B·A, loại bỏ adapter. Kết quả toán học giống hệt nhưng chỉ cần 1 matmul thay vì 2 matmul + 1 add mỗi layer.
- **FP16 (Half-precision):** Biểu diễn số thực 16-bit thay vì FP32 (32-bit). Giảm một nửa dung lượng và RAM, tốc độ nhanh hơn trên phần cứng hỗ trợ (ARM NEON có `fphp` flag).

### Vì sao (WHY)

Qualcomm RB3 Gen2 chỉ có ~4 GB RAM khả dụng. Lightning checkpoint 1.4 GB chứa nhiều dữ liệu không cần cho inference:
- Optimizer states: ~160 MB (Adam momentum + variance cho trainable params)
- Training metadata: epoch, lr scheduler state, Hydra config
- LoRA adapters chưa merge: tăng computation (192 phép tính thừa mỗi forward pass)

`torch.load()` đọc **toàn bộ file** vào RAM trước khi lọc → peak RAM ~3.5–3.8 GB, sát giới hạn RB3 → **nguy cơ OOM**.

### Làm gì (WHAT)

Pipeline export 4 bước:
1. Load full checkpoint trên máy dev (RAM thoải mái)
2. Bỏ optimizer states — chỉ giữ state_dict
3. Merge LoRA vào base model — W_merged = W + B·A
4. Chuyển FP32 → FP16

### Làm như thế nào (HOW)

```bash
python deployment/scripts/export_inference.py \
    --ckpt epoch=53-val_score=51.30.ckpt \
    --output-dir exported_model \
    --format both
```

**Kết quả:**

| | Checkpoint gốc | Export FP16 |
|---|---|---|
| File size | 1.4 GB | ~740 MB |
| Peak RAM khi load | ~2.8 GB | ~1.5 GB |
| RAM khi inference | ~2.4 GB | ~1.8 GB |
| Tốc độ / layer | 2 matmul + 1 add | 1 matmul |
| Dependencies | peft, lightning | chỉ torch |
| R@1 | 51.30% | 51.30% (giữ nguyên) |

### Suy nghĩ & cách tiếp cận

- **Tại sao merge LoRA thay vì giữ nguyên?** Trên server GPU, overhead LoRA không đáng kể. Nhưng trên ARM CPU (RB3), mỗi matmul thừa đều tốn thời gian. 24 layers × 4 projections = 96 lần compute thừa mỗi forward. Merge LoRA là "free optimization" — không mất accuracy, chỉ cần 1 dòng code (`merge_and_unload()`).
- **Tại sao FP16 mà không INT8?** FP16 là bước an toàn nhất — không cần calibration data, không mất accuracy. INT8 quantization cho tốc độ tốt hơn nhưng cần calibration set và có thể giảm accuracy. Nên làm FP16 trước, đánh giá, rồi mới thử INT8.
- **Tensor deduplication khi save FP16:** SigLIP share weights giữa vision/text encoder (`model.siglip.xxx` = `model.backbone.xxx`). Khi convert FP16, cần track `data_ptr()` để tránh lưu trùng → giảm file size thêm ~40%.

---

<!-- TEMPLATE CHO MỤC MỚI

## N. [Tiêu đề]

> **Ngày:** YYYY-MM  
> **Liên quan:** `file/path`, `another/path`

### Định nghĩa
- **Thuật ngữ 1:** Giải thích ngắn gọn
- **Thuật ngữ 2:** Giải thích ngắn gọn

### Vì sao (WHY)
Giải thích vấn đề / động lực.

### Làm gì (WHAT)
Mô tả giải pháp / hành động cụ thể.

### Làm như thế nào (HOW)
Code, commands, hoặc chi tiết kỹ thuật.

### Suy nghĩ & cách tiếp cận
Phân tích, trade-offs, lý do chọn cách này thay vì cách khác.

-->
