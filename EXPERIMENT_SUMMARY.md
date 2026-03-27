# Experiment Summary: mSigLIP + LoRA + Curriculum Circle Loss

## Model

| Setting | Value |
|---|---|
| Backbone | mSigLIP (`siglip-base-patch16-256-multilingual`) |
| Embedding dim | 768 |
| Image size | 384 x 128 |
| Patch size | 16 |
| Similarity function | Sigmoid (SigLIP-style) |
| Tokenizer | SiglipTokenizer (vocab: 250,000, max length: 64) |

## LoRA Configuration

| Setting | Value |
|---|---|
| Rank (r) | 32 |
| Alpha | 64 (scaling = alpha/r = 2.0) |
| Target modules | `q_proj`, `v_proj`, `k_proj`, `out_proj` |
| Dropout | 0.05 |
| Bias | none |
| Task type | FEATURE_EXTRACTION |

## Training

| Setting | Value |
|---|---|
| Max epochs | 60 |
| Precision | 16-mixed |
| Batch size | 64 (VN3K) / 32 (CUHK-PEDES) |
| Accumulate grad batches | 3 (effective batch = 192 for VN3K) |
| Seed | 2307 |
| Dataset | VN3K_VI (Vietnamese) / CUHK-PEDES (English) |
| Sampler | identity (VN3K) / random (CUHK-PEDES) |

## Optimizer

| Setting | Value |
|---|---|
| Type | AdamW |
| Learning rate | 1e-4 |
| Betas | [0.9, 0.999] |
| LoRA weight decay | 0.0 |
| Other weight decay | 0.01 |
| SimCLR/Cross weight decay | 0.05 |

## LR Scheduler

| Setting | Value |
|---|---|
| Type | Cosine with warmup |
| Warmup epochs | 20% of total (12 epochs) |
| Warmup method | Linear |
| Start LR | 1e-6 |
| End LR | 1e-5 |

## Loss Functions

### Total loss = NITC + Circle (curriculum) + CITC + SimCLR

| Loss | Weight | Details |
|---|---|---|
| **N-ITC** | 1.0 | Noise-robust contrastive loss (with MVS augmentation) |
| **Cross-Modal Circle Loss** | 0.0 -> 0.1 (curriculum) | margin=0.35, gamma=128 |
| **C-ITC** | 0.1 | Cyclic contrastive (inmodal=0.25, intermodal=0.25) |
| **SimCLR** | 0.4 | Self-supervised (temperature=0.1) |

### Curriculum Schedule (Circle Loss)

| Epoch | Circle Loss Weight |
|---|---|
| 0-5 | 0.0 (warmup, circle off) |
| 6-20 | Linear ramp: 0.0 -> 0.1 |
| 21-60 | 0.1 (stable) |

## Image Augmentation

- PILResize (384x128, bicubic)
- ColorJitter (brightness=0.1, contrast=0.1, saturation=0.1)
- RandomRotation (15 deg)
- RandomResizedCrop (scale 0.9-1.0)
- RandomGrayscale (p=0.1)
- RandomHorizontalFlip (p=0.5)
- RandomErasing (scale 0.10-0.20)
- Normalize (mean=0.5, std=0.5)
- MVS: 2 random augmented views per image

## Text Augmentation

- Random deletion (p=0.05)

## Results (VN3K - 3000VnPersonSearch)

| Method | R@1 | R@5 | R@10 | mAP | mINP |
|---|---|---|---|---|---|
| TBPS-mSigLIP (Full FT) | 49.70 | 75.93 | 84.75 | 54.96 | 48.66 |
| Ours (LoRA Only) | 49.90 | 78.05 | 86.30 | 55.83 | 49.45 |
| Ours (LoRA + Circle Fixed) | 50.53 | 77.78 | 86.43 | 55.94 | 49.37 |
| **Ours (LoRA + Curriculum)** | **51.30** | **78.20** | **86.68** | **56.46** | **49.89** |

### Key Takeaways

- **+1.60 R@1** over full fine-tuning baseline with significantly fewer trainable parameters
- Curriculum scheduling outperforms fixed circle loss weight (+0.77 R@1)
- LoRA alone already matches baseline while using ~3-5% of trainable parameters
- Circle Loss with curriculum adds fine-grained hard-negative discrimination without disrupting early global alignment


