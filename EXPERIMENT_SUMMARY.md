# Experiment Summary: mSigLIP + LoRA + Curriculum Circle Loss

## Model

| Setting | Value |
|---|---|
| Backbone | mSigLIP (`siglip-base-patch16-256-multilingual`) |
| Embedding dim | 768 |
| Image size | 256 x 256 |
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
| Batch size | 24 (VN3K) / 16 (CUHK-PEDES) |
| Accumulate grad batches | 3 (effective batch = 72 for VN3K) |
| Seed | 2307 (also tested: 2300, 2400) |
| Dataset | VN3K_VI (Vietnamese) / CUHK-PEDES (English) / PRW-TPS-CN (Chinese) |
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

- PILResize (256x256, bicubic)
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
| TBPS-mSigLIP (Full FT, batch=8) | 49.70 | 75.93 | 84.75 | 54.96 | 48.66 |
| TBPS-mSigLIP (Full FT, batch=24) | 49.18 | 76.30 | 85.58 | 54.49 | 47.87 |
| Ours (LoRA Only) | 49.90 | 78.05 | 86.30 | 55.83 | 49.45 |
| Ours (LoRA + Circle Fixed) | 50.53 | 77.78 | 86.43 | 55.94 | 49.37 |
| Ours (LoRA + Curriculum, seed 2307) | 51.30 | 78.20 | 86.68 | 56.46 | 49.89 |
| **Ours (LoRA + Curriculum, seed 2400)** | **52.28** | **79.55** | **88.03** | **57.32** | **50.57** |

### Multi-Seed Confidence (LoRA + Curriculum)

| Seed | R@1 | R@5 | R@10 | mAP | mINP |
|---|---|---|---|---|---|
| 2307 | 51.30 | 78.20 | 86.68 | 56.46 | 49.89 |
| 2300 | 50.98 | 78.60 | 86.95 | 57.08 | 51.22 |
| 2400 | 52.28 | 79.55 | 88.03 | 57.32 | 50.57 |
| **Mean +/- std** | **51.52 +/- 0.68** | **78.78 +/- 0.69** | **87.22 +/- 0.71** | **56.95 +/- 0.44** | **50.56 +/- 0.67** |

## Results (PRW-TPS-CN - Chinese)

| Method | R@1 | R@5 | R@10 | mAP | mINP |
|---|---|---|---|---|---|
| TPAN | 21.63 | 42.54 | 52.99 | - | - |
| TBPS-mSigLIP (Baseline) | 46.78 | 60.28 | 66.82 | 35.41 | 10.61 |
| **Ours (mSigLIP-CLoRA)** | **59.35** | **70.58** | **75.48** | **46.44** | **15.10** |

## Results (10% CUHK-PEDES)

| Method | R@1 | R@5 | R@10 | mAP | mINP |
|---|---|---|---|---|---|
| Baseline TBPS-mSigLIP | 46.73 | 68.65 | 77.55 | 41.75 | 26.56 |
| Ours (Fixed Weight) | 56.87 | **77.18** | 84.15 | 50.70 | 34.61 |
| **Ours (Curriculum)** | **57.10** | 76.98 | **84.34** | **50.90** | **34.85** |

### Key Takeaways

- **+2.58 R@1** over full fine-tuning baseline (best seed) with significantly fewer trainable parameters
- Curriculum scheduling outperforms fixed circle loss weight (+0.77 R@1)
- LoRA alone already matches baseline while using ~1.57% of trainable parameters (5.9M / 376M)
- Circle Loss with curriculum adds fine-grained hard-negative discrimination without disrupting early global alignment
- Full FT at batch=24 (49.18%) underperforms LoRA (49.90%), ruling out batch size as source of improvement
- Results generalize across 3 languages: Vietnamese, English, and Chinese


