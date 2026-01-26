import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader

if not OmegaConf.has_resolver("tuple"):
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))
if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    print("Please install safetensors")
    sys.exit(1)

from lightning_models import LitTBPS
from lightning_data import TBPSDataModule
from data.bases import ImageDataset, TextDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

def normalize_key(k):
    garbage = ["model.", "backbone.", "base_model.", "image_encoder.", "text_encoder.", 
               "vision_model.", "text_model.", "encoder.", "module."]
    k_clean = k
    for g in garbage: k_clean = k_clean.replace(g, "")
    return k_clean

def aggressive_load(model, state_dict):
    model_state = model.state_dict()
    new_sd = {}
    ckpt_map = {normalize_key(k): k for k in state_dict.keys()}
    for model_k in model_state.keys():
        clean = normalize_key(model_k)
        if clean in ckpt_map:
            ckpt_v = state_dict[ckpt_map[clean]]
            if model_state[model_k].shape == ckpt_v.shape:
                new_sd[model_k] = ckpt_v
    model.load_state_dict(new_sd, strict=False)
    return model

def load_model_and_extract(config_path, ckpt_path, enable_lora):
    print(f"🚀 Processing: {ckpt_path}")
    cfg = OmegaConf.load(config_path)
    if not enable_lora and "lora" in cfg: del cfg["lora"]
    
    dm = TBPSDataModule(cfg)
    dm.setup(stage='test')
    
    model = LitTBPS(cfg, dm.tokenizer.true_vocab_size, dm.tokenizer.pad_token_id, 1, 100, dm.num_classes)
    if enable_lora: model.setup_lora(cfg.lora)
        
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt['state_dict'].items()}
    aggressive_load(model, state_dict)
    model.to(DEVICE).eval()
    
    # 1. Extract Gallery
    loader = DataLoader(ImageDataset(dm.dataset.test, is_train=False, 
                                     image_size=dm.config.aug.img.size, 
                                     mean=dm.config.aug.img.mean, std=dm.config.aug.img.std),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    gal_feats, gal_pids = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Gallery"):
            imgs = batch['images'].to(DEVICE)
            f = F.normalize(model.get_image_features(imgs), dim=1)
            gal_feats.append(f.cpu())
            gal_pids.append(batch['pids'])
    gal_feats = torch.cat(gal_feats)
    gal_pids = torch.cat(gal_pids)

    # 2. Extract Query
    txt_loader = DataLoader(TextDataset(dm.dataset.test, tokenizer=dm.tokenizer),
                            batch_size=BATCH_SIZE, shuffle=False)
    q_feats, q_pids = [], []
    with torch.no_grad():
        for batch in tqdm(txt_loader, desc="Query"):
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'caption_input_ids', 'caption_attention_mask']}
            if 'caption_input_ids' in inputs:
                inputs['input_ids'] = inputs.pop('caption_input_ids')
                inputs['attention_mask'] = inputs.pop('caption_attention_mask')
            f = F.normalize(model.get_text_features(inputs), dim=1)
            q_feats.append(f.cpu())
            q_pids.append(batch['pids'])
    q_feats = torch.cat(q_feats)
    q_pids = torch.cat(q_pids)
    
    # 3. Compute Rank-1 Status
    print("   Computing Ranking...")
    sims = torch.matmul(q_feats.to(DEVICE), gal_feats.to(DEVICE).t())
    
    top1_scores, top1_indices = torch.max(sims, dim=1)
    top1_indices = top1_indices.cpu()
    
    # Check Correctness
    # top1_pids = gal_pids[top1_indices]
    correct_mask = (gal_pids[top1_indices] == q_pids)
    
    return correct_mask.numpy(), q_pids.numpy()

def main():
    CONFIG_PATH = "/mnt/data/user_data/lampt/PS/code/outputs/2026-01-16/10-20-32/.hydra/config.yaml"
    CKPT_BASELINE = "/mnt/data/user_data/lampt/PS/code/epoch=56-val_score=49.15.ckpt" 
    CKPT_OURS = "/mnt/data/user_data/lampt/PS/code/checkpoints/vn3k-curri/epoch=53-val_score=51.30.ckpt"
    
    # Lấy kết quả đúng/sai của từng model
    correct_base, pids_base = load_model_and_extract(CONFIG_PATH, CKPT_BASELINE, enable_lora=False)
    correct_ours, pids_ours = load_model_and_extract(CONFIG_PATH, CKPT_OURS, enable_lora=True)
    
    # Sanity check
    if not np.array_equal(pids_base, pids_ours):
        print("Fatal Error: PIDs do not match!")
        return

    flipped_indices = np.where((~correct_base) & (correct_ours))[0]
    
    flipped_pids = pids_base[flipped_indices]
    
    print("\n" + "="*60)
    print(f"FOUND {len(flipped_pids)} REAL FLIPPED CASES (Base: Wrong -> Ours: Right)")
    print("="*60)
    
    print("Danh sách ID chính xác để vẽ (Copy vào TARGET_PIDS):")
    print(list(flipped_pids[:10])) 
    print("\nChi tiết 10 ca đầu tiên:")
    for pid in flipped_pids[:10]:
        print(f"- Model ID: {pid} (JSON ID: {pid+1})")

if __name__ == "__main__":
    main()