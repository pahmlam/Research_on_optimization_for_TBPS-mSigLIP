import os
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader
import textwrap

if not OmegaConf.has_resolver("tuple"):
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)

# Import modules
try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    print("Please install safetensors: pip install safetensors")
    sys.exit(1)

from lightning_models import LitTBPS
from lightning_data import TBPSDataModule
from data.bases import ImageDataset, TextDataset

# --- CONFIGURATION ---
TARGET_PIDS = [np.int64(2000), np.int64(2003), np.int64(2006), np.int64(2006), 
               np.int64(2015), np.int64(2024), np.int64(2024), np.int64(2024), 
               np.int64(2030), np.int64(2033)]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

# --- MODEL LOADING FUNCTIONS ---
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

def load_model(config_path, ckpt_path, enable_lora):
    cfg = OmegaConf.load(config_path)
    
    if not enable_lora and "lora" in cfg: 
        del cfg["lora"]
    
    dm = TBPSDataModule(cfg)
    dm.setup(stage='test')
    
    model = LitTBPS(cfg, num_iters_per_epoch=1)
    if enable_lora: 
        model.setup_lora(cfg.lora)
        
    print(f"   Loading weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt['state_dict'].items()}
    aggressive_load(model, state_dict)
    
    model.to(DEVICE).eval()
    return model, dm

# --- IMAGE PROCESSING ---
def denormalize(tensor, target_size=(256, 128)):
    """Chuyển tensor về ảnh numpy, có resize."""
    if target_size is not None:
        tensor = F.interpolate(tensor.unsqueeze(0), size=target_size, mode='bicubic', align_corners=False).squeeze(0)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    
    img = tensor * std + mean
    img = img.clamp(0, 1)
    
    return img.permute(1, 2, 0).cpu().numpy()

def get_rank_results(model, dm, target_pids):
    # 1. Extract Gallery
    gal_feats, gal_pids, gal_imgs = [], [], []
    
    loader = DataLoader(ImageDataset(dm.dataset.test, is_train=False, 
                                     image_size=dm.config.aug.img.size, 
                                     mean=dm.config.aug.img.mean, std=dm.config.aug.img.std),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print("   Extracting Gallery...")
    with torch.no_grad():
        for batch in tqdm(loader):
            imgs = batch['images'].to(DEVICE)
            f = F.normalize(model.get_image_features(imgs), dim=1)
            gal_feats.append(f.cpu())
            gal_pids.append(batch['pids'])
            gal_imgs.append(batch['images'].cpu())
            
    gal_feats = torch.cat(gal_feats)
    gal_pids = torch.cat(gal_pids)
    gal_imgs = torch.cat(gal_imgs)

    # 2. Extract Queries
    results = {}
    txt_loader = DataLoader(TextDataset(dm.dataset.test, tokenizer=dm.tokenizer),
                            batch_size=BATCH_SIZE, shuffle=False)
    
    print("   Scanning Queries...")
    with torch.no_grad():
        for batch in txt_loader:
            pids = batch['pids']
            mask = torch.isin(pids, torch.tensor(target_pids, device=pids.device))
            if not mask.any(): continue
            
            inputs = {}
            for k, v in batch.items():
                if k in ['input_ids', 'attention_mask', 'caption_input_ids', 'caption_attention_mask']:
                    inputs[k] = v.to(DEVICE)
            
            if 'caption_input_ids' in inputs:
                inputs['input_ids'] = inputs.pop('caption_input_ids')
                inputs['attention_mask'] = inputs.pop('caption_attention_mask')

            all_feats = F.normalize(model.get_text_features(inputs), dim=1).cpu()
            indices = torch.where(mask)[0]
            for idx in indices:
                pid = pids[idx].item()
                if pid in results: continue 
                
                feat = all_feats[idx].unsqueeze(0)
                sims = torch.matmul(feat, gal_feats.t()).squeeze()
                topk_vals, topk_idx = torch.topk(sims, k=3)
                
                top_imgs = [gal_imgs[i] for i in topk_idx]
                top_pids = [gal_pids[i].item() for i in topk_idx]
                
 
                gt_idx = (gal_pids == pid).nonzero(as_tuple=True)[0]
                if len(gt_idx) > 0:
                    gt_img = gal_imgs[gt_idx[0]] 
                else:
                    gt_img = None 

                if 'caption_input_ids' in batch: raw_ids = batch['caption_input_ids'][idx]
                else: raw_ids = batch['input_ids'][idx]
                    
                caption = dm.tokenizer.decode(raw_ids, skip_special_tokens=True)
                
                results[pid] = {
                    'caption': caption,
                    'top_imgs': top_imgs,
                    'top_pids': top_pids,
                    'gt_img': gt_img, # Lưu ảnh GT
                    'is_correct': [p == pid for p in top_pids]
                }
    return results

def plot_qualitative(base_res, ours_res, pids):
    print("🎨 Drawing Flipped Cases...")
    
    rows = len(pids)
    cols = 8 
    
    fig, axes = plt.subplots(rows, cols, figsize=(22, 6 * rows))
    if rows == 1: axes = axes.reshape(1, -1)
    
    for r, pid in enumerate(pids):
        if pid not in base_res or pid not in ours_res:
            print(f"⚠️ PID {pid} missing. Skipping.")
            continue
            
        b_data = base_res[pid]
        o_data = ours_res[pid]
        
        # 1. Text Query (Cột 0)
        ax_txt = axes[r, 0]
        real_pid = pid + 1 
        wrapped_text = "\n".join(textwrap.wrap(f"Query ID: {real_pid}\n(Model ID: {pid})\n\n{b_data['caption']}", width=25))
        ax_txt.text(0.5, 0.5, wrapped_text, ha='center', va='center', fontsize=14, family='serif')
        ax_txt.axis('off')
        
        # 2. Baseline Images (Cột 1-3)
        for i in range(3):
            ax = axes[r, 1+i]
            img = denormalize(b_data['top_imgs'][i], target_size=(256, 128))
            ax.imshow(img, interpolation='bicubic') 
            
            is_correct = b_data['is_correct'][i]
            color = 'green' if is_correct else 'red'
            width = 4 if is_correct else 2
            
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(width)
                
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0 and i == 1: 
                ax.set_title("Baseline (mSigLIP)", fontsize=16, fontweight='bold', pad=15, family='serif')
            
            ax.text(5, 25, f"Rank-{i+1}", color='white', fontweight='bold', fontsize=12,
                    bbox=dict(facecolor=color, alpha=0.8, pad=2, edgecolor='none'))

        # 3. Ours Images (Cột 4-6)
        for i in range(3):
            ax = axes[r, 4+i]
            img = denormalize(o_data['top_imgs'][i], target_size=(256, 128))
            ax.imshow(img, interpolation='bicubic')
            
            is_correct = o_data['is_correct'][i]
            color = 'green' if is_correct else 'red'
            width = 4 if is_correct else 2
            
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(width)
                
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0 and i == 1: 
                ax.set_title("Ours (LoRA + Circle)", fontsize=16, fontweight='bold', pad=15, family ='serif')
            
            ax.text(5, 25, f"Rank-{i+1}", color='white', fontweight='bold', fontsize=12,
                    bbox=dict(facecolor=color, alpha=0.8, pad=2, edgecolor='none'))

        ax_gt = axes[r, 7]
        gt_tensor = o_data['gt_img'] 
        
        if gt_tensor is not None:
            img_gt = denormalize(gt_tensor, target_size=(256, 128))
            ax_gt.imshow(img_gt, interpolation='bicubic')
            
            for spine in ax_gt.spines.values():
                spine.set_edgecolor('blue')
                spine.set_linewidth(3)
        else:
            ax_gt.text(0.5, 0.5, "GT Missing", ha='center', va='center')
        
        ax_gt.set_xticks([]); ax_gt.set_yticks([])
        if r == 0:
            ax_gt.set_title("Ground Truth\n(Reference)", fontsize=16, fontweight='bold', pad=15, family ='serif')

    plt.tight_layout()
    plt.savefig("flipped_cases_visualization.png", dpi=300, bbox_inches='tight')
    print("✅ Saved to flipped_cases_visualization.png")

if __name__ == "__main__":
    CONFIG_PATH = "/mnt/data/user_data/lampt/PS/code/outputs/2026-01-16/10-20-32/.hydra/config.yaml"
    CKPT_BASELINE = "/mnt/data/user_data/lampt/PS/code/epoch=56-val_score=49.15.ckpt" 
    CKPT_OURS = "/mnt/data/user_data/lampt/PS/code/checkpoints/vn3k-curri/epoch=53-val_score=51.30.ckpt"

    print(">>> Processing Baseline...")
    m_base, dm = load_model(CONFIG_PATH, CKPT_BASELINE, enable_lora=False)
    res_base = get_rank_results(m_base, dm, TARGET_PIDS)
    del m_base; torch.cuda.empty_cache()

    print("\n>>> Processing Ours...")
    m_ours, _ = load_model(CONFIG_PATH, CKPT_OURS, enable_lora=True)
    res_ours = get_rank_results(m_ours, dm, TARGET_PIDS)
    
    plot_qualitative(res_base, res_ours, TARGET_PIDS)