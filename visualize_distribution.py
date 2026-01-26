import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.bases import ImageDataset, TextDataset

# --- [MOD] Import module để chỉnh font ---
import matplotlib.font_manager as font_manager

from lightning_models import LitTBPS
from lightning_data import TBPSDataModule

# Đăng ký resolver
def resolve_tuple(*args): return tuple(args)
try:
    OmegaConf.register_new_resolver("tuple", resolve_tuple)
    OmegaConf.register_new_resolver("eval", eval)
except: pass

# --- [MOD] Cấu hình Matplotlib cho chuẩn bài báo ---
def set_publication_style():
    plt.rcParams.update({
        'font.size': 16,           # Kích thước font cơ bản
        'axes.labelsize': 20,      # Kích thước nhãn trục (sn, sp)
        'axes.titlesize': 22,      # Kích thước tiêu đề biểu đồ
        'xtick.labelsize': 16,     # Kích thước số trên trục x
        'ytick.labelsize': 16,     # Kích thước số trên trục y
        'legend.fontsize': 14,     # Kích thước chú thích
        'figure.titlesize': 24,    # Tiêu đề toàn cục
        'font.family': 'serif',    # Dùng font có chân (Serif)
        'font.serif': ['Times New Roman', 'DejaVu Serif'], # Ưu tiên Times New Roman
        'mathtext.fontset': 'stix', # Font công thức toán học đẹp hơn
        'lines.linewidth': 2.5,     # Đường nét đậm hơn
        'lines.markersize': 8,      # Marker to hơn
    })

# Giữ nguyên hàm extract_test_scores không đổi
def extract_test_scores(config_path, ckpt_path, device='cuda', enable_lora=None):
    # (Giữ nguyên nội dung hàm này như code cũ của bạn)
    print(f"Loading model from: {ckpt_path}")
    config = OmegaConf.load(config_path)
    
    if enable_lora is False and "lora" in config: del config["lora"]
    elif enable_lora is True and "lora" not in config: raise ValueError("Missing LoRA config")

    dm = TBPSDataModule(config)
    dm.setup(stage='test') 
    tokenizer = dm.tokenizer

    test_img_loader = DataLoader(
        ImageDataset(dm.dataset.test, is_train=False, image_size=config.aug.img.size, 
                     mean=config.aug.img.mean, std=config.aug.img.std),
        batch_size=64, shuffle=False, num_workers=4
    )
    test_txt_loader = DataLoader(
        TextDataset(dm.dataset.test, tokenizer=tokenizer),
        batch_size=64, shuffle=False, num_workers=4
    )

    model = LitTBPS(
        config, vocab_size=tokenizer.true_vocab_size, pad_token_id=tokenizer.pad_token_id,
        num_iters_per_epoch=100, train_set_length=100, num_classes=dm.num_classes
    )
    if config.get("lora", None):
        print("Setting up LoRA...")
        model.setup_lora(config.lora)

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    gallery_feats, gallery_pids = [], []
    print("Extracting Gallery...")
    with torch.no_grad():
        for batch in tqdm(test_img_loader):
            imgs = batch['images'].to(device)
            feats = model.get_image_features(imgs)
            gallery_feats.append(F.normalize(feats, p=2, dim=1).cpu())
            gallery_pids.append(batch['pids'])
    gallery_feats = torch.cat(gallery_feats, dim=0)
    gallery_pids = torch.cat(gallery_pids, dim=0)

    query_feats, query_pids = [], []
    print("Extracting Query...")
    with torch.no_grad():
        for batch in tqdm(test_txt_loader):
            inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'caption_input_ids', 'caption_attention_mask']}
            if 'caption_input_ids' in inputs:
                inputs['input_ids'] = inputs.pop('caption_input_ids')
                inputs['attention_mask'] = inputs.pop('caption_attention_mask')
            feats = model.get_text_features(inputs)
            query_feats.append(F.normalize(feats, p=2, dim=1).cpu())
            query_pids.append(batch['pids'])
    query_feats = torch.cat(query_feats, dim=0)
    query_pids = torch.cat(query_pids, dim=0)

    print("Computing Matrix...")
    sim_matrix = torch.matmul(query_feats, gallery_feats.t())

    sp_list, sn_list = [], []
    print("Filtering pairs...")
    for i in tqdm(range(sim_matrix.shape[0])):
        q_pid = query_pids[i]
        q_sims = sim_matrix[i]
        is_pos = (gallery_pids == q_pid)
        is_neg = ~is_pos
        if is_pos.sum() > 0 and is_neg.sum() > 0:
            sp = q_sims[is_pos].max().item()
            sn = q_sims[is_neg].max().item()
            sp_list.append(sp)
            sn_list.append(sn)

    return np.array(sp_list), np.array(sn_list)

# --- [MOD] Hàm vẽ được chỉnh sửa ---
def plot_analysis(ax, sp, sn, title, target_margin=None):
    total = len(sp)
    pass_rate = (np.sum(sp > sn) / total) * 100
    avg_gap = np.mean(sp - sn)

    # Scatter: Tăng s (size) và giảm alpha để nhìn rõ mật độ hơn
    is_correct = sp > sn
    # Tăng kích thước điểm (s=20) để dễ nhìn hơn trên hình lớn
    ax.scatter(sn[is_correct], sp[is_correct], s=20, c='green', alpha=0.15, label=r'Correct', edgecolors='none')
    ax.scatter(sn[~is_correct], sp[~is_correct], s=20, c='red', alpha=0.15, label=r'Incorrect', edgecolors='none')
    
    # Centroid: Làm to nổi bật hẳn (s=300)
    ax.scatter(np.mean(sn), np.mean(sp), c='black', marker='X', s=300, edgecolors='white', linewidth=2, zorder=10, label='Centroid')

    limit_max = 0.8 # Mở rộng nhẹ giới hạn để chữ không bị cắt
    
    # 1. Decision Boundary (y=x)
    ax.plot([0, limit_max], [0, limit_max], 'k--', linewidth=2, label='y=x')

    # 2. Target Margin
    if target_margin is not None and target_margin > 0:
        radius_target = (1 - target_margin) / np.sqrt(2)
        x_c = np.linspace(0, radius_target, 200)
        y_c = 1 - np.sqrt(np.clip(radius_target**2 - x_c**2, 0, None))
        mask = (x_c <= limit_max) & (y_c >= 0)
        ax.plot(x_c[mask], y_c[mask], color='orange', linestyle='--', linewidth=3, label=f'Target (m={target_margin})')

    # 3. Empirical Boundary
    d_centroid = np.sqrt(np.mean(sn)**2 + (np.mean(sp) - 1)**2)
    x_eff = np.linspace(0, d_centroid, 200)
    y_eff = 1 - np.sqrt(np.clip(d_centroid**2 - x_eff**2, 0, None))
    mask_eff = (x_eff <= limit_max) & (y_eff >= 0)
    
    if target_margin is not None and target_margin > 0:
        ax.plot(x_eff[mask_eff], y_eff[mask_eff], color='blue', linestyle='-', linewidth=3.5, label='Learned Geometry')

    ax.set_xlim(0, limit_max)
    ax.set_ylim(0, limit_max)
    
    # Label trục đậm hơn
    ax.set_xlabel(r'$s_n$ (Negative)', fontweight='bold')
    ax.set_ylabel(r'$s_p$ (Positive)', fontweight='bold')
    ax.set_title(title, pad=15, fontweight='bold') # Pad để đẩy title lên cao một chút
    
    ax.grid(True, linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Stats Box: Tăng fontsize và padding
    stats_text = f"Rank-1: {pass_rate:.1f}%\nAvg Gap: {avg_gap:.3f}"
    # transform=ax.transAxes giúp cố định vị trí theo khung hình
    ax.text(0.04, 0.96, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=16, fontweight='medium',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, edgecolor='gray'))
    
    # Legend: Di chuyển xuống góc dưới phải, chữ to
    ax.legend(loc='lower right', framealpha=0.95, fancybox=True)

def main():
    # Gọi hàm setup style đầu tiên
    set_publication_style()

    CONFIG_PATH = "/mnt/data/user_data/lampt/PS/code/outputs/2026-01-16/10-20-32/.hydra/config.yaml"
    CKPT_BASELINE = "/mnt/data/user_data/lampt/PS/code/epoch=56-val_score=49.15.ckpt" 
    CKPT_OURS = "/mnt/data/user_data/lampt/PS/code/checkpoints/vn3k-curri/epoch=53-val_score=51.30.ckpt"
    
    print(">>> Processing Baseline...")
    sp_base, sn_base = extract_test_scores(CONFIG_PATH, CKPT_BASELINE, enable_lora=False)
    
    print("\n>>> Processing Ours...")
    sp_ours, sn_ours = extract_test_scores(CONFIG_PATH, CKPT_OURS, enable_lora=True)

    # Tăng kích thước ảnh tổng thể (Figure Size)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8)) # 16x8 inch để có không gian cho chữ to
    
    plot_analysis(axes[0], sp_base, sn_base, "Baseline (mSigLIP)", target_margin=0.0)
    plot_analysis(axes[1], sp_ours, sn_ours, "Ours (LoRA + Circle)", target_margin=0.25)
    
    plt.tight_layout()
    plt.savefig("distribution_final_v5_pub.png", dpi=300, bbox_inches='tight') # bbox_inches='tight' để không bị cắt chữ
    print("Saved distribution_final_v5_pub.png")
    # plt.show() # Comment lại nếu chạy trên server không có màn hình

if __name__ == "__main__":
    main()