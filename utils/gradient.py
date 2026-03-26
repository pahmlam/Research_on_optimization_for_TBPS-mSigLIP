import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as font_manager

# --- 1. CẤU HÌNH STYLE BÀI BÁO (HIGH VISIBILITY) ---
def set_publication_style():
    plt.rcParams.update({
        # Tăng kích thước font lên rất to để chống vỡ khi thu nhỏ hình
        'font.size': 20,           
        'axes.labelsize': 24,      # Nhãn trục (sn, sp)
        'axes.titlesize': 28,      # Tiêu đề biểu đồ
        'xtick.labelsize': 18,     # Số trên trục x
        'ytick.labelsize': 18,     # Số trên trục y
        'legend.fontsize': 20,     # Chú thích
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.labelweight': 'bold', # Chữ đậm cho dễ nhìn
        'axes.titleweight': 'bold',
    })

def clean_3d_axis(ax):
    """Làm sạch nền xám và lưới mặc định của biểu đồ 3D"""
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    # Làm lưới đậm hơn một chút để dễ nhìn khi in
    grid_color = (0.8, 0.8, 0.8, 1)
    ax.xaxis._axinfo["grid"]['color'] = grid_color
    ax.yaxis._axinfo["grid"]['color'] = grid_color
    ax.zaxis._axinfo["grid"]['color'] = grid_color

# --- 2. HÀM TÍNH TOÁN (GIỮ NGUYÊN) ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_absolute_gradients(sn_grid, sp_grid, gamma, margin, loss_type='baseline'):
    bias = -10.0 
    
    if loss_type == 'baseline':
        grad_sn = gamma * sigmoid(gamma * sn_grid + bias)
        grad_sp = gamma * (1 - sigmoid(gamma * sp_grid + bias))
        
    elif loss_type == 'circle':
        alpha_n = np.maximum(sn_grid + margin, 0)
        logit_n = gamma * alpha_n * (sn_grid - margin)
        grad_sn = gamma * alpha_n * sigmoid(logit_n)
        
        alpha_p = np.maximum(1 + margin - sp_grid, 0)
        logit_p = gamma * alpha_p * (sp_grid - (1 - margin))
        grad_sp = gamma * alpha_p * (1 - sigmoid(logit_p))
        
    return grad_sn, grad_sp

# --- 3. VẼ BIỂU ĐỒ ---
def plot_3d_absolute_optimized():
    set_publication_style()
    
    # Tăng độ phân giải
    n = 200
    s_n = np.linspace(0, 1, n)
    s_p = np.linspace(0, 1, n)
    SN, SP = np.meshgrid(s_n, s_p)
    
    gamma_base = 15.0
    gamma_circle = 80.0
    margin = 0.25

    base_gn, base_gp = calculate_absolute_gradients(SN, SP, gamma_base, 0, 'baseline')
    circ_gn, circ_gp = calculate_absolute_gradients(SN, SP, gamma_circle, margin, 'circle')

    # Tăng kích thước Figure để chứa font to (20x9 inch)
    fig = plt.figure(figsize=(20, 9))

    # ==========================================
    # PLOT 1: Gradient theo Sn (Negative)
    # ==========================================
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    clean_3d_axis(ax1)
    
    ax1.plot_surface(SN, SP, base_gn, color='#00CED1', alpha=0.3, 
                     rstride=8, cstride=8, linewidth=0, antialiased=True, shade=True)
    
    ax1.plot_surface(SN, SP, circ_gn, color='#DC143C', alpha=0.8, 
                     rstride=8, cstride=8, linewidth=0, antialiased=True, shade=True)

    ax1.contourf(SN, SP, circ_gn, zdir='z', offset=0, cmap='Reds', alpha=0.3)

    legend_1 = [
        Line2D([0], [0], color='#00CED1', lw=6, alpha=0.6, label='N-ITC Loss'), # Tăng lw lên 6
        Line2D([0], [0], color='#DC143C', lw=6, alpha=0.9, label='Circle Loss')
    ]
    ax1.legend(handles=legend_1, loc='upper left', frameon=False, bbox_to_anchor=(-0.1, 1.0))
    
    # Tăng labelpad lên 20-25 để chữ không đè vào số
    ax1.set_title("Gradient Magnitude\nw.r.t $s_n$ (Negative)", pad=20)
    ax1.set_xlabel('$s_n$ (Negative)', labelpad=25) 
    ax1.set_ylabel('$s_p$ (Positive)', labelpad=25)
    ax1.set_zlabel('Gradient Magnitude', labelpad=20)
    ax1.set_zlim(0, 120)
    
    ax1.view_init(elev=20, azim=-120)
    # Tăng kích thước tick labels riêng cho trục Z nếu cần
    ax1.tick_params(axis='z', pad=10)

    # ==========================================
    # PLOT 2: Gradient theo Sp (Positive)
    # ==========================================
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    clean_3d_axis(ax2)
    
    ax2.plot_surface(SN, SP, base_gp, color='#00CED1', alpha=0.3, 
                     rstride=8, cstride=8, linewidth=0, antialiased=True)
    
    ax2.plot_surface(SN, SP, circ_gp, color='#00008B', alpha=0.7, 
                     rstride=8, cstride=8, linewidth=0, antialiased=True)

    ax2.contourf(SN, SP, circ_gp, zdir='z', offset=0, cmap='Blues', alpha=0.3)

    legend_2 = [
        Line2D([0], [0], color='#00CED1', lw=6, alpha=0.6, label='N-ITC Loss'),
        Line2D([0], [0], color='#00008B', lw=6, alpha=0.9, label='Circle Loss')
    ]
    ax2.legend(handles=legend_2, loc='upper right', frameon=False, bbox_to_anchor=(1.1, 1.0))

    ax2.set_title("Gradient Magnitude\nw.r.t $s_p$ (Positive)", pad=20)
    ax2.set_xlabel('$s_n$ (Negative)', labelpad=25)
    ax2.set_ylabel('$s_p$ (Positive)', labelpad=25)
    ax2.set_zlabel('Gradient Magnitude', labelpad=20)
    ax2.set_zlim(0, 120)
    
    ax2.view_init(elev=30, azim=45)
    ax2.tick_params(axis='z', pad=10)

    # Căn chỉnh lề tự động
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15) 
    
    # Lưu file
    plt.savefig("gradient_3d_optimized_pub.png", dpi=300, bbox_inches='tight')
    print("Done! Saved to gradient_3d_optimized_pub.png")

if __name__ == "__main__":
    plot_3d_absolute_optimized()