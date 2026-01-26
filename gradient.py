import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as font_manager

# --- 1. CẤU HÌNH STYLE BÀI BÁO ---
def set_publication_style():
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 14,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
    })

def clean_3d_axis(ax):
    """Làm sạch nền xám và lưới mặc định của biểu đồ 3D"""
    # Làm nền trong suốt
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # Làm mờ lưới
    ax.xaxis._axinfo["grid"]['color'] =  (0.9, 0.9, 0.9, 1)
    ax.yaxis._axinfo["grid"]['color'] =  (0.9, 0.9, 0.9, 1)
    ax.zaxis._axinfo["grid"]['color'] =  (0.9, 0.9, 0.9, 1)

# --- 2. HÀM TÍNH TOÁN (LOGIC CỐT LÕI) ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_absolute_gradients(sn_grid, sp_grid, gamma, margin, loss_type='baseline'):
    bias = -10.0 
    
    if loss_type == 'baseline':
        # Baseline: N-ITC (Sigmoid thuần)
        grad_sn = gamma * sigmoid(gamma * sn_grid + bias)
        grad_sp = gamma * (1 - sigmoid(gamma * sp_grid + bias))
        
    elif loss_type == 'circle':
        # Ours: Circle Loss
        # Negative (Wall): Dựng tường chắn cứng tại sn > 0
        alpha_n = np.maximum(sn_grid + margin, 0)
        logit_n = gamma * alpha_n * (sn_grid - margin)
        grad_sn = gamma * alpha_n * sigmoid(logit_n)
        
        # Positive (Slope): Dốc trượt mạnh khi sp < 1
        alpha_p = np.maximum(1 + margin - sp_grid, 0)
        logit_p = gamma * alpha_p * (sp_grid - (1 - margin))
        grad_sp = gamma * alpha_p * (1 - sigmoid(logit_p))
        
    return grad_sn, grad_sp

# --- 3. VẼ BIỂU ĐỒ ---
def plot_3d_absolute_optimized():
    set_publication_style()
    
    # Tăng độ phân giải lên 200 để hình mịn hơn
    n = 200
    s_n = np.linspace(0, 1, n)
    s_p = np.linspace(0, 1, n)
    SN, SP = np.meshgrid(s_n, s_p)
    
    # Tham số
    gamma_base = 15.0
    gamma_circle = 80.0
    margin = 0.25

    # Tính toán
    base_gn, base_gp = calculate_absolute_gradients(SN, SP, gamma_base, 0, 'baseline')
    circ_gn, circ_gp = calculate_absolute_gradients(SN, SP, gamma_circle, margin, 'circle')

    # Khởi tạo hình
    fig = plt.figure(figsize=(18, 8))

    # ==========================================
    # PLOT 1: Gradient theo Sn (Negative)
    # ==========================================
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    clean_3d_axis(ax1)
    
    # Baseline: Màu xanh cyan nhạt, trong suốt hơn để làm nền
    surf1 = ax1.plot_surface(SN, SP, base_gn, color='#00CED1', alpha=0.3, 
                     rstride=8, cstride=8, linewidth=0, antialiased=True, shade=True)
    
    # Ours: Màu đỏ đậm, rõ ràng
    surf2 = ax1.plot_surface(SN, SP, circ_gn, color='#DC143C', alpha=0.8, 
                     rstride=8, cstride=8, linewidth=0, antialiased=True, shade=True)

    # Thêm bóng đổ (Contour) dưới đáy để dễ nhìn cường độ
    ax1.contourf(SN, SP, circ_gn, zdir='z', offset=0, cmap='Reds', alpha=0.3)

    # Legend & Labels
    legend_1 = [
        Line2D([0], [0], color='#00CED1', lw=4, alpha=0.6, label='N-ITC Loss'),
        Line2D([0], [0], color='#DC143C', lw=4, alpha=0.9, label='Circle Loss')
    ]
    ax1.legend(handles=legend_1, loc='upper left', frameon=False)
    
    ax1.set_title("Gradient Magnitude w.r.t $s_n$ (Negative)", fontweight='bold', pad=10)
    ax1.set_xlabel('\n$s_n$ (Negative)', linespacing=3.0) # Thêm newline để đẩy label ra xa trục
    ax1.set_ylabel('\n$s_p$ (Positive)', linespacing=3.0)
    ax1.set_zlabel('\nGradient Magnitude', linespacing=3.0)
    ax1.set_zlim(0, 120) # Mở rộng trục Z một chút
    
    # Góc nhìn: Nhìn ngang để thấy độ dốc dựng đứng (The Wall)
    ax1.view_init(elev=20, azim=-120)

    # ==========================================
    # PLOT 2: Gradient theo Sp (Positive)
    # ==========================================
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    clean_3d_axis(ax2)
    
    # Baseline
    ax2.plot_surface(SN, SP, base_gp, color='#00CED1', alpha=0.3, 
                     rstride=8, cstride=8, linewidth=0, antialiased=True)
    
    # Ours: Màu xanh navy đậm cho Positive
    ax2.plot_surface(SN, SP, circ_gp, color='#00008B', alpha=0.7, 
                     rstride=8, cstride=8, linewidth=0, antialiased=True)

    # Bóng đổ
    ax2.contourf(SN, SP, circ_gp, zdir='z', offset=0, cmap='Blues', alpha=0.3)

    legend_2 = [
        Line2D([0], [0], color='#00CED1', lw=4, alpha=0.6, label='N-ITC Loss'),
        Line2D([0], [0], color='#00008B', lw=4, alpha=0.9, label='Circle Loss')
    ]
    ax2.legend(handles=legend_2, loc='upper right', frameon=False)

    ax2.set_title("Gradient Magnitude w.r.t $s_p$ (Positive)", fontweight='bold', pad=10)
    ax2.set_xlabel('\n$s_n$ (Negative)', linespacing=3.0)
    ax2.set_ylabel('\n$s_p$ (Positive)', linespacing=3.0)
    ax2.set_zlabel('\nGradient Magnitude', linespacing=3.0)
    ax2.set_zlim(0, 120)
    
    # Góc nhìn: Nhìn chéo để thấy dốc trượt
    ax2.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1) # Giảm khoảng cách giữa 2 hình
    
    plt.savefig("gradient_3d_optimized_pub.png", dpi=300, bbox_inches='tight')
    print("Done! Saved to gradient_3d_optimized_pub.png")
    # plt.show()

if __name__ == "__main__":
    plot_3d_absolute_optimized()