"""
Forward (Direct) vs Inverse Problem Sketch

Creates a diagram showing:
  - Forward: Model + Parameters -> Solution
  - Inverse: Model + Data -> Parameters
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def draw_box(ax, x, y, w, h, text, color='#2196F3', textcolor='white', fontsize=9):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.08",
                         facecolor=color, edgecolor='white',
                         linewidth=1.5, alpha=0.95)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=textcolor, fontweight='bold', family='sans-serif')


def draw_arrow(ax, x1, y1, x2, y2, color='white', lw=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                               connectionstyle='arc3,rad=0'))


def make_forward_inverse_figure():
    """Forward (direct) vs Inverse problem sketch."""
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), facecolor='black')

    box_w = 1.2
    box_h = 0.55
    plus_gap = 0.35

    for ax in axes:
        ax.set_facecolor('black')
        ax.set_xlim(-1, 8.5)
        ax.set_ylim(-1.0, 1.0)
        ax.set_aspect('equal')
        ax.set_axis_off()

    # ---- Forward / Direct Problem ----
    ax = axes[0]
    ax.text(3.75, 0.85, 'Forward (Direct) Problem', fontsize=18,
            ha='center', va='top', color='white', fontweight='bold', family='sans-serif')

    model_x = 0.5
    draw_box(ax, model_x, 0.0, box_w, box_h, 'Model', color='#2196F3', fontsize=13)
    ax.text(model_x, -0.48, r'$\mathcal{F}(u) = 0$', fontsize=10, ha='center', va='top',
            color='#90CAF9', style='italic')

    ax.text(model_x + box_w/2 + plus_gap, 0.0, '+', fontsize=20, ha='center', va='center',
            color='white', fontweight='bold')

    param_x = model_x + box_w + 2 * plus_gap + box_w/2
    draw_box(ax, param_x, 0.0, box_w, box_h, 'Parameters', color='#4CAF50', fontsize=13)
    ax.text(param_x, -0.48, r'coefficients, BCs, ICs', fontsize=9, ha='center', va='top',
            color='#A5D6A7', style='italic')

    arrow_start = param_x + box_w/2 + 0.1
    arrow_end = arrow_start + 1.5
    draw_arrow(ax, arrow_start, 0.0, arrow_end, 0.0, color='white', lw=3)

    sol_x = arrow_end + box_w/2 + 0.1
    draw_box(ax, sol_x, 0.0, box_w, box_h, 'Solution', color='#FF9800', fontsize=13)
    ax.text(sol_x, -0.48, r'$u(x, t)$', fontsize=10, ha='center', va='top',
            color='#FFCC80', style='italic')

    # ---- Inverse Problem ----
    ax = axes[1]
    ax.text(3.75, 0.85, 'Inverse Problem', fontsize=18,
            ha='center', va='top', color='white', fontweight='bold', family='sans-serif')

    draw_box(ax, model_x, 0.0, box_w, box_h, 'Model', color='#2196F3', fontsize=13)
    ax.text(model_x, -0.48, r'$\mathcal{F}(u) = 0$', fontsize=10, ha='center', va='top',
            color='#90CAF9', style='italic')

    ax.text(model_x + box_w/2 + plus_gap, 0.0, '+', fontsize=20, ha='center', va='center',
            color='white', fontweight='bold')

    draw_box(ax, param_x, 0.0, box_w, box_h, 'Data', color='#E91E63', fontsize=13)
    ax.text(param_x, -0.48, r'observations $\hat{u}$', fontsize=9, ha='center', va='top',
            color='#F48FB1', style='italic')

    draw_arrow(ax, arrow_start, 0.0, arrow_end, 0.0, color='white', lw=3)

    draw_box(ax, sol_x, 0.0, box_w, box_h, 'Parameters', color='#4CAF50', fontsize=13)
    ax.text(sol_x, -0.48, r'coefficients, BCs, ICs', fontsize=9, ha='center', va='top',
            color='#A5D6A7', style='italic')

    plt.tight_layout()
    plt.savefig('forward_inverse_sketch.pdf', dpi=150, facecolor='black',
                bbox_inches='tight', pad_inches=0.3)
    print("Saved: forward_inverse_sketch.pdf")
    plt.close()


if __name__ == '__main__':
    make_forward_inverse_figure()
