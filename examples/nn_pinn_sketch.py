"""
Neural Network & PINN Process Sketches

Creates two visual diagrams:
  1. Standard Neural Network supervised learning process
  2. Physics-Informed Neural Network (PINN) process
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ============================================================================
# Shared drawing helpers (same style as forward_fem_sketch.py)
# ============================================================================

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


# ============================================================================
# Mini-icon drawing functions for NN sketch
# ============================================================================

def icon_data(ax, cx, cy, s):
    """Scatter plot of (x, y) data points."""
    np.random.seed(3)
    n = 12
    xd = np.random.uniform(-1, 1, n) * s + cx
    yd = np.random.uniform(-1, 1, n) * s + cy
    ax.plot(xd, yd, 'o', color='#00BFFF', markersize=3, alpha=0.9, zorder=3)


def icon_network(ax, cx, cy, s):
    """Tiny neural network cartoon (3 layers)."""
    layers = [1, 3, 1]
    xs = [cx - s*0.8, cx, cx + s*0.8]
    positions = []
    for li, (nx, lx) in enumerate(zip(layers, xs)):
        ys = [cy + (j - (nx-1)/2) * s*0.55 for j in range(nx)]
        positions.append([(lx, yy) for yy in ys])
    # edges
    for l in range(len(layers)-1):
        for p1 in positions[l]:
            for p2 in positions[l+1]:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        color='#9B59B6', linewidth=0.7, alpha=0.6, zorder=2)
    # nodes
    for l, col in zip(positions, ['#4ECDC4', '#9B59B6', '#E74C3C']):
        for p in l:
            ax.plot(p[0], p[1], 'o', color=col, markersize=4,
                    markeredgecolor='white', markeredgewidth=0.5, zorder=4)


def icon_forward(ax, cx, cy, s):
    """Arrow going through a small box = forward pass."""
    # small box
    bw, bh = s*1.2, s*0.9
    rect = FancyBboxPatch((cx - bw/2, cy - bh/2), bw, bh,
                          boxstyle="round,pad=0.02", facecolor='#1a1a2e',
                          edgecolor='#00BCD4', linewidth=1, zorder=2)
    ax.add_patch(rect)
    ax.text(cx, cy, r'$f_\theta$', fontsize=9, ha='center', va='center',
            color='#00BCD4', fontweight='bold', zorder=3)
    # arrow through
    ax.annotate('', xy=(cx + bw/2 + s*0.15, cy),
                xytext=(cx - bw/2 - s*0.15, cy),
                arrowprops=dict(arrowstyle='->', color='#00BCD4', lw=1.2), zorder=3)


def icon_loss(ax, cx, cy, s):
    """Parabola representing loss function."""
    t = np.linspace(-1, 1, 40) * s
    l = 0.8 * s * (t / s)**2 + cy - s*0.4
    ax.plot(t + cx, l, color='#FF6B6B', linewidth=1.5, zorder=3)
    ax.plot(cx, cy - s*0.4, 'o', color='#2ECC71', markersize=4, zorder=4)


def icon_backprop(ax, cx, cy, s):
    """Backward arrows representing gradient flow."""
    for dy in [-0.4, 0, 0.4]:
        ax.annotate('', xy=(cx - s*0.7, cy + dy*s),
                    xytext=(cx + s*0.7, cy + dy*s),
                    arrowprops=dict(arrowstyle='->', color='#F39C12', lw=1.3), zorder=3)
    ax.text(cx, cy - s*0.8, r'$\frac{\partial L}{\partial \theta}$', fontsize=8,
            ha='center', va='center', color='#F39C12', zorder=3)


def icon_update(ax, cx, cy, s):
    """Parameter update: θ ← θ − η∇L."""
    ax.text(cx, cy + s*0.15, r'$\theta \leftarrow \theta - \eta \nabla L$', fontsize=8,
            ha='center', va='center', color='#2ECC71', fontweight='bold', zorder=3)
    # circular arrow
    t = np.linspace(0.3, 2*np.pi - 0.5, 50)
    r = s * 0.55
    ax.plot(r*np.cos(t) + cx, r*np.sin(t)*0.5 + cy - s*0.25, color='#2ECC71',
            linewidth=1.2, zorder=3)


# ============================================================================
# Mini-icon drawing functions for PINN sketch
# ============================================================================

def icon_domain(ax, cx, cy, s):
    """Blob-shaped domain Ω."""
    t = np.linspace(0, 2*np.pi, 100)
    r = s * (1 + 0.15*np.sin(3*t) + 0.1*np.cos(5*t))
    ax.fill(r*np.cos(t) + cx, r*np.sin(t) + cy,
            color='#FF9800', alpha=0.3, edgecolor='#FF9800', linewidth=1.2)
    ax.text(cx, cy, 'Ω', fontsize=11, ha='center', va='center',
            color='#FF9800', fontweight='bold', style='italic', zorder=3)


def icon_collocation(ax, cx, cy, s):
    """Random dots inside a domain circle."""
    t = np.linspace(0, 2*np.pi, 60)
    ax.plot(s*0.9*np.cos(t) + cx, s*0.9*np.sin(t) + cy,
            color='#2196F3', linewidth=1, alpha=0.5, zorder=2)
    np.random.seed(10)
    n = 20
    r = np.random.uniform(0, s*0.8, n)
    a = np.random.uniform(0, 2*np.pi, n)
    ax.plot(r*np.cos(a) + cx, r*np.sin(a) + cy, '.',
            color='#2196F3', markersize=3, zorder=3)


def icon_pde_loss(ax, cx, cy, s):
    """PDE residual equation."""
    ax.text(cx, cy + s*0.15, r'$\mathcal{R}[u_\theta]$', fontsize=10,
            ha='center', va='center', color='#4CAF50', fontweight='bold', zorder=3)
    ax.text(cx, cy - s*0.35, r'$\approx 0$', fontsize=8,
            ha='center', va='center', color='#4CAF50', alpha=0.8, zorder=3)


def icon_bc_loss(ax, cx, cy, s):
    """Boundary condition loss icon."""
    t = np.linspace(0, 2*np.pi, 50)
    r = s * 0.7
    ax.plot(r*np.cos(t) + cx, r*np.sin(t) + cy,
            color='#9C27B0', linewidth=1.5, alpha=0.8, zorder=2)
    angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    for a in angles:
        ax.plot(cx + r*np.cos(a), cy + r*np.sin(a), '^',
                color='#FF5722', markersize=5, zorder=4)
    ax.text(cx, cy, 'BC', fontsize=7, ha='center', va='center',
            color='#9C27B0', fontweight='bold', zorder=3)


def icon_total_loss(ax, cx, cy, s):
    """Sum of losses."""
    ax.text(cx, cy + s*0.1, r'$L_{pde} + L_{bc}$', fontsize=8,
            ha='center', va='center', color='#FF6B6B', fontweight='bold', zorder=3)
    # small parabola
    t = np.linspace(-1, 1, 30) * s * 0.6
    l = 0.5 * s * (t / (s*0.6))**2 + cy - s*0.4
    ax.plot(t + cx, l, color='#FF6B6B', linewidth=1.2, zorder=3)


# ============================================================================
# Figure builders
# ============================================================================

def make_nn_figure():
    """Standard Neural Network supervised learning process."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(22, 4.5), facecolor='black')
    ax.set_facecolor('black')
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-1.2, 1.5)
    ax.set_aspect('equal')
    ax.set_axis_off()

    ax.text(2.75, 1.35, 'Neural Network Training Process', fontsize=20,
            ha='center', va='top', color='white', fontweight='bold', family='sans-serif')

    box_w = 0.7
    box_h = 0.35
    y_center = 0.55
    spacing = 1.1
    x_positions = [i * spacing for i in range(6)]
    positions = [(x, y_center) for x in x_positions]

    labels = ['1. Data', '2. Network', '3. Forward', '4. Data Loss', '5. Backprop', '6. Update']
    colors = ['#00BFFF', '#9B59B6', '#00BCD4', '#FF6B6B', '#F39C12', '#2ECC71']

    # Draw boxes
    for pos, label, color in zip(positions, labels, colors):
        draw_box(ax, pos[0], pos[1], box_w, box_h, label, color=color, fontsize=9)

    # Forward arrows between boxes
    arrow_off = box_w/2 + 0.05
    for i in range(len(positions) - 1):
        draw_arrow(ax, positions[i][0] + arrow_off, positions[i][1],
                   positions[i+1][0] - arrow_off, positions[i+1][1])

    # Icons
    icon_offset = 0.65
    icon_size = 0.25
    icons = [icon_data, icon_network, icon_forward, icon_loss, icon_backprop, icon_update]
    for pos, icon_fn in zip(positions, icons):
        icon_fn(ax, pos[0], pos[1] - icon_offset, icon_size)

    # Subtexts above boxes
    subtexts = [
        '(x, y) pairs',
        r'$f_\theta(x)$',
        r'$\hat{y} = f_\theta(x)$',
        r'$L(\hat{y}, y)$',
        r'$\nabla_\theta L$',
        r'$\theta \leftarrow \theta - \eta \nabla L$',
    ]
    for pos, st in zip(positions, subtexts):
        ax.text(pos[0], pos[1] + 0.32, st, fontsize=7, ha='center', va='center',
                color='gray', style='italic', family='sans-serif')

    # Loop arrow: Update -> Forward (steps 6 -> 3)
    update_x = positions[5][0]
    forward_x = positions[2][0]
    icon_bottom = y_center - icon_offset - icon_size - 0.05
    loop_y = y_center - icon_offset - icon_size - 0.35

    ax.annotate('', xy=(update_x, loop_y), xytext=(update_x, icon_bottom),
                arrowprops=dict(arrowstyle='-', color='#2ECC71', lw=2.5), zorder=1)
    ax.annotate('', xy=(forward_x, loop_y), xytext=(update_x, loop_y),
                arrowprops=dict(arrowstyle='-', color='#2ECC71', lw=2.5), zorder=1)
    ax.annotate('', xy=(forward_x, icon_bottom), xytext=(forward_x, loop_y),
                arrowprops=dict(arrowstyle='->', color='#2ECC71', lw=2.5), zorder=1)

    mid_x = (update_x + forward_x) / 2
    ax.text(mid_x, loop_y - 0.18, 'Repeat until convergence', fontsize=9,
            ha='center', va='center', color='#2ECC71', fontweight='bold',
            style='italic', family='sans-serif',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                      edgecolor='#2ECC71', alpha=0.8),
            zorder=5)

    # Black backgrounds behind icons to cover the loop arrow
    icon_bg_pad = 0.32
    for pos in positions:
        bg = FancyBboxPatch(
            (pos[0] - icon_bg_pad, pos[1] - icon_offset - icon_bg_pad),
            2*icon_bg_pad, 2*icon_bg_pad,
            boxstyle="round,pad=0.02", facecolor='black', edgecolor='none',
            alpha=0.7, zorder=2)
        ax.add_patch(bg)

    # Re-draw icons on top of backgrounds
    for pos, icon_fn in zip(positions, icons):
        icon_fn(ax, pos[0], pos[1] - icon_offset, icon_size)

    plt.tight_layout()
    plt.savefig('nn_process_sketch.pdf', dpi=150, facecolor='black',
                bbox_inches='tight', pad_inches=0.2)
    print("Saved: nn_process_sketch.pdf")
    plt.close()


def make_pinn_figure():
    """Physics-Informed Neural Network (PINN) process with parallel loss branches."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(24, 6), facecolor='black')
    ax.set_facecolor('black')
    ax.set_xlim(-0.5, 9.2)
    ax.set_ylim(-2.0, 2.2)
    ax.set_aspect('equal')
    ax.set_axis_off()

    ax.text(4.3, 2.05, 'Physics-Informed Neural Network (PINN) Process', fontsize=20,
            ha='center', va='top', color='white', fontweight='bold', family='sans-serif')

    box_w = 0.7
    box_h = 0.35
    y_center = 0.55
    spacing = 1.1
    arrow_off = box_w / 2 + 0.05
    icon_offset = 0.65
    icon_size = 0.25

    # --- Sequential boxes: Domain, Sample, Network, Forward ---
    seq_labels = ['1. Domain', '2. Sample', '3. Network', '4. Forward']
    seq_colors = ['#FF9800', '#2196F3', '#9B59B6', '#00BCD4']
    seq_x = [i * spacing for i in range(4)]
    seq_positions = [(x, y_center) for x in seq_x]

    for pos, label, color in zip(seq_positions, seq_labels, seq_colors):
        draw_box(ax, pos[0], pos[1], box_w, box_h, label, color=color, fontsize=9)

    seq_subtexts = ['Geometry Ω', 'Collocation pts', r'$u_\theta(x)$', r'$\hat{u} = u_\theta(x)$']
    for pos, st in zip(seq_positions, seq_subtexts):
        ax.text(pos[0], pos[1] + 0.32, st, fontsize=7, ha='center', va='center',
                color='gray', style='italic', family='sans-serif')

    # Arrows between sequential boxes
    for i in range(len(seq_positions) - 1):
        draw_arrow(ax, seq_positions[i][0] + arrow_off, seq_positions[i][1],
                   seq_positions[i+1][0] - arrow_off, seq_positions[i+1][1])

    # --- Parallel loss branches from Forward ---
    branch_x = seq_x[3] + spacing  # x for the parallel loss boxes
    branch_ys = [y_center + 0.8, y_center, y_center - 0.8]  # top, mid, bottom

    loss_labels = ['5a. PDE Loss', '5b. BC Loss', '5c. Data Loss']
    loss_colors = ['#4CAF50', '#9C27B0', '#00BFFF']
    loss_subtexts = [
        r'$\mathcal{R}[u_\theta] \approx 0$',
        r'$u_\theta|_{\partial\Omega}$',
        r'$L(\hat{y}, y)$',
    ]
    loss_icons = [icon_pde_loss, icon_bc_loss, icon_loss]

    for (by, label, color, st) in zip(branch_ys, loss_labels, loss_colors, loss_subtexts):
        draw_box(ax, branch_x, by, box_w, box_h, label, color=color, fontsize=8)
        ax.text(branch_x, by + 0.32, st, fontsize=7, ha='center', va='center',
                color='gray', style='italic', family='sans-serif')

    # Arrows: Forward -> each loss branch (fan-out)
    fwd_x = seq_x[3]
    for by in branch_ys:
        draw_arrow(ax, fwd_x + arrow_off, y_center,
                   branch_x - arrow_off, by)

    # --- Remaining sequential: Backprop, Update ---
    tail_labels = ['6. Backprop', '7. Update']
    tail_colors = ['#F39C12', '#2ECC71']
    backprop_x = branch_x + spacing
    tail_x = [backprop_x, backprop_x + spacing]
    tail_positions = [(x, y_center) for x in tail_x]

    for pos, label, color in zip(tail_positions, tail_labels, tail_colors):
        draw_box(ax, pos[0], pos[1], box_w, box_h, label, color=color, fontsize=9)

    tail_subtexts = [r'$\nabla_\theta L$', r'$\theta \leftarrow \theta - \eta \nabla L$']
    for pos, st in zip(tail_positions, tail_subtexts):
        ax.text(pos[0], pos[1] + 0.32, st, fontsize=7, ha='center', va='center',
                color='gray', style='italic', family='sans-serif')

    # Arrows: each loss branch -> Backprop (fan-in)
    for by in branch_ys:
        draw_arrow(ax, branch_x + arrow_off, by,
                   backprop_x - arrow_off, y_center)

    # Arrow: Backprop -> Update
    draw_arrow(ax, tail_positions[0][0] + arrow_off, y_center,
               tail_positions[1][0] - arrow_off, y_center)

    # --- Loop arrow: Update -> Forward ---
    update_x = tail_x[1]
    network_x = seq_x[3]
    icon_bottom = y_center - icon_offset - icon_size - 0.05
    loop_y = y_center - icon_offset - icon_size - 0.55

    ax.annotate('', xy=(update_x, loop_y), xytext=(update_x, icon_bottom),
                arrowprops=dict(arrowstyle='-', color='#2ECC71', lw=2.5), zorder=1)
    ax.annotate('', xy=(network_x, loop_y), xytext=(update_x, loop_y),
                arrowprops=dict(arrowstyle='-', color='#2ECC71', lw=2.5), zorder=1)
    ax.annotate('', xy=(network_x, icon_bottom), xytext=(network_x, loop_y),
                arrowprops=dict(arrowstyle='->', color='#2ECC71', lw=2.5), zorder=1)

    mid_loop_x = (update_x + network_x) / 2
    ax.text(mid_loop_x, loop_y - 0.18, 'Repeat until convergence', fontsize=9,
            ha='center', va='center', color='#2ECC71', fontweight='bold',
            style='italic', family='sans-serif',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                      edgecolor='#2ECC71', alpha=0.8),
            zorder=5)

    # --- Icons below sequential boxes ---
    seq_icon_fns = [icon_domain, icon_collocation, icon_network, icon_forward]
    # Black backgrounds behind icons
    icon_bg_pad = 0.32
    for pos in seq_positions:
        bg = FancyBboxPatch(
            (pos[0] - icon_bg_pad, pos[1] - icon_offset - icon_bg_pad),
            2*icon_bg_pad, 2*icon_bg_pad,
            boxstyle="round,pad=0.02", facecolor='black', edgecolor='none',
            alpha=0.7, zorder=2)
        ax.add_patch(bg)
    for pos, icon_fn in zip(seq_positions, seq_icon_fns):
        icon_fn(ax, pos[0], pos[1] - icon_offset, icon_size)

    # Icons below backprop, update
    tail_icon_fns = [icon_backprop, icon_update]
    for pos in tail_positions:
        bg = FancyBboxPatch(
            (pos[0] - icon_bg_pad, pos[1] - icon_offset - icon_bg_pad),
            2*icon_bg_pad, 2*icon_bg_pad,
            boxstyle="round,pad=0.02", facecolor='black', edgecolor='none',
            alpha=0.7, zorder=2)
        ax.add_patch(bg)
    for pos, icon_fn in zip(tail_positions, tail_icon_fns):
        icon_fn(ax, pos[0], pos[1] - icon_offset, icon_size)

    plt.tight_layout()
    plt.savefig('pinn_process_sketch.pdf', dpi=150, facecolor='black',
                bbox_inches='tight', pad_inches=0.2)
    print("Saved: pinn_process_sketch.pdf")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    make_nn_figure()
    make_pinn_figure()

if __name__ == '__main__':
    main()
