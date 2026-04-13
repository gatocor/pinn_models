"""
Forward FEM Process Sketch

Creates a visual diagram showing the steps of the forward
Finite Element Method (FEM) process.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.tri import Triangulation
from scipy.spatial import Delaunay

def draw_box(ax, x, y, w, h, text, color='#2196F3', textcolor='white', fontsize=11):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.08",
                         facecolor=color, edgecolor='white',
                         linewidth=1.5, alpha=0.95)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=textcolor, fontweight='bold', family='sans-serif')

def draw_arrow(ax, x1, y1, x2, y2, color='white'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2,
                               connectionstyle='arc3,rad=0'))

def create_mini_mesh(ax, cx, cy, size):
    """Draw a small triangulated mesh in a box."""
    np.random.seed(7)
    n = 25
    x = np.random.uniform(-1, 1, n) * size + cx
    y = np.random.uniform(-1, 1, n) * size + cy
    # Add boundary points
    t = np.linspace(0, 2*np.pi, 16, endpoint=False)
    x = np.concatenate([x, size * np.cos(t) + cx])
    y = np.concatenate([y, size * np.sin(t) + cy])
    tri = Delaunay(np.column_stack([x, y]))
    triang = Triangulation(x, y, tri.simplices)
    ax.triplot(triang, color='dodgerblue', linestyle='-', linewidth=0.6, alpha=0.8)
    ax.plot(x, y, 'o', color='dodgerblue', markersize=1.5, alpha=0.8)

def create_mini_domain(ax, cx, cy, size):
    """Draw a small domain shape."""
    t = np.linspace(0, 2*np.pi, 100)
    r = size * (1 + 0.15*np.sin(3*t) + 0.1*np.cos(5*t))
    ax.fill(r*np.cos(t) + cx, r*np.sin(t) + cy, 
            color='#FF9800', alpha=0.3, edgecolor='#FF9800', linewidth=1.5)
    ax.text(cx, cy, 'Ω', fontsize=14, ha='center', va='center', 
            color='#FF9800', fontweight='bold', style='italic')

def create_mini_equation(ax, cx, cy):
    """Draw a small equation representation."""
    ax.text(cx, cy + 0.02, r'$\nabla^2 u = f$', fontsize=11, ha='center', va='center',
            color='#4CAF50', fontweight='bold', style='italic')

def create_mini_matrix(ax, cx, cy, size):
    """Draw a small matrix representation."""
    s = size * 0.7
    # Draw bracket-like lines
    ax.plot([cx-s, cx-s*1.1, cx-s*1.1, cx-s], 
            [cy-s, cy-s, cy+s, cy+s], color='#E91E63', linewidth=1.5)
    ax.plot([cx+s, cx+s*1.1, cx+s*1.1, cx+s], 
            [cy-s, cy-s, cy+s, cy+s], color='#E91E63', linewidth=1.5)
    # Grid dots for matrix entries
    for i in range(4):
        for j in range(4):
            xi = cx - s*0.7 + j * s*0.47
            yi = cy - s*0.7 + i * s*0.47
            alpha = 0.8 if abs(i-j) <= 1 else 0.2
            ax.plot(xi, yi, 's', color='#E91E63', markersize=3, alpha=alpha)
    ax.text(cx, cy - s*1.4, r'$\mathbf{K u} = \mathbf{f}$', fontsize=9, 
            ha='center', va='center', color='#E91E63', fontweight='bold')

def create_mini_bc(ax, cx, cy, size):
    """Draw boundary condition icons."""
    # Small domain outline
    t = np.linspace(0, 2*np.pi, 50)
    r = size * 0.8
    ax.plot(r*np.cos(t) + cx, r*np.sin(t) + cy, 
            color='#9C27B0', linewidth=1.5, alpha=0.8)
    # BC markers
    angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    for a in angles:
        bx = cx + r*np.cos(a)
        by = cy + r*np.sin(a)
        ax.plot(bx, by, '^', color='#FF5722', markersize=6, zorder=5)
    ax.text(cx, cy, 'BC', fontsize=8, ha='center', va='center', 
            color='#9C27B0', fontweight='bold')

def create_mini_solution(ax, cx, cy, size):
    """Draw a small contour-like solution."""
    n = 30
    xi = np.linspace(-1, 1, n) * size + cx
    yi = np.linspace(-1, 1, n) * size + cy
    X, Y = np.meshgrid(xi, yi)
    Z = np.exp(-((X-cx)**2 + (Y-cy)**2) / (size*0.5)**2)
    mask = (X-cx)**2 + (Y-cy)**2 < (size*0.9)**2
    Z[~mask] = np.nan
    ax.contourf(X, Y, Z, levels=8, cmap='copper', alpha=0.7)
    ax.contour(X, Y, Z, levels=6, colors='white', linewidths=0.4, alpha=0.5)

def draw_forward(ax, positions, box_w, box_h, y_center, icon_offset, icon_size):
    """Draw the forward FEM process icons and subtexts."""
    # Draw mini illustrations below each step
    for i in range(6):
        ix = positions[i][0]
        iy = positions[i][1] - icon_offset
    
    create_mini_domain(ax, positions[0][0], positions[0][1] - icon_offset, icon_size)
    create_mini_mesh(ax, positions[1][0], positions[1][1] - icon_offset, icon_size)
    create_mini_equation(ax, positions[2][0], positions[2][1] - icon_offset)
    create_mini_matrix(ax, positions[3][0], positions[3][1] - icon_offset, icon_size)
    create_mini_bc(ax, positions[4][0], positions[4][1] - icon_offset, icon_size)
    create_mini_solution(ax, positions[5][0], positions[5][1] - icon_offset, icon_size)

    subtexts = [
        'Geometry Ω',
        'Triangulation',
        'Weak Form',
        'Kᵉ → K, fᵉ → f',
        'Dirichlet / Neumann',
        'u = K⁻¹ f',
    ]
    for pos, st in zip(positions, subtexts):
        ax.text(pos[0], pos[1] + 0.32, st, fontsize=7, ha='center', va='center',
                color='gray', style='italic', family='sans-serif')


def make_forward_figure():
    """Create the forward FEM process figure."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(22, 4.5), facecolor='black')
    ax.set_facecolor('black')
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-1.2, 1.5)
    ax.set_aspect('equal')
    ax.set_axis_off()

    ax.text(2.75, 1.35, 'Forward FEM Process', fontsize=20, ha='center', va='top',
            color='white', fontweight='bold', family='sans-serif')

    box_w = 0.7
    box_h = 0.35
    y_center = 0.55
    spacing = 1.1
    x_positions = [i * spacing for i in range(6)]
    positions = [(x, y_center) for x in x_positions]

    labels = ['1. Domain', '2. Mesh', '3. PDE', '4. Assembly', '5. BCs', '6. Solve']
    colors = ['#FF9800', '#2196F3', '#4CAF50', '#E91E63', '#9C27B0', '#00BCD4']

    for pos, label, color in zip(positions, labels, colors):
        draw_box(ax, pos[0], pos[1], box_w, box_h, label, color=color, fontsize=9)

    arrow_offset_h = box_w/2 + 0.05
    for i in range(len(positions) - 1):
        draw_arrow(ax, positions[i][0] + arrow_offset_h, positions[i][1],
                   positions[i+1][0] - arrow_offset_h, positions[i+1][1])

    icon_offset = 0.65
    icon_size = 0.25
    draw_forward(ax, positions, box_w, box_h, y_center, icon_offset, icon_size)

    plt.tight_layout()
    plt.savefig('forward_fem_sketch.pdf', dpi=150, facecolor='black',
                bbox_inches='tight', pad_inches=0.2)
    print("Saved: forward_fem_sketch.pdf")


def make_inverse_figure():
    """Create the inverse FEM process figure."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(26, 5.5), facecolor='black')
    ax.set_facecolor('black')
    ax.set_xlim(-0.5, 7.7)
    ax.set_ylim(-1.6, 1.5)
    ax.set_aspect('equal')
    ax.set_axis_off()

    ax.text(3.5, 1.35, 'Inverse FEM Process', fontsize=20, ha='center', va='top',
            color='white', fontweight='bold', family='sans-serif')

    box_w = 0.7
    box_h = 0.35
    y_center = 0.55
    spacing = 1.1
    x_positions = [i * spacing for i in range(7)]
    positions = [(x, y_center) for x in x_positions]

    labels = ['1. Domain', '2. Mesh', '3. PDE', '4. Assembly', '5. BCs', '6. Solve', '7. Compare']
    colors = ['#FF9800', '#2196F3', '#4CAF50', '#E91E63', '#9C27B0', '#00BCD4', '#FF5722']

    for pos, label, color in zip(positions, labels, colors):
        draw_box(ax, pos[0], pos[1], box_w, box_h, label, color=color, fontsize=9)

    arrow_offset_h = box_w/2 + 0.05
    for i in range(len(positions) - 1):
        draw_arrow(ax, positions[i][0] + arrow_offset_h, positions[i][1],
                   positions[i+1][0] - arrow_offset_h, positions[i+1][1])

    icon_offset = 0.65
    icon_size = 0.25

    # Draw feedback arrow FIRST (behind everything)
    compare_x = positions[6][0]
    pde_x = positions[2][0]

    icon_bottom = y_center - icon_offset - icon_size - 0.05
    loop_y = y_center - icon_offset - icon_size - 0.35

    ax.annotate('', xy=(compare_x, loop_y), xytext=(compare_x, icon_bottom),
                arrowprops=dict(arrowstyle='-', color='#FF5722', lw=2.5), zorder=1)
    ax.annotate('', xy=(pde_x, loop_y), xytext=(compare_x, loop_y),
                arrowprops=dict(arrowstyle='-', color='#FF5722', lw=2.5), zorder=1)
    ax.annotate('', xy=(pde_x, icon_bottom), xytext=(pde_x, loop_y),
                arrowprops=dict(arrowstyle='->', color='#FF5722', lw=2.5), zorder=1)

    mid_x = (compare_x + pde_x) / 2
    ax.text(mid_x, loop_y - 0.18, 'Update parameters', fontsize=9, ha='center', va='center',
            color='#FF5722', fontweight='bold', style='italic', family='sans-serif',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', edgecolor='#FF5722', alpha=0.8),
            zorder=5)

    # Black backgrounds behind icons
    icon_bg_pad = 0.32
    for i_step in range(7):
        ix = positions[i_step][0]
        iy = positions[i_step][1] - icon_offset
        bg = FancyBboxPatch((ix - icon_bg_pad, iy - icon_bg_pad),
                            2*icon_bg_pad, 2*icon_bg_pad,
                            boxstyle="round,pad=0.02",
                            facecolor='black', edgecolor='none', alpha=0.7, zorder=2)
        ax.add_patch(bg)

    # Forward steps icons (1-6)
    draw_forward(ax, positions, box_w, box_h, y_center, icon_offset, icon_size)

    # 7. Compare to data icon
    cx, cy = positions[6][0], positions[6][1] - icon_offset
    t_data = np.linspace(-0.8, 0.8, 8) * icon_size + cx
    y_model = np.sin(np.linspace(0, np.pi, 8)) * icon_size * 0.8 + cy
    y_data = y_model + np.random.normal(0, icon_size * 0.15, 8)
    ax.plot(t_data, y_model, '-', color='#00BCD4', linewidth=1.5, alpha=0.8, zorder=3)
    ax.plot(t_data, y_data, 'o', color='#FF5722', markersize=4, alpha=0.9, zorder=3)
    ax.text(cx, cy - icon_size * 0.7, r'$\|u - d\|^2$', fontsize=9,
            ha='center', va='center', color='#FF5722', fontweight='bold', zorder=3)

    # Add 7th subtext
    ax.text(positions[6][0], positions[6][1] + 0.32, 'Loss(u, data)', fontsize=7,
            ha='center', va='center', color='gray', style='italic', family='sans-serif')

    plt.tight_layout()
    plt.savefig('inverse_fem_sketch.pdf', dpi=150, facecolor='black',
                bbox_inches='tight', pad_inches=0.2)
    print("Saved: inverse_fem_sketch.pdf")


def main():
    make_forward_figure()
    make_inverse_figure()

if __name__ == '__main__':
    main()
