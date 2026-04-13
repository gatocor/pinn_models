"""
Laser Ablation Visualization with Triangulated Mesh

Creates a 3D visualization of the laser ablation surface using
a triangulated mesh to show the cut geometry.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def create_laser_ablation_surface():
    """Create the laser ablation cut surface with irregular mesh."""
    
    # Smaller cut parameters - thin laser cut
    cut_border_x = 0.08
    cut_border_y = 0.005
    sigma = 0.002
    
    np.random.seed(42)
    
    # Domain bounds
    x_min, x_max = -0.5, 0.5
    y_min, y_max = -0.3, 0.3
    
    # Generate irregular scattered points with density varying by distance to cut
    n_points_coarse = 300   # Coarse points far from cut
    n_points_fine = 800     # Fine points near the cut
    
    # Coarse random points over the whole domain
    x_coarse = np.random.uniform(x_min, x_max, n_points_coarse)
    y_coarse = np.random.uniform(y_min, y_max, n_points_coarse)
    
    # Fine points concentrated near the cut (using rejection sampling)
    # Sample from a tighter distribution around the cut
    x_fine = np.random.normal(0, 0.1, n_points_fine * 3)
    y_fine = np.random.normal(0, 0.02, n_points_fine * 3)
    
    # Keep points within bounds
    mask = (np.abs(x_fine) < 0.25) & (np.abs(y_fine) < 0.05)
    x_fine = x_fine[mask][:n_points_fine]
    y_fine = y_fine[mask][:n_points_fine]
    
    # Extra ring of points exactly at the cut boundary for sharp transition
    n_boundary = 150
    theta = np.linspace(0, 2*np.pi, n_boundary, endpoint=False)
    # Add some noise to make it irregular
    theta += np.random.uniform(-0.1, 0.1, n_boundary)
    
    # Multiple rings at different radii near the cut
    rings = []
    for scale in [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]:
        x_ring = cut_border_x * scale * np.cos(theta) + np.random.normal(0, 0.001, n_boundary)
        y_ring = cut_border_y * scale * np.sin(theta) + np.random.normal(0, 0.0005, n_boundary)
        rings.append(np.column_stack([x_ring, y_ring]))
    
    # Combine all points
    points_coarse = np.column_stack([x_coarse, y_coarse])
    points_fine = np.column_stack([x_fine, y_fine])
    points_rings = np.vstack(rings)
    
    # Add boundary points to ensure clean edges
    n_edge = 30
    edge_x_top = np.linspace(x_min, x_max, n_edge)
    edge_x_bot = np.linspace(x_min, x_max, n_edge)
    edge_y_left = np.linspace(y_min, y_max, n_edge)
    edge_y_right = np.linspace(y_min, y_max, n_edge)
    
    boundary_points = np.vstack([
        np.column_stack([edge_x_top, np.full(n_edge, y_max)]),
        np.column_stack([edge_x_bot, np.full(n_edge, y_min)]),
        np.column_stack([np.full(n_edge, x_min), edge_y_left]),
        np.column_stack([np.full(n_edge, x_max), edge_y_right]),
    ])
    
    points = np.vstack([points_coarse, points_fine, points_rings, boundary_points])
    
    # Remove duplicates and points outside domain
    points = np.unique(np.round(points, decimals=5), axis=0)
    mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
           (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    points = points[mask]
    
    # Compute height field (f_cut - the ablation shape)
    x = points[:, 0]
    y = points[:, 1]
    
    s1_x = sigmoid((x + cut_border_x) / sigma)
    s2_x = sigmoid((cut_border_x - x) / sigma)
    inside_x = s1_x * s2_x
    
    s1_y = sigmoid((y + cut_border_y) / sigma)
    s2_y = sigmoid((cut_border_y - y) / sigma)
    inside_y = s1_y * s2_y
    
    f_cut = 1 - inside_x * inside_y
    
    return points, f_cut


def main():
    print("Creating laser ablation surface mesh...")
    
    # Generate surface data
    points, h = create_laser_ablation_surface()
    x = points[:, 0]
    y = points[:, 1]
    
    # Create Delaunay triangulation
    tri = Delaunay(points)
    triangulation = Triangulation(x, y, tri.simplices)
    
    # Set up the figure with dark background
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 10), facecolor='black')
    
    # 3D view
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
    # Plot triangulated surface - blue edges with copper colormap
    surf = ax.plot_trisurf(triangulation, h, cmap='copper', 
                            edgecolor='dodgerblue', linewidth=0.05, alpha=0.95,
                            antialiased=True)
    
    # Remove axis for cleaner look
    ax.set_axis_off()
    ax.set_title('Laser Ablation Cut - Triangulated Mesh', color='white', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # View angle
    ax.view_init(elev=35, azim=-45)
    
    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Height (h)', color='white', fontsize=11)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    # Add annotation
    ax.text2D(0.02, 0.98, f'Triangles: {len(tri.simplices)}\nVertices: {len(x)}', 
              transform=ax.transAxes, color='white', fontsize=10,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig('laser_ablation_mesh.pdf', dpi=150, facecolor='black', 
                bbox_inches='tight', pad_inches=0.2)
    print("Saved: laser_ablation_mesh.pdf")
    
    # Also create a top-down 2D view showing the mesh
    fig2, ax2 = plt.subplots(figsize=(12, 8), facecolor='black')
    ax2.set_facecolor('black')
    
    # Plot triangulated mesh with height coloring - blue edges
    tripcolor = ax2.tripcolor(triangulation, h, cmap='copper', 
                               edgecolors='dodgerblue', linewidth=0.08)
    
    # Remove axis for cleaner look
    ax2.set_axis_off()
    ax2.set_title('Laser Ablation Cut - Top View', color='white', 
                  fontsize=16, fontweight='bold')
    ax2.set_aspect('equal')
    
    # Colorbar
    cbar2 = fig2.colorbar(tripcolor, ax=ax2, shrink=0.8)
    cbar2.set_label('Height (h)', color='white', fontsize=11)
    cbar2.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color='white')
    
    plt.tight_layout()
    plt.savefig('laser_ablation_mesh_2d.pdf', dpi=150, facecolor='black',
                bbox_inches='tight', pad_inches=0.2)
    print("Saved: laser_ablation_mesh_2d.pdf")
    
    plt.show()


if __name__ == '__main__':
    main()
