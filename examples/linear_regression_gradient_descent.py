"""
Linear Regression: Gradient Descent vs Random Search

Compares gradient descent (with gradient arrows in 3D) against random search
on a 2D loss landscape (weight, bias). Shows discrete steps with line fitting.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D

def main():
    # =========================================================================
    # Generate noisy line data
    # =========================================================================
    np.random.seed(42)
    
    # True parameters
    w_true = 2.0
    b_true = 1.0
    
    # Generate data
    n_points = 30
    x_data = np.linspace(-2, 2, n_points)
    noise = np.random.randn(n_points) * 0.5
    y_data = w_true * x_data + b_true + noise
    
    # =========================================================================
    # Define loss function and gradient
    # =========================================================================
    def mse_loss(w, b):
        y_pred = w * x_data + b
        return np.mean((y_pred - y_data) ** 2)
    
    def compute_gradient(w, b):
        y_pred = w * x_data + b
        error = y_pred - y_data
        grad_w = 2 * np.mean(error * x_data)
        grad_b = 2 * np.mean(error)
        return grad_w, grad_b
    
    # =========================================================================
    # Compute loss landscape
    # =========================================================================
    w_range = np.linspace(-1, 5, 100)
    b_range = np.linspace(-2, 4, 100)
    W, B = np.meshgrid(w_range, b_range)
    
    Z = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Z[i, j] = mse_loss(W[i, j], B[i, j])
    
    # =========================================================================
    # Method 1: Gradient Descent (fewer steps for clarity)
    # =========================================================================
    w_gd, b_gd = -0.5, 3.5
    learning_rate = 0.15
    n_steps = 25
    
    history_gd_w = [w_gd]
    history_gd_b = [b_gd]
    history_gd_loss = [mse_loss(w_gd, b_gd)]
    history_gd_grad_w = []
    history_gd_grad_b = []
    
    for step in range(n_steps):
        grad_w, grad_b = compute_gradient(w_gd, b_gd)
        history_gd_grad_w.append(grad_w)
        history_gd_grad_b.append(grad_b)
        
        w_gd = w_gd - learning_rate * grad_w
        b_gd = b_gd - learning_rate * grad_b
        
        history_gd_w.append(w_gd)
        history_gd_b.append(b_gd)
        history_gd_loss.append(mse_loss(w_gd, b_gd))
    
    history_gd_w = np.array(history_gd_w)
    history_gd_b = np.array(history_gd_b)
    history_gd_loss = np.array(history_gd_loss)
    history_gd_grad_w = np.array(history_gd_grad_w)
    history_gd_grad_b = np.array(history_gd_grad_b)
    
    # =========================================================================
    # Method 2: Random Search (made worse - fewer trials, smaller steps)
    # =========================================================================
    np.random.seed(456)  # Different seed for more erratic behavior
    w_rs, b_rs = -0.5, 3.5
    
    history_rs_w = [w_rs]
    history_rs_b = [b_rs]
    history_rs_loss = [mse_loss(w_rs, b_rs)]
    
    for step in range(n_steps):
        # Try a single random perturbation and keep if better
        current_loss = mse_loss(w_rs, b_rs)
        
        # Single random step
        dw = np.random.randn() * 0.4
        db = np.random.randn() * 0.4
        new_w = w_rs + dw
        new_b = b_rs + db
        new_loss = mse_loss(new_w, new_b)
        
        # Only update if better
        if new_loss < current_loss:
            w_rs, b_rs = new_w, new_b
        
        history_rs_w.append(w_rs)
        history_rs_b.append(b_rs)
        history_rs_loss.append(mse_loss(w_rs, b_rs))
    
    history_rs_w = np.array(history_rs_w)
    history_rs_b = np.array(history_rs_b)
    history_rs_loss = np.array(history_rs_loss)
    
    print("=== Gradient Descent ===")
    print(f"Initial: w={history_gd_w[0]:.3f}, b={history_gd_b[0]:.3f}, loss={history_gd_loss[0]:.4f}")
    print(f"Final:   w={history_gd_w[-1]:.3f}, b={history_gd_b[-1]:.3f}, loss={history_gd_loss[-1]:.4f}")
    print("\n=== Random Search ===")
    print(f"Initial: w={history_rs_w[0]:.3f}, b={history_rs_b[0]:.3f}, loss={history_rs_loss[0]:.4f}")
    print(f"Final:   w={history_rs_w[-1]:.3f}, b={history_rs_b[-1]:.3f}, loss={history_rs_loss[-1]:.4f}")
    print(f"\nTrue:    w={w_true:.3f}, b={b_true:.3f}")
    
    # =========================================================================
    # Create animation
    # =========================================================================
    plt.style.use('dark_background')
    
    fig = plt.figure(figsize=(18, 8), facecolor='black')
    
    # Left: Gradient Descent 3D
    ax_gd = fig.add_subplot(131, projection='3d', facecolor='black')
    ax_gd.set_xlabel('Weight (w)', color='white', fontsize=10)
    ax_gd.set_ylabel('Bias (b)', color='white', fontsize=10)
    ax_gd.set_zlabel('Loss', color='white', fontsize=10)
    ax_gd.set_title('Gradient Descent', color='#FF6B6B', fontsize=14, fontweight='bold')
    
    # Plot surface
    ax_gd.plot_surface(W, B, Z, cmap='viridis', alpha=0.4, linewidth=0, antialiased=True, zorder=1)
    ax_gd.contour(W, B, Z, zdir='z', offset=Z.min(), cmap='viridis', alpha=0.4, levels=15)
    ax_gd.scatter([w_true], [b_true], [mse_loss(w_true, b_true)], 
                  color='#00FF00', s=150, marker='*', zorder=10)
    
    ax_gd.view_init(elev=45, azim=-60)
    ax_gd.set_xlim(w_range.min(), w_range.max())
    ax_gd.set_ylim(b_range.min(), b_range.max())
    ax_gd.set_zlim(Z.min(), Z.max() * 0.8)
    ax_gd.tick_params(colors='white', labelsize=8)
    ax_gd.xaxis.pane.fill = False
    ax_gd.yaxis.pane.fill = False
    ax_gd.zaxis.pane.fill = False
    
    # Middle: Random Search 3D
    ax_rs = fig.add_subplot(132, projection='3d', facecolor='black')
    ax_rs.set_xlabel('Weight (w)', color='white', fontsize=10)
    ax_rs.set_ylabel('Bias (b)', color='white', fontsize=10)
    ax_rs.set_zlabel('Loss', color='white', fontsize=10)
    ax_rs.set_title('Random Search', color='#4ECDC4', fontsize=14, fontweight='bold')
    
    ax_rs.plot_surface(W, B, Z, cmap='viridis', alpha=0.4, linewidth=0, antialiased=True, zorder=1)
    ax_rs.contour(W, B, Z, zdir='z', offset=Z.min(), cmap='viridis', alpha=0.4, levels=15)
    ax_rs.scatter([w_true], [b_true], [mse_loss(w_true, b_true)], 
                  color='#00FF00', s=150, marker='*', zorder=10)
    
    ax_rs.view_init(elev=45, azim=-60)
    ax_rs.set_xlim(w_range.min(), w_range.max())
    ax_rs.set_ylim(b_range.min(), b_range.max())
    ax_rs.set_zlim(Z.min(), Z.max() * 0.8)
    ax_rs.tick_params(colors='white', labelsize=8)
    ax_rs.xaxis.pane.fill = False
    ax_rs.yaxis.pane.fill = False
    ax_rs.zaxis.pane.fill = False
    
    # Right: Line fitting visualization
    ax_fit = fig.add_subplot(133, facecolor='black')
    ax_fit.set_xlabel('x', color='white', fontsize=11)
    ax_fit.set_ylabel('y', color='white', fontsize=11)
    ax_fit.set_title('Line Fitting Comparison', color='white', fontsize=14, fontweight='bold')
    ax_fit.set_xlim(x_data.min() - 0.5, x_data.max() + 0.5)
    ax_fit.set_ylim(y_data.min() - 1.5, y_data.max() + 1.5)
    ax_fit.grid(True, alpha=0.2, color='gray')
    ax_fit.tick_params(colors='white')
    for spine in ax_fit.spines.values():
        spine.set_color('gray')
    
    # Plot data points
    ax_fit.scatter(x_data, y_data, color='#00BFFF', s=50, alpha=0.8, label='Data', zorder=5)
    
    # True line
    x_line = np.linspace(x_data.min() - 0.5, x_data.max() + 0.5, 100)
    ax_fit.plot(x_line, w_true * x_line + b_true, '--', color='#00FF00', 
                linewidth=2, label='True line', alpha=0.7)
    
    # Current fit lines
    fit_line_gd, = ax_fit.plot([], [], color='#FF6B6B', linewidth=2.5, label='GD fit')
    fit_line_rs, = ax_fit.plot([], [], color='#4ECDC4', linewidth=2.5, label='RS fit')
    
    ax_fit.legend(loc='upper left', facecolor='black', edgecolor='gray', labelcolor='white')
    
    # Initialize paths and points
    gd_path_3d, = ax_gd.plot([], [], [], color='#FF6B6B', linewidth=4, marker='o', 
                              markersize=8, markerfacecolor='white', markeredgecolor='#FF6B6B', zorder=100)
    gd_point_3d = ax_gd.scatter([], [], [], color='#FF6B6B', s=200, marker='o', 
                                 edgecolors='white', linewidths=2, zorder=100)
    
    rs_path_3d, = ax_rs.plot([], [], [], color='#4ECDC4', linewidth=4, marker='o',
                              markersize=8, markerfacecolor='white', markeredgecolor='#4ECDC4', zorder=100)
    rs_point_3d = ax_rs.scatter([], [], [], color='#4ECDC4', s=200, marker='o',
                                 edgecolors='white', linewidths=2, zorder=100)
    
    # 3D Gradient arrow (using quiver, will be recreated each frame)
    gradient_quiver = None
    
    # Text
    step_text = fig.text(0.5, 0.02, '', ha='center', fontsize=13, color='white', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    
    # Z offset to lift paths above surface
    z_offset = 1.5
    
    def animate(frame):
        nonlocal gradient_quiver
        
        # Update GD path (with z offset to stay above surface)
        gd_path_3d.set_data(history_gd_w[:frame+1], history_gd_b[:frame+1])
        gd_path_3d.set_3d_properties(history_gd_loss[:frame+1] + z_offset)
        gd_point_3d._offsets3d = ([history_gd_w[frame]], [history_gd_b[frame]], [history_gd_loss[frame] + z_offset])
        
        # Update RS path (with z offset)
        rs_path_3d.set_data(history_rs_w[:frame+1], history_rs_b[:frame+1])
        rs_path_3d.set_3d_properties(history_rs_loss[:frame+1] + z_offset)
        rs_point_3d._offsets3d = ([history_rs_w[frame]], [history_rs_b[frame]], [history_rs_loss[frame] + z_offset])
        
        # Update fit lines
        fit_line_gd.set_data(x_line, history_gd_w[frame] * x_line + history_gd_b[frame])
        fit_line_rs.set_data(x_line, history_rs_w[frame] * x_line + history_rs_b[frame])
        
        # Remove old gradient arrow
        if gradient_quiver is not None:
            gradient_quiver.remove()
            gradient_quiver = None
        
        # Draw gradient arrow in 3D at current GD position
        if frame < len(history_gd_grad_w):
            gw = history_gd_grad_w[frame]
            gb = history_gd_grad_b[frame]
            grad_mag = np.sqrt(gw**2 + gb**2)
            
            if grad_mag > 0.01:
                # Scale gradient for visibility
                scale = 1.5 / grad_mag
                # Arrow points in negative gradient direction (descent)
                dw = -gw * scale
                db = -gb * scale
                # Compute loss change for z component
                current_loss = history_gd_loss[frame]
                next_loss = history_gd_loss[frame + 1] if frame + 1 < len(history_gd_loss) else current_loss
                dz = (next_loss - current_loss) * 0.5
                
                gradient_quiver = ax_gd.quiver(
                    history_gd_w[frame], history_gd_b[frame], history_gd_loss[frame] + z_offset,
                    dw, db, dz,
                    color='#FFD700', arrow_length_ratio=0.3, linewidth=3
                )
        
        # Update text
        step_text.set_text(
            f'Step {frame}/{n_steps}  |  '
            f'GD Loss: {history_gd_loss[frame]:.4f}  |  '
            f'RS Loss: {history_rs_loss[frame]:.4f}'
        )
        
        return []
    
    # Create animation - slower (600ms per frame, 2.5 fps)
    anim = FuncAnimation(fig, animate, frames=len(history_gd_w), interval=600, blit=False)
    
    # Save video - slower fps for clear steps
    print("Saving animation...")
    writer = FFMpegWriter(fps=2, metadata=dict(artist='PINN'), bitrate=3000)
    anim.save('linear_regression_gradient_descent.mp4', writer=writer, dpi=100)
    print("Saved: linear_regression_gradient_descent.mp4")
    
    plt.close()

if __name__ == '__main__':
    main()
