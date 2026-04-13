"""
Random Search Visualization

Shows random search on a 2D loss landscape (weight, bias) for linear regression.
Displays both accepted (improved) and failed (rejected) trial points.
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
    
    w_true = 2.0
    b_true = 1.0
    
    n_points = 30
    x_data = np.linspace(-2, 2, n_points)
    noise = np.random.randn(n_points) * 0.5
    y_data = w_true * x_data + b_true + noise
    
    # =========================================================================
    # Loss function
    # =========================================================================
    def mse_loss(w, b):
        y_pred = w * x_data + b
        return np.mean((y_pred - y_data) ** 2)
    
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
    # Random Search - track all trials (accepted + rejected)
    # =========================================================================
    np.random.seed(456)
    w_rs, b_rs = -0.5, 3.5
    n_steps = 40
    
    # History of the accepted path
    history_w = [w_rs]
    history_b = [b_rs]
    history_loss = [mse_loss(w_rs, b_rs)]
    
    # Per-step trial info (the point tried at each step)
    trial_w = []
    trial_b = []
    trial_loss = []
    trial_accepted = []
    
    for step in range(n_steps):
        current_loss = mse_loss(w_rs, b_rs)
        
        dw = np.random.randn() * 0.5
        db = np.random.randn() * 0.5
        new_w = w_rs + dw
        new_b = b_rs + db
        new_loss = mse_loss(new_w, new_b)
        
        trial_w.append(new_w)
        trial_b.append(new_b)
        trial_loss.append(new_loss)
        
        if new_loss < current_loss:
            w_rs, b_rs = new_w, new_b
            trial_accepted.append(True)
        else:
            trial_accepted.append(False)
        
        history_w.append(w_rs)
        history_b.append(b_rs)
        history_loss.append(mse_loss(w_rs, b_rs))
    
    history_w = np.array(history_w)
    history_b = np.array(history_b)
    history_loss = np.array(history_loss)
    trial_w = np.array(trial_w)
    trial_b = np.array(trial_b)
    trial_loss = np.array(trial_loss)
    trial_accepted = np.array(trial_accepted)
    
    n_accepted = trial_accepted.sum()
    n_rejected = (~trial_accepted).sum()
    
    print("=== Random Search ===")
    print(f"Initial: w={history_w[0]:.3f}, b={history_b[0]:.3f}, loss={history_loss[0]:.4f}")
    print(f"Final:   w={history_w[-1]:.3f}, b={history_b[-1]:.3f}, loss={history_loss[-1]:.4f}")
    print(f"True:    w={w_true:.3f}, b={b_true:.3f}")
    print(f"Accepted: {n_accepted}/{n_steps}, Rejected: {n_rejected}/{n_steps}")
    
    # =========================================================================
    # Create animation
    # =========================================================================
    plt.style.use('dark_background')
    
    fig = plt.figure(figsize=(16, 7), facecolor='black')
    
    # Left: 3D loss landscape
    ax3d = fig.add_subplot(121, projection='3d', facecolor='black')
    ax3d.set_xlabel('Weight (w)', color='white', fontsize=10)
    ax3d.set_ylabel('Bias (b)', color='white', fontsize=10)
    ax3d.set_zlabel('Loss', color='white', fontsize=10)
    ax3d.set_title('Random Search on Loss Landscape', color='#4ECDC4', fontsize=14, fontweight='bold')
    
    ax3d.plot_surface(W, B, Z, cmap='viridis', alpha=0.4, linewidth=0, antialiased=True, zorder=1)
    ax3d.contour(W, B, Z, zdir='z', offset=Z.min(), cmap='viridis', alpha=0.4, levels=15)
    ax3d.view_init(elev=45, azim=-60)
    ax3d.set_xlim(w_range.min(), w_range.max())
    ax3d.set_ylim(b_range.min(), b_range.max())
    ax3d.set_zlim(Z.min(), Z.max() * 0.8)
    ax3d.tick_params(colors='white', labelsize=8)
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    
    # Right: Line fitting
    ax_fit = fig.add_subplot(122, facecolor='black')
    ax_fit.set_xlabel('x', color='white', fontsize=11)
    ax_fit.set_ylabel('y', color='white', fontsize=11)
    ax_fit.set_title('Line Fitting', color='white', fontsize=14, fontweight='bold')
    ax_fit.set_xlim(x_data.min() - 0.5, x_data.max() + 0.5)
    ax_fit.set_ylim(y_data.min() - 1.5, y_data.max() + 1.5)
    ax_fit.grid(True, alpha=0.2, color='gray')
    ax_fit.tick_params(colors='white')
    for spine in ax_fit.spines.values():
        spine.set_color('gray')
    
    ax_fit.scatter(x_data, y_data, color='#00BFFF', s=50, alpha=0.8, label='Data', zorder=5)
    
    x_line = np.linspace(x_data.min() - 0.5, x_data.max() + 0.5, 100)
    
    fit_line, = ax_fit.plot([], [], color='#4ECDC4', linewidth=2.5, label='RS fit')
    trial_line, = ax_fit.plot([], [], color='#FF6B6B', linewidth=1.5, alpha=0.5, 
                               linestyle='--', label='Trial (rejected)')
    
    ax_fit.legend(loc='upper left', facecolor='black', edgecolor='gray', labelcolor='white')
    
    # 3D path and points
    z_offset = 1.5
    
    path_3d, = ax3d.plot([], [], [], color='#4ECDC4', linewidth=3, marker='o',
                          markersize=6, markerfacecolor='white', markeredgecolor='#4ECDC4', zorder=100)
    current_3d = ax3d.scatter([], [], [], color='#4ECDC4', s=200, marker='o',
                               edgecolors='white', linewidths=2, zorder=100)
    
    # We'll collect failed/accepted scatter data for cumulative display
    failed_scatter = ax3d.scatter([], [], [], color='#FF6B6B', s=60, marker='x',
                                   linewidths=1.5, alpha=0.6, zorder=90)
    accepted_scatter = ax3d.scatter([], [], [], color='#00FF00', s=60, marker='o',
                                     alpha=0.5, zorder=90)
    
    # Dashed line from current pos to trial
    trial_link, = ax3d.plot([], [], [], color='#FF6B6B', linewidth=1, linestyle='--', alpha=0.6, zorder=95)
    
    step_text = fig.text(0.5, 0.97, '', ha='center', va='top', fontsize=13, color='white', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Cumulative arrays for scatter
    cum_fail_w, cum_fail_b, cum_fail_l = [], [], []
    cum_acc_w, cum_acc_b, cum_acc_l = [], [], []
    
    def animate(frame):
        # Update accepted path
        path_3d.set_data(history_w[:frame+1], history_b[:frame+1])
        path_3d.set_3d_properties(history_loss[:frame+1] + z_offset)
        current_3d._offsets3d = ([history_w[frame]], [history_b[frame]], 
                                 [history_loss[frame] + z_offset])
        
        # Current fit line
        fit_line.set_data(x_line, history_w[frame] * x_line + history_b[frame])
        
        # Show trial point for this step
        if frame > 0 and frame - 1 < len(trial_w):
            step_idx = frame - 1
            tw, tb, tl = trial_w[step_idx], trial_b[step_idx], trial_loss[step_idx]
            accepted = trial_accepted[step_idx]
            
            # Accumulate scatter points
            if accepted:
                cum_acc_w.append(tw)
                cum_acc_b.append(tb)
                cum_acc_l.append(tl + z_offset)
            else:
                cum_fail_w.append(tw)
                cum_fail_b.append(tb)
                cum_fail_l.append(tl + z_offset)
            
            # Update cumulative scatters
            if cum_fail_w:
                failed_scatter._offsets3d = (cum_fail_w, cum_fail_b, cum_fail_l)
            if cum_acc_w:
                accepted_scatter._offsets3d = (cum_acc_w, cum_acc_b, cum_acc_l)
            
            # Dashed line from previous position to trial point
            prev_w, prev_b = history_w[frame-1], history_b[frame-1]
            prev_l = history_loss[frame-1]
            trial_link.set_data([prev_w, tw], [prev_b, tb])
            trial_link.set_3d_properties([prev_l + z_offset, tl + z_offset])
            
            # Show trial line on fit plot if rejected
            if not accepted:
                trial_line.set_data(x_line, tw * x_line + tb)
            else:
                trial_line.set_data([], [])
        else:
            trial_link.set_data([], [])
            trial_link.set_3d_properties([])
            trial_line.set_data([], [])
        
        # Count accepted/rejected so far
        so_far = min(frame, len(trial_accepted))
        n_acc = trial_accepted[:so_far].sum() if so_far > 0 else 0
        n_rej = so_far - n_acc
        
        status = ''
        if frame > 0 and frame - 1 < len(trial_accepted):
            status = '  ACCEPTED ✓' if trial_accepted[frame-1] else '  REJECTED ✗'
        
        step_text.set_text(
            f'Step {frame}/{n_steps}  |  '
            f'Loss: {history_loss[frame]:.4f}  |  '
            f'Accepted: {n_acc}  Rejected: {n_rej}'
            f'{status}'
        )
        
        return []
    
    anim = FuncAnimation(fig, animate, frames=len(history_w), interval=700, blit=False)
    
    print("Saving animation...")
    writer = FFMpegWriter(fps=2, metadata=dict(artist='PINN'), bitrate=3000)
    anim.save('random_search_visualization.mp4', writer=writer, dpi=100)
    print("Saved: random_search_visualization.mp4")
    
    plt.close()

if __name__ == '__main__':
    main()
