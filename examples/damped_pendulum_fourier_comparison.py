"""
Damped Pendulum: Basic Neural Network vs Random Fourier Features

This example demonstrates the difference between a basic neural network and 
one with Random Fourier Features for fitting the damped pendulum dynamics.

The damped pendulum (small angle approximation) is governed by:
    d²θ/dt² + β(dθ/dt) + ω²θ = 0

Analytical solution (underdamped case):
    θ(t) = A·e^(-β/2·t)·cos(ω_d·t + φ)
    where ω_d = sqrt(ω² - β²/4)

Network architecture:
    Basic NN: Input (1) -> Hidden1 (16) -> Hidden2 (16) -> Output (1)
    Fourier NN: Input (1) -> FourierFeatures (32) -> Hidden1 (16) -> Hidden2 (16) -> Output (1)
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# -----------------------------------------------------------------------------
# 1. Fourier Features Encoding
# -----------------------------------------------------------------------------

class FourierFeatures:
    """Random Fourier Feature encoding for high-frequency learning."""
    
    def __init__(self, input_dim, n_features, sigma=1.0, seed=0):
        self.input_dim = input_dim
        self.n_features = n_features
        self.sigma = sigma
        self.output_dim = 2 * n_features
        
        # Generate random projection matrix
        torch.manual_seed(seed)
        self.B = torch.randn(n_features, input_dim) * sigma
    
    def __call__(self, x):
        if self.B.device != x.device:
            self.B = self.B.to(x.device)
        
        Bx = x @ self.B.T
        cos_features = torch.cos(2 * np.pi * Bx)
        sin_features = torch.sin(2 * np.pi * Bx)
        return torch.cat([cos_features, sin_features], dim=-1)


# -----------------------------------------------------------------------------
# 2. Define Neural Networks
# -----------------------------------------------------------------------------

class BasicNet(nn.Module):
    """Basic 2-hidden-layer network without Fourier features."""
    
    def __init__(self, hidden_size=16):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.layer1 = nn.Linear(1, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in [self.layer1, self.layer2, self.layer3]:
            nn.init.xavier_normal_(layer.weight, gain=0.5)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.layer3(x)
        return x


class FourierNet(nn.Module):
    """Neural network with Random Fourier Features encoding."""
    
    def __init__(self, hidden_size=16, n_fourier=16, sigma=5.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_fourier = n_fourier
        
        # Fourier encoding
        self.fourier = FourierFeatures(1, n_fourier, sigma=sigma)
        
        # Network after Fourier encoding
        self.layer1 = nn.Linear(2 * n_fourier, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in [self.layer1, self.layer2, self.layer3]:
            nn.init.xavier_normal_(layer.weight, gain=0.5)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        x = self.fourier(x)  # Apply Fourier features
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.layer3(x)
        return x


# -----------------------------------------------------------------------------
# 3. Damped Pendulum Analytical Solution
# -----------------------------------------------------------------------------

def damped_pendulum_analytical(t, theta0=1.0, omega=5.0, beta=0.5):
    """
    Analytical solution for damped pendulum (small angle approximation).
    
    θ(t) = θ₀ · e^(-β/2·t) · cos(ω_d·t)
    where ω_d = sqrt(ω² - β²/4)
    """
    omega_d = np.sqrt(omega**2 - (beta/2)**2)  # Damped frequency
    decay = np.exp(-beta/2 * t)
    oscillation = np.cos(omega_d * t)
    return theta0 * decay * oscillation


def generate_data(n_points=100, noise_std=0.05, t_max=10.0, 
                  theta0=1.0, omega=5.0, beta=0.5):
    """Generate noisy damped pendulum data."""
    t = np.linspace(0, t_max, n_points).reshape(-1, 1)
    theta_clean = damped_pendulum_analytical(t, theta0, omega, beta)
    noise = np.random.randn(*theta_clean.shape) * noise_std
    theta_noisy = theta_clean + noise
    return t, theta_noisy, theta_clean


# -----------------------------------------------------------------------------
# 4. Training Loop
# -----------------------------------------------------------------------------

def train_and_record(model, x_train, y_train, epochs=500, lr=0.01, model_name="model"):
    """Train the network and record weights at each epoch."""
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = {
        'loss': [],
        'layer1_weights': [],
        'layer2_weights': [],
        'layer3_weights': [],
        'predictions': []
    }
    
    for epoch in range(epochs):
        y_pred = model(x_tensor)
        loss = criterion(y_pred, y_tensor)
        
        history['loss'].append(loss.item())
        history['layer1_weights'].append(model.layer1.weight.detach().clone().numpy())
        history['layer2_weights'].append(model.layer2.weight.detach().clone().numpy())
        history['layer3_weights'].append(model.layer3.weight.detach().clone().numpy())
        history['predictions'].append(y_pred.detach().clone().numpy())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"  {model_name} - Epoch {epoch:4d} | Loss: {loss.item():.6f}")
    
    return history


# -----------------------------------------------------------------------------
# 5. Create Animation
# -----------------------------------------------------------------------------

def create_comparison_animation(t_train, theta_train, theta_analytical,
                                 history_basic, history_fourier,
                                 save_path='pendulum_fourier_comparison.mp4', fps=30):
    """Create animation comparing basic NN vs Fourier NN."""
    
    n_epochs = len(history_basic['loss'])
    hidden_size_basic = history_basic['layer1_weights'][0].shape[0]
    hidden_size_fourier = history_fourier['layer1_weights'][0].shape[0]
    
    # Calculate weight ranges
    all_w_basic = np.concatenate([
        np.array(history_basic['layer1_weights']).flatten(),
        np.array(history_basic['layer2_weights']).flatten(),
        np.array(history_basic['layer3_weights']).flatten()
    ])
    all_w_fourier = np.concatenate([
        np.array(history_fourier['layer1_weights']).flatten(),
        np.array(history_fourier['layer2_weights']).flatten(),
        np.array(history_fourier['layer3_weights']).flatten()
    ])
    w_max = max(abs(all_w_basic).max(), abs(all_w_fourier).max())
    print(f"DEBUG: w_max = {w_max:.4f}")
    print(f"DEBUG: weight range basic = [{all_w_basic.min():.4f}, {all_w_basic.max():.4f}]")
    print(f"DEBUG: weight range fourier = [{all_w_fourier.min():.4f}, {all_w_fourier.max():.4f}]")
    
    # Use a tighter normalization range (±1.0) to show more color variation
    color_range = min(w_max, 1.0)
    
    # Set dark style
    plt.style.use('dark_background')
    
    # Create figure - wide aspect ratio for slides (title will be above)
    fig = plt.figure(figsize=(22, 10), facecolor='black')
    gs = GridSpec(2, 3, figure=fig, height_ratios=[2.5, 1], width_ratios=[1.2, 1, 1],
                  hspace=0.35, wspace=0.2)
    
    # =========================================================================
    # Subplot 1: Curve Fitting Comparison (left, large)
    # =========================================================================
    ax_curve = fig.add_subplot(gs[0, 0], facecolor='black')
    ax_curve.set_title('Damped Pendulum - Curve Fitting', fontsize=14, color='white', fontweight='bold')
    ax_curve.set_xlabel('Time (t)', color='white')
    ax_curve.set_ylabel('θ (angle)', color='white')
    ax_curve.set_xlim(t_train.min() - 0.2, t_train.max() + 0.2)
    ax_curve.set_ylim(theta_train.min() - 0.3, theta_train.max() + 0.3)
    ax_curve.grid(True, alpha=0.2, color='gray')
    ax_curve.tick_params(colors='white')
    for spine in ax_curve.spines.values():
        spine.set_color('gray')
    
    # Plot analytical solution (dashed)
    ax_curve.plot(t_train, theta_analytical, '--', color='#FFFFFF', linewidth=2, 
                  label='Analytical', alpha=0.8)
    # Plot noisy data
    ax_curve.scatter(t_train, theta_train, color='#00BFFF', s=20, alpha=0.5, label='Noisy Data')
    # Network predictions
    pred_basic, = ax_curve.plot([], [], color='#FF6B6B', linewidth=2.5, label='Basic NN')
    pred_fourier, = ax_curve.plot([], [], color='#2ECC71', linewidth=2.5, label='Fourier NN')
    ax_curve.legend(loc='upper right', facecolor='black', edgecolor='gray', labelcolor='white')
    
    # =========================================================================
    # Subplot 2: Basic NN Network Diagram (top middle)
    # =========================================================================
    ax_basic = fig.add_subplot(gs[0, 1], facecolor='black')
    ax_basic.set_title('Basic Neural Network', fontsize=12, color='#FF6B6B', fontweight='bold')
    ax_basic.set_xlim(-0.5, 9.5)
    ax_basic.set_ylim(-1.5, hidden_size_basic + 1)
    ax_basic.axis('off')
    
    # Node positions for basic network
    layer_spacing = 3
    input_pos_b = [(0, hidden_size_basic / 2)]
    hidden1_pos_b = [(layer_spacing, i + 0.5) for i in range(hidden_size_basic)]
    hidden2_pos_b = [(2 * layer_spacing, i + 0.5) for i in range(hidden_size_basic)]
    output_pos_b = [(3 * layer_spacing, hidden_size_basic / 2)]
    
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(vmin=-color_range, vmax=color_range)
    
    # ===== Basic Network - Use LineCollection for efficient color updates =====
    # Layer 1 segments: input -> hidden1
    basic_l1_segments = [[(input_pos_b[0][0], input_pos_b[0][1]), (h_pos[0], h_pos[1])] 
                         for h_pos in hidden1_pos_b]
    basic_lc1 = LineCollection(basic_l1_segments, linewidths=2, cmap=cmap, norm=norm)
    basic_lc1.set_array(np.zeros(len(basic_l1_segments)))
    ax_basic.add_collection(basic_lc1)
    
    # Layer 2 segments: hidden1 -> hidden2
    basic_l2_segments = []
    basic_l2_indices = []  # (i, j) indices for weight lookup
    for i, h1_pos in enumerate(hidden1_pos_b):
        for j, h2_pos in enumerate(hidden2_pos_b):
            basic_l2_segments.append([(h1_pos[0], h1_pos[1]), (h2_pos[0], h2_pos[1])])
            basic_l2_indices.append((i, j))
    basic_lc2 = LineCollection(basic_l2_segments, linewidths=1, cmap=cmap, norm=norm, alpha=0.8)
    basic_lc2.set_array(np.zeros(len(basic_l2_segments)))
    ax_basic.add_collection(basic_lc2)
    
    # Layer 3 segments: hidden2 -> output
    basic_l3_segments = [[(h_pos[0], h_pos[1]), (output_pos_b[0][0], output_pos_b[0][1])] 
                         for h_pos in hidden2_pos_b]
    basic_lc3 = LineCollection(basic_l3_segments, linewidths=2, cmap=cmap, norm=norm)
    basic_lc3.set_array(np.zeros(len(basic_l3_segments)))
    ax_basic.add_collection(basic_lc3)
    
    # Draw nodes on top
    node_size = 250
    ax_basic.scatter([p[0] for p in input_pos_b], [p[1] for p in input_pos_b], 
                     s=node_size, c='#4ECDC4', edgecolors='white', linewidths=2, zorder=5)
    ax_basic.scatter([p[0] for p in hidden1_pos_b], [p[1] for p in hidden1_pos_b], 
                     s=node_size, c='#9B59B6', edgecolors='white', linewidths=2, zorder=5)
    ax_basic.scatter([p[0] for p in hidden2_pos_b], [p[1] for p in hidden2_pos_b], 
                     s=node_size, c='#9B59B6', edgecolors='white', linewidths=2, zorder=5)
    ax_basic.scatter([p[0] for p in output_pos_b], [p[1] for p in output_pos_b], 
                     s=node_size, c='#E74C3C', edgecolors='white', linewidths=2, zorder=5)
    
    # Labels
    ax_basic.text(0, -0.5, 'In', ha='center', va='top', color='white', fontsize=8)
    ax_basic.text(layer_spacing, -0.5, 'H1', ha='center', va='top', color='white', fontsize=8)
    ax_basic.text(2*layer_spacing, -0.5, 'H2', ha='center', va='top', color='white', fontsize=8)
    ax_basic.text(3*layer_spacing, -0.5, 'Out', ha='center', va='top', color='white', fontsize=8)
    
    # =========================================================================
    # Subplot 3: Fourier NN Network Diagram (top right)
    # =========================================================================
    ax_fourier = fig.add_subplot(gs[0, 2], facecolor='black')
    ax_fourier.set_title('Fourier Features NN', fontsize=12, color='#2ECC71', fontweight='bold')
    ax_fourier.set_xlim(-0.5, 12.5)
    ax_fourier.set_ylim(-1.5, hidden_size_fourier + 1)
    ax_fourier.axis('off')
    
    # Node positions for Fourier network (extra layer for Fourier features)
    n_fourier_display = 8  # Display fewer Fourier nodes for clarity
    input_pos_f = [(0, hidden_size_fourier / 2)]
    fourier_pos_f = [(layer_spacing, i * (hidden_size_fourier / n_fourier_display) + 0.25) 
                     for i in range(n_fourier_display)]
    hidden1_pos_f = [(2 * layer_spacing, i + 0.5) for i in range(hidden_size_fourier)]
    hidden2_pos_f = [(3 * layer_spacing, i + 0.5) for i in range(hidden_size_fourier)]
    output_pos_f = [(4 * layer_spacing, hidden_size_fourier / 2)]
    
    # Draw edges for Fourier network
    # Input -> Fourier (static, special color - these are the fixed random projections)
    for f_pos in fourier_pos_f:
        ax_fourier.plot([input_pos_f[0][0], f_pos[0]], [input_pos_f[0][1], f_pos[1]], 
                        color='#FFD700', linewidth=1, alpha=0.6)
    
    # ===== Fourier Network - Use LineCollection for efficient color updates =====
    # Layer 1 segments: Fourier -> Hidden1 (trainable)
    fourier_l1_segments = []
    fourier_l1_indices = []
    for i, f_pos in enumerate(fourier_pos_f):
        for j, h_pos in enumerate(hidden1_pos_f):
            fourier_l1_segments.append([(f_pos[0], f_pos[1]), (h_pos[0], h_pos[1])])
            fourier_l1_indices.append((i, j))
    fourier_lc1 = LineCollection(fourier_l1_segments, linewidths=1, cmap=cmap, norm=norm, alpha=0.8)
    fourier_lc1.set_array(np.zeros(len(fourier_l1_segments)))
    ax_fourier.add_collection(fourier_lc1)
    
    # Layer 2 segments: Hidden1 -> Hidden2 (trainable)
    fourier_l2_segments = []
    fourier_l2_indices = []
    for i, h1_pos in enumerate(hidden1_pos_f):
        for j, h2_pos in enumerate(hidden2_pos_f):
            fourier_l2_segments.append([(h1_pos[0], h1_pos[1]), (h2_pos[0], h2_pos[1])])
            fourier_l2_indices.append((i, j))
    fourier_lc2 = LineCollection(fourier_l2_segments, linewidths=1, cmap=cmap, norm=norm, alpha=0.8)
    fourier_lc2.set_array(np.zeros(len(fourier_l2_segments)))
    ax_fourier.add_collection(fourier_lc2)
    
    # Layer 3 segments: Hidden2 -> Output (trainable)
    fourier_l3_segments = [[(h_pos[0], h_pos[1]), (output_pos_f[0][0], output_pos_f[0][1])] 
                           for h_pos in hidden2_pos_f]
    fourier_lc3 = LineCollection(fourier_l3_segments, linewidths=2, cmap=cmap, norm=norm)
    fourier_lc3.set_array(np.zeros(len(fourier_l3_segments)))
    ax_fourier.add_collection(fourier_lc3)
    
    # Draw nodes on top
    ax_fourier.scatter([p[0] for p in input_pos_f], [p[1] for p in input_pos_f], 
                       s=node_size, c='#4ECDC4', edgecolors='white', linewidths=2, zorder=5)
    ax_fourier.scatter([p[0] for p in fourier_pos_f], [p[1] for p in fourier_pos_f], 
                       s=node_size*0.7, c='#FFD700', edgecolors='white', linewidths=1.5, zorder=5,
                       marker='s')  # Square for Fourier
    ax_fourier.scatter([p[0] for p in hidden1_pos_f], [p[1] for p in hidden1_pos_f], 
                       s=node_size, c='#9B59B6', edgecolors='white', linewidths=2, zorder=5)
    ax_fourier.scatter([p[0] for p in hidden2_pos_f], [p[1] for p in hidden2_pos_f], 
                       s=node_size, c='#9B59B6', edgecolors='white', linewidths=2, zorder=5)
    ax_fourier.scatter([p[0] for p in output_pos_f], [p[1] for p in output_pos_f], 
                       s=node_size, c='#E74C3C', edgecolors='white', linewidths=2, zorder=5)
    
    # Labels
    ax_fourier.text(0, -0.5, 'In', ha='center', va='top', color='white', fontsize=8)
    ax_fourier.text(layer_spacing, -0.5, 'Fourier', ha='center', va='top', color='#FFD700', fontsize=8)
    ax_fourier.text(2*layer_spacing, -0.5, 'H1', ha='center', va='top', color='white', fontsize=8)
    ax_fourier.text(3*layer_spacing, -0.5, 'H2', ha='center', va='top', color='white', fontsize=8)
    ax_fourier.text(4*layer_spacing, -0.5, 'Out', ha='center', va='top', color='white', fontsize=8)
    
    # =========================================================================
    # Subplot 4: Loss Curves (bottom, spanning all)
    # =========================================================================
    ax_loss = fig.add_subplot(gs[1, :], facecolor='black')
    ax_loss.set_title('Training Loss Comparison', fontsize=14, color='white', fontweight='bold')
    ax_loss.set_xlabel('Epoch', color='white')
    ax_loss.set_ylabel('MSE Loss', color='white')
    ax_loss.set_xlim(0, n_epochs)
    min_loss = min(min(history_basic['loss']), min(history_fourier['loss']))
    max_loss = max(max(history_basic['loss']), max(history_fourier['loss']))
    ax_loss.set_ylim(min_loss * 0.5, max_loss * 1.2)
    ax_loss.set_yscale('log')
    ax_loss.grid(True, alpha=0.2, color='gray')
    ax_loss.tick_params(colors='white')
    for spine in ax_loss.spines.values():
        spine.set_color('gray')
    
    loss_basic_line, = ax_loss.plot([], [], color='#FF6B6B', linewidth=2, label='Basic NN')
    loss_fourier_line, = ax_loss.plot([], [], color='#2ECC71', linewidth=2, label='Fourier NN')
    loss_basic_point, = ax_loss.plot([], [], 'o', color='#FF6B6B', markersize=8)
    loss_fourier_point, = ax_loss.plot([], [], 'o', color='#2ECC71', markersize=8)
    ax_loss.legend(loc='upper right', facecolor='black', edgecolor='gray', labelcolor='white')
    
    # Epoch counter (small, at bottom since slide has title above)
    epoch_text = fig.suptitle('', fontsize=14, fontweight='bold', color='white', y=0.98)
    
    plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.96])
    
    def init():
        pred_basic.set_data([], [])
        pred_fourier.set_data([], [])
        loss_basic_line.set_data([], [])
        loss_fourier_line.set_data([], [])
        loss_basic_point.set_data([], [])
        loss_fourier_point.set_data([], [])
        return []
    
    def animate(frame):
        # Update epoch text
        epoch_text.set_text(f'Epoch {frame} / {n_epochs-1}  |  Basic Loss: {history_basic["loss"][frame]:.6f}  |  Fourier Loss: {history_fourier["loss"][frame]:.6f}')
        
        # Update predictions
        pred_basic.set_data(t_train.flatten(), history_basic['predictions'][frame].flatten())
        pred_fourier.set_data(t_train.flatten(), history_fourier['predictions'][frame].flatten())
        
        # ===== Update Basic NN edges using LineCollection =====
        # Layer 1: input -> hidden1
        w1_b = history_basic['layer1_weights'][frame]
        weights_l1_b = np.array([w1_b[j, 0] for j in range(hidden_size_basic)])
        basic_lc1.set_array(weights_l1_b)
        basic_lc1.set_linewidths(1 + 3 * np.abs(weights_l1_b) / w_max)
        
        # Layer 2: hidden1 -> hidden2
        w2_b = history_basic['layer2_weights'][frame]
        weights_l2_b = np.array([w2_b[j, i] for i, j in basic_l2_indices])
        basic_lc2.set_array(weights_l2_b)
        basic_lc2.set_linewidths(0.5 + 1.5 * np.abs(weights_l2_b) / w_max)
        
        # Layer 3: hidden2 -> output
        w3_b = history_basic['layer3_weights'][frame]
        weights_l3_b = np.array([w3_b[0, j] for j in range(hidden_size_basic)])
        basic_lc3.set_array(weights_l3_b)
        basic_lc3.set_linewidths(1 + 3 * np.abs(weights_l3_b) / w_max)
        
        # ===== Update Fourier NN edges using LineCollection =====
        # Layer 1: Fourier -> Hidden1
        w1_f = history_fourier['layer1_weights'][frame]
        weights_l1_f = []
        for i, j in fourier_l1_indices:
            weight_idx = i * (w1_f.shape[1] // n_fourier_display) if n_fourier_display < w1_f.shape[1] else i
            if weight_idx < w1_f.shape[1]:
                weights_l1_f.append(w1_f[j, weight_idx])
            else:
                weights_l1_f.append(w1_f[j, 0])
        weights_l1_f = np.array(weights_l1_f)
        fourier_lc1.set_array(weights_l1_f)
        fourier_lc1.set_linewidths(0.5 + 1.5 * np.abs(weights_l1_f) / w_max)
        
        # Layer 2: Hidden1 -> Hidden2
        w2_f = history_fourier['layer2_weights'][frame]
        weights_l2_f = np.array([w2_f[j, i] for i, j in fourier_l2_indices])
        fourier_lc2.set_array(weights_l2_f)
        fourier_lc2.set_linewidths(0.5 + 1.5 * np.abs(weights_l2_f) / w_max)
        
        # Layer 3: Hidden2 -> Output
        w3_f = history_fourier['layer3_weights'][frame]
        weights_l3_f = np.array([w3_f[0, j] for j in range(hidden_size_fourier)])
        fourier_lc3.set_array(weights_l3_f)
        fourier_lc3.set_linewidths(1 + 3 * np.abs(weights_l3_f) / w_max)
        
        # Update loss curves
        loss_basic_line.set_data(range(frame + 1), history_basic['loss'][:frame + 1])
        loss_fourier_line.set_data(range(frame + 1), history_fourier['loss'][:frame + 1])
        loss_basic_point.set_data([frame], [history_basic['loss'][frame]])
        loss_fourier_point.set_data([frame], [history_fourier['loss'][frame]])
        
        return []
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=n_epochs, interval=1000/fps, blit=False
    )
    
    # Save as video
    print(f"\nSaving animation to {save_path}...")
    writer = animation.FFMpegWriter(fps=fps, metadata={'title': 'Fourier Features Comparison'})
    anim.save(save_path, writer=writer, dpi=150)
    print(f"Animation saved to {save_path}")
    
    plt.close()
    plt.style.use('default')
    
    return anim


# -----------------------------------------------------------------------------
# 6. Main Execution
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Parameters
    HIDDEN_SIZE = 16     # Neurons per hidden layer
    N_FOURIER = 16       # Number of Fourier features
    FOURIER_SIGMA = 5.0  # Fourier features sigma (higher = more high-freq)
    EPOCHS = 500
    LEARNING_RATE = 0.01
    NOISE_STD = 0.05
    
    # Pendulum parameters
    THETA0 = 1.0    # Initial angle
    OMEGA = 5.0     # Angular frequency
    BETA = 0.5      # Damping coefficient
    T_MAX = 10.0    # Simulation time
    
    print("="*70)
    print("Damped Pendulum: Basic NN vs Random Fourier Features")
    print("="*70)
    
    print(f"\nPendulum parameters: θ₀={THETA0}, ω={OMEGA}, β={BETA}")
    print(f"Network: Hidden layers with {HIDDEN_SIZE} neurons each")
    print(f"Fourier features: {N_FOURIER} features, σ={FOURIER_SIGMA}")
    print(f"Training: {EPOCHS} epochs, lr={LEARNING_RATE}")
    
    # Generate data
    print("\n1. Generating damped pendulum data...")
    t_train, theta_train, theta_analytical = generate_data(
        n_points=150, noise_std=NOISE_STD, t_max=T_MAX,
        theta0=THETA0, omega=OMEGA, beta=BETA
    )
    print(f"   Data points: {len(t_train)}")
    print(f"   Noise std: {NOISE_STD}")
    
    # Create models
    print("\n2. Creating networks...")
    model_basic = BasicNet(hidden_size=HIDDEN_SIZE)
    model_fourier = FourierNet(hidden_size=HIDDEN_SIZE, n_fourier=N_FOURIER, sigma=FOURIER_SIGMA)
    
    params_basic = sum(p.numel() for p in model_basic.parameters())
    params_fourier = sum(p.numel() for p in model_fourier.parameters())
    print(f"   Basic NN parameters: {params_basic}")
    print(f"   Fourier NN parameters: {params_fourier}")
    
    # Train both models
    print("\n3. Training Basic Neural Network...")
    history_basic = train_and_record(model_basic, t_train, theta_train, 
                                      epochs=EPOCHS, lr=LEARNING_RATE, 
                                      model_name="Basic NN")
    
    print("\n4. Training Fourier Features Neural Network...")
    history_fourier = train_and_record(model_fourier, t_train, theta_train, 
                                        epochs=EPOCHS, lr=LEARNING_RATE,
                                        model_name="Fourier NN")
    
    print(f"\n   Basic NN final loss: {history_basic['loss'][-1]:.8f}")
    print(f"   Fourier NN final loss: {history_fourier['loss'][-1]:.8f}")
    
    # Create animation
    print("\n5. Creating comparison animation...")
    create_comparison_animation(
        t_train, theta_train, theta_analytical,
        history_basic, history_fourier,
        save_path='pendulum_fourier_comparison.mp4', fps=30
    )
    
    print("\n" + "="*70)
    print("Done! Check 'pendulum_fourier_comparison.mp4' for the video.")
    print("="*70)
