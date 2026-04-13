"""
Simple Curve Fitting Example with Weight Evolution Visualization

This example demonstrates how a small neural network with 2 hidden layers learns 
to fit noisy data, and creates a video showing how the weights evolve during training.

Network architecture:
    Input (1) -> Hidden1 (8 neurons) -> Hidden2 (8 neurons) -> Output (1)
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# -----------------------------------------------------------------------------
# 1. Define a 2-hidden-layer network
# -----------------------------------------------------------------------------

class SimpleNet(nn.Module):
    """A network with 2 hidden layers: Input -> Hidden1 -> Hidden2 -> Output"""
    
    def __init__(self, hidden_size=8):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Layer 1: Input (1) -> Hidden1
        self.layer1 = nn.Linear(1, hidden_size)
        
        # Layer 2: Hidden1 -> Hidden2
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        
        # Layer 3: Hidden2 -> Output (1)
        self.layer3 = nn.Linear(hidden_size, 1)
        
        # Initialize with small weights for better visualization
        nn.init.xavier_normal_(self.layer1.weight, gain=0.5)
        nn.init.zeros_(self.layer1.bias)
        nn.init.xavier_normal_(self.layer2.weight, gain=0.5)
        nn.init.zeros_(self.layer2.bias)
        nn.init.xavier_normal_(self.layer3.weight, gain=0.5)
        nn.init.zeros_(self.layer3.bias)
    
    def forward(self, x):
        x = torch.tanh(self.layer1(x))  # Hidden layer 1
        x = torch.tanh(self.layer2(x))  # Hidden layer 2
        x = self.layer3(x)               # Output layer (linear)
        return x


# -----------------------------------------------------------------------------
# 2. Generate target data (noisy sinusoidal curve)
# -----------------------------------------------------------------------------

def generate_data(n_points=100, noise_std=0.1):
    """Generate a noisy sinusoidal curve to fit"""
    x = np.linspace(-np.pi, np.pi, n_points).reshape(-1, 1)
    y_clean = np.sin(x) + 0.3 * np.sin(3 * x)  # A composite sine wave
    noise = np.random.randn(*y_clean.shape) * noise_std
    y = y_clean + noise
    return x, y, y_clean


# -----------------------------------------------------------------------------
# 3. Training loop with weight recording
# -----------------------------------------------------------------------------

def train_and_record(model, x_train, y_train, epochs=500, lr=0.05):
    """
    Train the network and record weights at each epoch.
    
    Returns:
        history: dict with loss, weights, predictions at each epoch
    """
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Recording storage (now with 3 layers)
    history = {
        'loss': [],
        'layer1_weights': [],
        'layer1_bias': [],
        'layer2_weights': [],
        'layer2_bias': [],
        'layer3_weights': [],
        'layer3_bias': [],
        'predictions': []
    }
    
    for epoch in range(epochs):
        # Forward pass
        y_pred = model(x_tensor)
        loss = criterion(y_pred, y_tensor)
        
        # Record current state
        history['loss'].append(loss.item())
        history['layer1_weights'].append(model.layer1.weight.detach().clone().numpy())
        history['layer1_bias'].append(model.layer1.bias.detach().clone().numpy())
        history['layer2_weights'].append(model.layer2.weight.detach().clone().numpy())
        history['layer2_bias'].append(model.layer2.bias.detach().clone().numpy())
        history['layer3_weights'].append(model.layer3.weight.detach().clone().numpy())
        history['layer3_bias'].append(model.layer3.bias.detach().clone().numpy())
        history['predictions'].append(y_pred.detach().clone().numpy())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")
    
    return history


# -----------------------------------------------------------------------------
# 4. Create animation video
# -----------------------------------------------------------------------------

def create_animation(x_train, y_train, history, x_plot=None, save_path='weights_evolution.mp4', fps=20):
    """Create an animation showing curve fitting and weight evolution with network diagram."""
    
    n_epochs = len(history['loss'])
    hidden_size = history['layer1_weights'][0].shape[0]
    
    # Use x_plot for smooth curve if provided, otherwise use x_train
    if x_plot is None:
        x_plot = x_train
    
    # Calculate weight ranges for consistent color/width scaling
    all_w1 = np.array(history['layer1_weights'])
    all_w2 = np.array(history['layer2_weights'])
    all_w3 = np.array(history['layer3_weights'])
    w_max = max(abs(all_w1).max(), abs(all_w2).max(), abs(all_w3).max())
    
    # Set dark style
    plt.style.use('dark_background')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 9), facecolor='black')
    gs = GridSpec(2, 2, figure=fig, height_ratios=[2, 1], width_ratios=[1, 1],
                  hspace=0.3, wspace=0.3)
    
    # Subplot 1: Curve fitting (top left)
    ax_curve = fig.add_subplot(gs[0, 0], facecolor='black')
    ax_curve.set_title('Curve Fitting Progress', fontsize=14, color='white', fontweight='bold')
    ax_curve.set_xlabel('x', color='white')
    ax_curve.set_ylabel('y', color='white')
    ax_curve.set_xlim(x_train.min() - 0.2, x_train.max() + 0.2)
    ax_curve.set_ylim(y_train.min() - 0.5, y_train.max() + 0.5)
    ax_curve.grid(True, alpha=0.2, color='gray')
    ax_curve.tick_params(colors='white')
    for spine in ax_curve.spines.values():
        spine.set_color('gray')
    
    # Plot noisy target data as scatter points
    ax_curve.scatter(x_train, y_train, color='#00BFFF', s=30, alpha=0.6, label='Noisy Data')
    prediction_line, = ax_curve.plot([], [], color='#FF6B6B', linewidth=2.5, 
                                      label='Network')
    ax_curve.legend(loc='upper right', facecolor='black', edgecolor='gray', 
                    labelcolor='white')
    
    # Subplot 2: Network diagram (top right) - Now with 2 hidden layers
    ax_net = fig.add_subplot(gs[0, 1], facecolor='black')
    ax_net.set_title('Network Weights', fontsize=14, color='white', fontweight='bold')
    ax_net.set_xlim(-1, 10)
    ax_net.set_ylim(-0.5, hidden_size + 0.5)
    ax_net.set_aspect('equal')
    ax_net.axis('off')
    
    # Define node positions for 4 layers (spread out horizontally)
    layer_spacing = 3  # Horizontal spacing between layers
    # Input layer (1 node)
    input_pos = [(0, hidden_size / 2)]
    # Hidden layer 1 (hidden_size nodes)
    hidden1_pos = [(layer_spacing, i + 0.5) for i in range(hidden_size)]
    # Hidden layer 2 (hidden_size nodes)
    hidden2_pos = [(2 * layer_spacing, i + 0.5) for i in range(hidden_size)]
    # Output layer (1 node)
    output_pos = [(3 * layer_spacing, hidden_size / 2)]
    
    # Create colormap for weights (blue = negative, red = positive)
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(vmin=-w_max, vmax=w_max)
    
    # Draw initial edges (will be updated in animation)
    # Layer 1 edges: input -> hidden1
    layer1_lines = []
    for j, h_pos in enumerate(hidden1_pos):
        line, = ax_net.plot([input_pos[0][0], h_pos[0]], [input_pos[0][1], h_pos[1]], 
                           color='gray', linewidth=1, alpha=0.8)
        layer1_lines.append(line)
    
    # Layer 2 edges: hidden1 -> hidden2
    layer2_lines = []
    for i, h1_pos in enumerate(hidden1_pos):
        for j, h2_pos in enumerate(hidden2_pos):
            line, = ax_net.plot([h1_pos[0], h2_pos[0]], [h1_pos[1], h2_pos[1]], 
                               color='gray', linewidth=1, alpha=0.5)
            layer2_lines.append((i, j, line))
    
    # Layer 3 edges: hidden2 -> output
    layer3_lines = []
    for j, h_pos in enumerate(hidden2_pos):
        line, = ax_net.plot([h_pos[0], output_pos[0][0]], [h_pos[1], output_pos[0][1]], 
                           color='gray', linewidth=1, alpha=0.8)
        layer3_lines.append(line)
    
    # Draw nodes on top of edges
    node_size = 300
    # Input node
    ax_net.scatter([p[0] for p in input_pos], [p[1] for p in input_pos], 
                   s=node_size, c='#4ECDC4', edgecolors='white', linewidths=2, zorder=5)
    # Hidden layer 1 nodes
    ax_net.scatter([p[0] for p in hidden1_pos], [p[1] for p in hidden1_pos], 
                   s=node_size, c='#9B59B6', edgecolors='white', linewidths=2, zorder=5)
    # Hidden layer 2 nodes
    ax_net.scatter([p[0] for p in hidden2_pos], [p[1] for p in hidden2_pos], 
                   s=node_size, c='#9B59B6', edgecolors='white', linewidths=2, zorder=5)
    # Output node
    ax_net.scatter([p[0] for p in output_pos], [p[1] for p in output_pos], 
                   s=node_size, c='#E74C3C', edgecolors='white', linewidths=2, zorder=5)
    
    # Add layer labels
    ax_net.text(0, -0.3, 'Input', ha='center', va='top', color='white', fontsize=9)
    ax_net.text(layer_spacing, -0.3, 'Hidden1', ha='center', va='top', color='white', fontsize=9)
    ax_net.text(2 * layer_spacing, -0.3, 'Hidden2', ha='center', va='top', color='white', fontsize=9)
    ax_net.text(3 * layer_spacing, -0.3, 'Output', ha='center', va='top', color='white', fontsize=9)
    
    # Add colorbar for weights
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_net, fraction=0.03, pad=0.02, aspect=20)
    cbar.set_label('Weight Value', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    # Subplot 3: Loss curve (bottom, spanning both columns)
    ax_loss = fig.add_subplot(gs[1, :], facecolor='black')
    ax_loss.set_title('Training Loss', fontsize=14, color='white', fontweight='bold')
    ax_loss.set_xlabel('Epoch', color='white')
    ax_loss.set_ylabel('MSE Loss', color='white')
    ax_loss.set_xlim(0, n_epochs)
    min_loss = min(history['loss'])
    max_loss = max(history['loss'])
    ax_loss.set_ylim(min_loss * 0.5, max_loss * 1.2)
    ax_loss.set_yscale('log')
    ax_loss.grid(True, alpha=0.2, color='gray')
    ax_loss.tick_params(colors='white')
    for spine in ax_loss.spines.values():
        spine.set_color('gray')
    
    loss_line, = ax_loss.plot([], [], color='#2ECC71', linewidth=2)
    loss_point, = ax_loss.plot([], [], 'o', color='#2ECC71', markersize=10)
    
    # Title for epoch counter
    epoch_text = fig.suptitle('', fontsize=16, fontweight='bold', color='white')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    def init():
        prediction_line.set_data([], [])
        loss_line.set_data([], [])
        loss_point.set_data([], [])
        all_lines = layer1_lines + [l[2] for l in layer2_lines] + layer3_lines
        return [prediction_line, loss_line, loss_point] + all_lines
    
    def animate(frame):
        # Update epoch text
        epoch_text.set_text(f'Epoch {frame} / {n_epochs-1}  |  Loss: {history["loss"][frame]:.6f}')
        
        # Update prediction curve
        prediction_line.set_data(x_plot.flatten(), history['predictions'][frame].flatten())
        
        # Update network edges - Layer 1 (input -> hidden1)
        w1 = history['layer1_weights'][frame]  # Shape: (hidden_size, 1)
        for j, line in enumerate(layer1_lines):
            weight = w1[j, 0]
            color = cmap(norm(weight))
            width = 1 + 3 * abs(weight) / w_max  # Width from 1 to 4
            line.set_color(color)
            line.set_linewidth(width)
            line.set_alpha(1.0)
        
        # Update network edges - Layer 2 (hidden1 -> hidden2)
        w2 = history['layer2_weights'][frame]  # Shape: (hidden_size, hidden_size)
        for i, j, line in layer2_lines:
            weight = w2[j, i]  # w2[out, in]
            color = cmap(norm(weight))
            width = 0.5 + 2 * abs(weight) / w_max  # Width from 0.5 to 2.5
            line.set_color(color)
            line.set_linewidth(width)
            line.set_alpha(0.8)
        
        # Update network edges - Layer 3 (hidden2 -> output)
        w3 = history['layer3_weights'][frame]  # Shape: (1, hidden_size)
        for j, line in enumerate(layer3_lines):
            weight = w3[0, j]
            color = cmap(norm(weight))
            width = 1 + 3 * abs(weight) / w_max  # Width from 1 to 4
            line.set_color(color)
            line.set_linewidth(width)
            line.set_alpha(1.0)
        
        # Update loss curve
        loss_line.set_data(range(frame + 1), history['loss'][:frame + 1])
        loss_point.set_data([frame], [history['loss'][frame]])
        
        all_lines = layer1_lines + [l[2] for l in layer2_lines] + layer3_lines
        return [prediction_line, loss_line, loss_point] + all_lines
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=n_epochs, interval=1000/fps, blit=False
    )
    
    # Save as video
    print(f"\nSaving animation to {save_path}...")
    writer = animation.FFMpegWriter(fps=fps, metadata={'title': 'Weight Evolution'})
    anim.save(save_path, writer=writer, dpi=150)
    print(f"Animation saved to {save_path}")
    
    plt.close()
    
    # Reset style
    plt.style.use('default')
    
    return anim


# -----------------------------------------------------------------------------
# 5. Main execution
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Parameters
    HIDDEN_SIZE = 8      # Number of neurons in each hidden layer
    EPOCHS = 500         # Training epochs
    LEARNING_RATE = 0.05  # Learning rate
    NOISE_STD = 0.15     # Noise standard deviation
    
    print("="*60)
    print("Neural Network Curve Fitting with Weight Evolution")
    print("="*60)
    print(f"\nNetwork: 1 -> {HIDDEN_SIZE} -> {HIDDEN_SIZE} -> 1 (tanh activation)")
    print(f"Training: {EPOCHS} epochs, lr={LEARNING_RATE}")
    
    # Generate noisy data
    print("\n1. Generating noisy target data...")
    x_train, y_train, y_clean = generate_data(n_points=100, noise_std=NOISE_STD)
    print(f"   Data points: {len(x_train)}")
    print(f"   x range: [{x_train.min():.2f}, {x_train.max():.2f}]")
    print(f"   y range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    print(f"   Noise std: {NOISE_STD}")
    
    # Create model
    print("\n2. Creating network...")
    model = SimpleNet(hidden_size=HIDDEN_SIZE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params}")
    print(f"   Layer 1: 1x{HIDDEN_SIZE} weights + {HIDDEN_SIZE} bias = {HIDDEN_SIZE + HIDDEN_SIZE}")
    print(f"   Layer 2: {HIDDEN_SIZE}x{HIDDEN_SIZE} weights + {HIDDEN_SIZE} bias = {HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE}")
    print(f"   Layer 3: {HIDDEN_SIZE}x1 weights + 1 bias = {HIDDEN_SIZE + 1}")
    
    # Train and record
    print("\n3. Training network...")
    history = train_and_record(model, x_train, y_train, 
                               epochs=EPOCHS, lr=LEARNING_RATE)
    
    print(f"\n   Final loss: {history['loss'][-1]:.8f}")
    
    # Create animation
    print("\n4. Creating animation video...")
    create_animation(x_train, y_train, history, x_plot=x_train,
                     save_path='weights_evolution.mp4', fps=30)
    
    print("\n" + "="*60)
    print("Done! Check 'weights_evolution.mp4' for the video.")
    print("="*60)
