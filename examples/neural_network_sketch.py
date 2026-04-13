"""
Simple Neural Network Sketch
Shows the key ingredients: weights, nonlinear activations, and output function.
Architecture: Input(1) -> Hidden1(2, tanh) -> Hidden2(2, tanh) -> Output(1, linear)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def draw_network_sketch(save_path='neural_network_sketch.pdf'):
    plt.style.use('dark_background')
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-2.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # --- Layout ---
    # Layers x-positions
    x_input = 1.0
    x_sum1 = 3.5      # summation nodes layer 1
    x_act1 = 5.5      # activation layer 1
    x_sum2 = 8.0      # summation nodes layer 2
    x_act2 = 10.0     # activation layer 2
    x_sum3 = 12.5     # output summation
    x_out = 14.0      # output
    
    # y-positions
    y_input = 1.5
    y_h = [2.8, 0.2]   # 2 neurons
    y_out = 1.5
    
    node_r = 0.4  # node radius
    act_w = 0.7   # activation box width
    act_h = 0.7   # activation box height
    
    # Colors
    c_input = '#4ECDC4'
    c_sum = '#2C2C2C'
    c_act = '#FFD93D'
    c_output = '#E74C3C'
    c_weight = '#5DADE2'
    c_bias = '#BB8FCE'
    edge_color = 'white'
    
    # =========================================================================
    # Helper functions
    # =========================================================================
    def draw_node(x, y, r, color, label='', fontsize=12, lw=2):
        circle = plt.Circle((x, y), r, facecolor=color, edgecolor=edge_color, linewidth=lw, zorder=5)
        ax.add_patch(circle)
        if label:
            ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, 
                    fontweight='bold', color='white', zorder=6)
    
    def draw_activation_box(x, y, label='tanh'):
        """Draw a box with activation function and a small plot of it inside."""
        rect = mpatches.FancyBboxPatch(
            (x - act_w/2, y - act_h/2), act_w, act_h,
            boxstyle="round,pad=0.05", facecolor=c_act, edgecolor=edge_color, linewidth=2, zorder=5
        )
        ax.add_patch(rect)
        # Draw a tiny tanh curve inside
        t = np.linspace(-2, 2, 50)
        s = np.tanh(t)
        # Scale to fit inside the box
        t_scaled = x - act_w/2 + 0.08 + (t - t.min()) / (t.max() - t.min()) * (act_w - 0.16)
        s_scaled = y - act_h/2 + 0.08 + (s - s.min()) / (s.max() - s.min()) * (act_h - 0.16)
        ax.plot(t_scaled, s_scaled, color='black', linewidth=1.5, zorder=6)
        # Label below
        ax.text(x, y - act_h/2 - 0.18, label, ha='center', va='top', fontsize=10, 
                color='white', fontstyle='italic')
    
    def draw_linear_box(x, y):
        """Draw a box with linear identity function."""
        rect = mpatches.FancyBboxPatch(
            (x - act_w/2, y - act_h/2), act_w, act_h,
            boxstyle="round,pad=0.05", facecolor='#AED6F1', edgecolor=edge_color, linewidth=2, zorder=5
        )
        ax.add_patch(rect)
        # Draw identity line inside
        t_scaled = [x - act_w/2 + 0.1, x + act_w/2 - 0.1]
        s_scaled = [y - act_h/2 + 0.1, y + act_h/2 - 0.1]
        ax.plot(t_scaled, s_scaled, color='black', linewidth=1.5, zorder=6)
        ax.text(x, y - act_h/2 - 0.18, 'linear', ha='center', va='top', fontsize=10, 
                color='white', fontstyle='italic')
    
    def draw_arrow(x1, y1, x2, y2, color=edge_color, lw=1.5, label='', label_above=True,
                   label_color=c_weight, fontsize=10, shrink_start=0, shrink_end=0):
        """Draw an arrow with optional weight label."""
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        ux, uy = dx/length, dy/length
        
        # Shrink start and end
        sx = x1 + ux * shrink_start
        sy = y1 + uy * shrink_start
        ex = x2 - ux * shrink_end
        ey = y2 - uy * shrink_end
        
        ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw, 
                                   shrinkA=0, shrinkB=0),
                    zorder=3)
        
        if label:
            mx = (sx + ex) / 2
            my = (sy + ey) / 2
            offset = 0.2 if label_above else -0.2
            # Perpendicular offset
            px, py = -uy * offset, ux * offset
            ax.text(mx + px, my + py, label, ha='center', va='center', 
                    fontsize=fontsize, color=label_color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='black', 
                             edgecolor='none', alpha=0.8),
                    zorder=4)
    
    # =========================================================================
    # Draw the network
    # =========================================================================
    
    # --- Input node ---
    draw_node(x_input, y_input, node_r, c_input, '$x$', fontsize=14)
    ax.text(x_input, y_input + node_r + 0.3, 'Input', ha='center', va='bottom', 
            fontsize=11, color=edge_color, fontweight='bold')
    
    # --- Layer 1: Input -> Summation nodes ---
    # Weights from input to hidden1
    for i, yh in enumerate(y_h):
        # Arrow: input -> sum node
        draw_arrow(x_input, y_input, x_sum1, yh, color=c_weight, lw=2,
                   label=f'$w^{{(1)}}_{{{i+1}}}$', label_above=(i == 0),
                   shrink_start=node_r, shrink_end=node_r)
    
    # Summation nodes layer 1
    for i, yh in enumerate(y_h):
        draw_node(x_sum1, yh, node_r, c_sum, '$\\Sigma$', fontsize=14)
        # Bias arrow (coming from above/below)
        bias_y_start = yh + 1.2 if i == 0 else yh - 1.2
        draw_arrow(x_sum1, bias_y_start, x_sum1, yh, color=c_bias, lw=1.5,
                   shrink_start=0, shrink_end=node_r)
        bias_label_y = bias_y_start + (0.2 if i == 0 else -0.2)
        ax.text(x_sum1 + 0.3, bias_label_y, f'$b^{{(1)}}_{{{i+1}}}$', ha='left', va='center',
                fontsize=10, color=c_bias, fontweight='bold')
    
    # --- Arrows: sum nodes -> activation boxes (layer 1) ---
    for i, yh in enumerate(y_h):
        draw_arrow(x_sum1, yh, x_act1, yh, color=edge_color, lw=1.5,
                   shrink_start=node_r, shrink_end=act_w/2)
    
    # Activation boxes layer 1
    for yh in y_h:
        draw_activation_box(x_act1, yh, 'tanh')
    
    # --- Layer 2: Activation1 outputs -> Summation nodes layer 2 ---
    # This has 2x2 = 4 connections
    weight_labels_l2 = [
        ['$w^{(2)}_{11}$', '$w^{(2)}_{21}$'],  # from neuron 1 to neurons 1,2
        ['$w^{(2)}_{12}$', '$w^{(2)}_{22}$'],  # from neuron 2 to neurons 1,2
    ]
    
    for i, yh_from in enumerate(y_h):
        for j, yh_to in enumerate(y_h):
            above = (yh_from >= yh_to)
            draw_arrow(x_act1, yh_from, x_sum2, yh_to, color=c_weight, lw=2,
                       label=weight_labels_l2[i][j], label_above=above,
                       shrink_start=act_w/2, shrink_end=node_r,
                       fontsize=9)
    
    # Summation nodes layer 2
    for i, yh in enumerate(y_h):
        draw_node(x_sum2, yh, node_r, c_sum, '$\\Sigma$', fontsize=14)
        bias_y_start = yh + 1.2 if i == 0 else yh - 1.2
        draw_arrow(x_sum2, bias_y_start, x_sum2, yh, color=c_bias, lw=1.5,
                   shrink_start=0, shrink_end=node_r)
        bias_label_y = bias_y_start + (0.2 if i == 0 else -0.2)
        ax.text(x_sum2 + 0.3, bias_label_y, f'$b^{{(2)}}_{{{i+1}}}$', ha='left', va='center',
                fontsize=10, color=c_bias, fontweight='bold')
    
    # --- Arrows: sum nodes -> activation boxes (layer 2) ---
    for i, yh in enumerate(y_h):
        draw_arrow(x_sum2, yh, x_act2, yh, color=edge_color, lw=1.5,
                   shrink_start=node_r, shrink_end=act_w/2)
    
    # Activation boxes layer 2
    for yh in y_h:
        draw_activation_box(x_act2, yh, 'tanh')
    
    # --- Output layer: Activation2 -> output sum -> linear -> output ---
    for i, yh in enumerate(y_h):
        draw_arrow(x_act2, yh, x_sum3, y_out, color=c_weight, lw=2,
                   label=f'$w^{{(3)}}_{{{i+1}}}$', label_above=(i == 0),
                   shrink_start=act_w/2, shrink_end=node_r)
    
    # Output summation
    draw_node(x_sum3, y_out, node_r, c_sum, '$\\Sigma$', fontsize=14)
    bias_y_start = y_out + 1.5
    draw_arrow(x_sum3, bias_y_start, x_sum3, y_out, color=c_bias, lw=1.5,
               shrink_start=0, shrink_end=node_r)
    ax.text(x_sum3 + 0.3, bias_y_start + 0.2, '$b^{(3)}$', ha='left', va='center',
            fontsize=10, color=c_bias, fontweight='bold')
    
    # Arrow to linear activation
    draw_arrow(x_sum3, y_out, x_out, y_out, color=edge_color, lw=1.5,
               shrink_start=node_r, shrink_end=act_w/2)
    
    # Linear output box
    draw_linear_box(x_out, y_out)
    
    # Output label
    ax.text(x_out + act_w/2 + 0.3, y_out, '$\\hat{y}$', ha='left', va='center',
            fontsize=16, color=c_output, fontweight='bold')
    
    # =========================================================================
    # Layer labels at top
    # =========================================================================
    label_y = 4.8
    ax.text(x_input, label_y, 'Input\nLayer', ha='center', va='center', fontsize=11,
            color=edge_color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=c_input, alpha=0.3, edgecolor='none'))
    ax.text((x_sum1 + x_act1)/2, label_y, 'Hidden Layer 1', ha='center', va='center', fontsize=11,
            color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=c_act, alpha=0.3, edgecolor='none'))
    ax.text((x_sum2 + x_act2)/2, label_y, 'Hidden Layer 2', ha='center', va='center', fontsize=11,
            color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=c_act, alpha=0.3, edgecolor='none'))
    ax.text((x_sum3 + x_out)/2, label_y, 'Output\nLayer', ha='center', va='center', fontsize=11,
            color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#AED6F1', alpha=0.3, edgecolor='none'))
    
    # =========================================================================
    # Legend
    # =========================================================================
    legend_x = 0.5
    legend_y = -1.5
    legend_items = [
        ('$w$  Weights', c_weight),
        ('$b$  Biases', c_bias),
        ('$\\Sigma$  Weighted sum', edge_color),
        ('tanh  Nonlinear activation', '#B8860B'),
        ('linear  Output activation', '#2471A3'),
    ]
    for i, (label, color) in enumerate(legend_items):
        ax.text(legend_x + i * 3.0, legend_y, label, ha='center', va='center',
                fontsize=9, color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a1a', edgecolor=color, alpha=0.8))
    
    # =========================================================================
    # Equation at bottom
    # =========================================================================
    eq_y = -2.2
    ax.text(7.5, eq_y, 
            r'$\hat{y} = W^{(3)} \cdot \tanh\left(W^{(2)} \cdot \tanh\left(W^{(1)} x + b^{(1)}\right) + b^{(2)}\right) + b^{(3)}$',
            ha='center', va='center', fontsize=13, color=edge_color,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#2C2C2C', edgecolor='gray', alpha=0.8))
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == '__main__':
    draw_network_sketch('neural_network_sketch.pdf')
