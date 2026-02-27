import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple, Callable
from pinns.domain import DomainCubicPartition


class FourierFeatures:
    """
    Random Fourier Feature encoding to mitigate spectral bias.
    
    Maps input coordinates into high-frequency signals before passing through MLP.
    Based on Tancik et al. "Fourier Features Let Networks Learn High Frequency
    Functions in Low Dimensional Domains" (NeurIPS 2020).
    
    The encoding γ: R^d → R^(2m) is defined by:
        γ(x) = [cos(2π B x), sin(2π B x)]
    where B ∈ R^(m×d) has entries sampled from N(0, σ²).
    
    Example:
        fourier = FourierFeatures(input_dim=3, n_features=128, sigma=10.0)
        # Network input will be 2*128 = 256 features
        model = FNN([256, 64, 1], feature_encoding=fourier)
    
    Attributes:
        B: The random projection matrix of shape (n_features, input_dim)
        output_dim: Dimension of encoded output (2*n_features or 2*n_features + input_dim)
    """
    
    def __init__(self, 
                 input_dim: int, 
                 n_features: int, 
                 sigma: float = 1.0, 
                 seed: int = 0,
                 include_input: bool = False):
        """
        Initialize Fourier feature encoding.
        
        Args:
            input_dim: Dimension of input coordinates (e.g., 2 for 2D, 3 for 3D)
            n_features: Number of Fourier features (output will be 2*n_features)
            sigma: Standard deviation for sampling B matrix. Higher values = higher
                   frequencies. Typical values: 1-10 for normalized inputs.
            seed: Random seed for reproducible B matrix
            include_input: If True, concatenate original input to Fourier features.
                          Output dim becomes 2*n_features + input_dim.
        """
        self.input_dim = input_dim
        self.n_features = n_features
        self.sigma = sigma
        self.include_input = include_input
        
        # Generate random B matrix: (n_features, input_dim)
        # Entries sampled from N(0, σ²)
        torch.manual_seed(seed)
        self.B = torch.randn(n_features, input_dim) * sigma
        self._device = 'cpu'
        
        # Output dimension
        self.output_dim = 2 * n_features + (input_dim if include_input else 0)
    
    def to(self, device):
        """Move B matrix to specified device."""
        self.B = self.B.to(device)
        self._device = device
        return self
    
    def __call__(self, x: torch.Tensor, params: Optional[Dict] = None) -> torch.Tensor:
        """
        Apply Fourier feature encoding.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            params: Ignored (for API compatibility with input_transform)
        
        Returns:
            Encoded tensor of shape (batch_size, output_dim)
        """
        # Ensure B is on same device as x
        if self.B.device != x.device:
            self.B = self.B.to(x.device)
        
        # x: (batch, input_dim)
        # B: (n_features, input_dim)
        # Bx = x @ B.T: (batch, n_features)
        Bx = x @ self.B.T
        
        # γ(x) = [cos(2π Bx), sin(2π Bx)]
        cos_features = torch.cos(2 * np.pi * Bx)
        sin_features = torch.sin(2 * np.pi * Bx)
        features = torch.cat([cos_features, sin_features], dim=-1)
        
        # Optionally include original input
        if self.include_input:
            features = torch.cat([x, features], dim=-1)
        
        return features
    
    def transform(self, x: torch.Tensor, params: Optional[Dict] = None) -> torch.Tensor:
        """Alias for __call__ for explicit usage."""
        return self.__call__(x, params)


class LinearRWF(nn.Module):
    """
    Linear layer with Random Weight Factorization (RWF).
    
    Implements W = diag(exp(s)) · V where s and V are trainable.
    Based on Wang et al. "On the eigenvector bias of Fourier feature networks".
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        rwf_mu: Mean for initializing s from N(mu, sigma*I). Recommended: 0.5 or 1.0
        rwf_sigma: Std for initializing s from N(mu, sigma*I). Recommended: 0.1
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 rwf_mu: float = 0.5, rwf_sigma: float = 0.1):
        super(LinearRWF, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rwf_mu = rwf_mu
        self.rwf_sigma = rwf_sigma
        
        # V matrix: (out_features, in_features) - initialized with Glorot
        self.V = nn.Parameter(torch.empty(out_features, in_features))
        
        # s vector: (out_features,) - initialized from N(mu, sigma*I)
        self.s = nn.Parameter(torch.empty(out_features))
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        # V: Glorot/Xavier initialization
        nn.init.xavier_normal_(self.V)
        
        # s: N(mu, sigma*I)
        nn.init.normal_(self.s, mean=self.rwf_mu, std=self.rwf_sigma)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: y = (diag(exp(s)) · V) @ x + b"""
        # Compute W = diag(exp(s)) · V
        # exp(s): (out_features,) -> (out_features, 1) for broadcasting
        W = torch.exp(self.s).unsqueeze(1) * self.V  # (out_features, in_features)
        
        # Linear transform: y = x @ W.T + b
        return torch.nn.functional.linear(x, W, self.bias)


class FNN(nn.Module):
    """
    A fully-connected neural network with configurable layer sizes and activation function.
    
    Includes optional input normalization and output unnormalization layers.
    The normalization bounds are set by the Trainer based on the problem definition.
    
    Args:
        layer_sizes (list): List of integers specifying the size of each layer.
        activation (nn.Module or str): Activation function. Default: 'tanh'
        output_activation (nn.Module or None): Optional activation for output layer. Default: None
        normalize_input (bool): Whether to normalize inputs. Default: True
        unnormalize_output (bool): Whether to unnormalize outputs. Default: True
        input_transform (callable or None): Optional transformation applied BEFORE normalization.
        output_transform (callable or None): Optional transformation applied after unnormalization.
        feature_encoding: Optional feature encoding (e.g., FourierFeatures)
        seed (int): Random seed for weight initialization. Default: None
    """
    
    def __init__(self, layer_sizes, activation='tanh', output_activation=None,
                 normalize_input=True, unnormalize_output=True, 
                 input_transform=None, output_transform=None, 
                 feature_encoding=None, seed=None):
        super(FNN, self).__init__()
        
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 elements (input and output sizes)")
        
        self.layer_sizes = layer_sizes
        self.activation = self._get_activation(activation)
        self.output_activation = self._get_activation(output_activation) if output_activation else None
        self.normalize_input = normalize_input
        self.unnormalize_output = unnormalize_output
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.feature_encoding = feature_encoding
        self.seed = seed
        
        # Input normalization parameters (set by Trainer)
        self.register_buffer('input_min', None)
        self.register_buffer('input_max', None)
        
        # Output unnormalization parameters (set by Trainer)
        self.register_buffer('output_min', None)
        self.register_buffer('output_max', None)
        
        # Build layers
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        
        self.layers = nn.ModuleList(layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def set_input_range(self, xmin, xmax):
        """Set input normalization range. Called by Trainer."""
        self.input_min = torch.as_tensor(xmin, dtype=torch.float32, device=self.layers[0].weight.device)
        self.input_max = torch.as_tensor(xmax, dtype=torch.float32, device=self.layers[0].weight.device)
    
    def set_output_range(self, ymin, ymax):
        """Set output unnormalization range. Called by Trainer."""
        self.output_min = torch.as_tensor(ymin, dtype=torch.float32, device=self.layers[0].weight.device)
        self.output_max = torch.as_tensor(ymax, dtype=torch.float32, device=self.layers[0].weight.device)
    
    def _get_activation(self, activation):
        """Convert activation string to nn.Module or return the module directly."""
        if isinstance(activation, nn.Module):
            return activation
        
        activation_dict = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'softplus': nn.Softplus(),
        }
        
        if isinstance(activation, str):
            activation_lower = activation.lower()
            if activation_lower in activation_dict:
                return activation_dict[activation_lower]
            else:
                raise ValueError(f"Unknown activation: {activation}. "
                               f"Available: {list(activation_dict.keys())}")
        
        raise ValueError(f"Activation must be a string or nn.Module, got {type(activation)}")
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        if self.seed is not None:
            torch.manual_seed(self.seed)
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def _normalize_input(self, x):
        """Normalize input from [input_min, input_max] to [-1, 1]."""
        if self.input_min is None or not self.normalize_input:
            return x
        # x_norm = 2 * (x - xmin) / (xmax - xmin) - 1
        return 2.0 * (x - self.input_min) / (self.input_max - self.input_min + 1e-8) - 1.0
    
    def _unnormalize_output(self, y):
        """Unnormalize output from [-1, 1] to [output_min, output_max]."""
        if self.output_min is None or not self.unnormalize_output:
            return y
        # y_unnorm = (y + 1) / 2 * (ymax - ymin) + ymin
        return (y + 1.0) / 2.0 * (self.output_max - self.output_min) + self.output_min
    
    def forward(self, x, params=None):
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, n_inputs)
            params: Optional dict of parameters passed to output_transform
        """
        # Store original input for output_transform
        x_original = x
        
        # Apply input transform (before normalization, e.g., for symmetries)
        if self.input_transform is not None:
            x = self.input_transform(x, params)
        
        # Normalize input
        x = self._normalize_input(x)
        
        # Apply feature encoding (e.g., Fourier features) after normalization
        if self.feature_encoding is not None:
            x = self.feature_encoding(x, params)
        
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
        
        # Output layer (no activation by default)
        x = self.layers[-1](x)
        
        if self.output_activation:
            x = self.output_activation(x)
        
        # Unnormalize output
        x = self._unnormalize_output(x)
        
        # Apply output transform for hard constraints
        if self.output_transform is not None:
            x = self.output_transform(x_original, x, params)
        
        return x
    
    def predict(self, x_np: np.ndarray, params_dict=None) -> np.ndarray:
        """
        Predict with numpy I/O.
        
        Args:
            x_np: Input numpy array of shape (batch_size, n_inputs)
            params_dict: Optional dict passed to transforms
            
        Returns:
            Output numpy array of shape (batch_size, n_outputs)
        """
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x_np, device=next(self.parameters()).device, 
                                   dtype=next(self.parameters()).dtype)
            y = self.forward(x_tensor, params_dict)
            return y.cpu().numpy()


class WFFNN(nn.Module):
    """
    Weight-Factorized Fully-connected Neural Network for PyTorch.
    
    Uses Random Weight Factorization (RWF): W = diag(exp(s)) · V
    Based on Wang et al. "On the eigenvector bias of Fourier feature networks".
    
    Recommended settings: rwf_mu=0.5 or 1.0, rwf_sigma=0.1
    
    Args:
        layer_sizes (list): List of integers specifying the size of each layer.
        activation (nn.Module or str): Activation function. Default: 'tanh'
        output_activation (nn.Module or None): Optional output activation. Default: None
        normalize_input (bool): Whether to normalize inputs. Default: True
        unnormalize_output (bool): Whether to unnormalize outputs. Default: True
        input_transform (callable or None): Optional transformation before normalization.
        output_transform (callable or None): Optional transformation after unnormalization.
        feature_encoding: Optional feature encoding (e.g., FourierFeatures)
        rwf_mu (float): Mean for RWF s initialization. Recommended: 0.5 or 1.0
        rwf_sigma (float): Std for RWF s initialization. Recommended: 0.1
        seed (int): Random seed for weight initialization. Default: None
    """
    
    def __init__(self, layer_sizes, activation='tanh', output_activation=None,
                 normalize_input=True, unnormalize_output=True, 
                 input_transform=None, output_transform=None, 
                 feature_encoding=None, rwf_mu=0.5, rwf_sigma=0.1, seed=None):
        super(WFFNN, self).__init__()
        
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 elements (input and output sizes)")
        
        self.layer_sizes = layer_sizes
        self.activation = self._get_activation(activation)
        self.output_activation = self._get_activation(output_activation) if output_activation else None
        self.normalize_input = normalize_input
        self.unnormalize_output = unnormalize_output
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.feature_encoding = feature_encoding
        self.rwf_mu = rwf_mu
        self.rwf_sigma = rwf_sigma
        self.seed = seed
        
        # Input normalization parameters (set by Trainer)
        self.register_buffer('input_min', None)
        self.register_buffer('input_max', None)
        
        # Output unnormalization parameters (set by Trainer)
        self.register_buffer('output_min', None)
        self.register_buffer('output_max', None)
        
        # Build layers with RWF
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(LinearRWF(layer_sizes[i], layer_sizes[i + 1], 
                                   rwf_mu=rwf_mu, rwf_sigma=rwf_sigma))
        
        self.layers = nn.ModuleList(layers)
    
    def set_input_range(self, xmin, xmax):
        """Set input normalization range. Called by Trainer."""
        self.input_min = torch.as_tensor(xmin, dtype=torch.float32, device=self.layers[0].V.device)
        self.input_max = torch.as_tensor(xmax, dtype=torch.float32, device=self.layers[0].V.device)
    
    def set_output_range(self, ymin, ymax):
        """Set output unnormalization range. Called by Trainer."""
        self.output_min = torch.as_tensor(ymin, dtype=torch.float32, device=self.layers[0].V.device)
        self.output_max = torch.as_tensor(ymax, dtype=torch.float32, device=self.layers[0].V.device)
    
    def _get_activation(self, activation):
        """Convert activation string to nn.Module or return the module directly."""
        if isinstance(activation, nn.Module):
            return activation
        
        activation_dict = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'softplus': nn.Softplus(),
        }
        
        if isinstance(activation, str):
            activation_lower = activation.lower()
            if activation_lower in activation_dict:
                return activation_dict[activation_lower]
            else:
                raise ValueError(f"Unknown activation: {activation}. "
                               f"Available: {list(activation_dict.keys())}")
        
        raise ValueError(f"Activation must be a string or nn.Module, got {type(activation)}")
    
    def _normalize_input(self, x):
        """Normalize input from [input_min, input_max] to [-1, 1]."""
        if self.input_min is None or not self.normalize_input:
            return x
        return 2.0 * (x - self.input_min) / (self.input_max - self.input_min + 1e-8) - 1.0
    
    def _unnormalize_output(self, y):
        """Unnormalize output from [-1, 1] to [output_min, output_max]."""
        if self.output_min is None or not self.unnormalize_output:
            return y
        return (y + 1.0) / 2.0 * (self.output_max - self.output_min) + self.output_min
    
    def forward(self, x, params=None):
        """Forward pass through the network."""
        x_original = x
        
        # Apply input transform (e.g., symmetry)
        if self.input_transform is not None:
            x = self.input_transform(x, params)
        
        # Normalize input
        x = self._normalize_input(x)
        
        # Apply feature encoding
        if self.feature_encoding is not None:
            x = self.feature_encoding(x, params)
        
        # Hidden layers with RWF
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        
        # Output layer
        x = self.layers[-1](x)
        
        # Output activation
        if self.output_activation is not None:
            x = self.output_activation(x)
        
        # Unnormalize output
        x = self._unnormalize_output(x)
        
        # Apply output transform
        if self.output_transform is not None:
            x = self.output_transform(x_original, x, params)
        
        return x
    
    def predict(self, x_np, params=None):
        """Predict with numpy I/O."""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x_np, device=self.layers[0].V.device, 
                                   dtype=self.layers[0].V.dtype)
            y = self.forward(x_tensor, params)
            return y.cpu().numpy()



class PirateNetBlock(nn.Module):
    """
    Single residual block for PirateNet with Random Weight Factorization.
    
    Each block has 3 dense layers with gating operations:
    f = σ(W1·x + b1)
    z1 = f ⊙ U + (1-f) ⊙ V
    g = σ(W2·z1 + b2)
    z2 = g ⊙ U + (1-g) ⊙ V
    h = σ(W3·z2 + b3)
    x_next = α·h + (1-α)·x
    """
    
    def __init__(self, hidden_dim: int, activation='tanh',
                 rwf_mu: float = 0.5, rwf_sigma: float = 0.1):
        super(PirateNetBlock, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.activation = self._get_activation(activation)
        self.rwf_mu = rwf_mu
        self.rwf_sigma = rwf_sigma
        
        # Three dense layers per block (using RWF)
        self.dense1 = LinearRWF(hidden_dim, hidden_dim, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
        self.dense2 = LinearRWF(hidden_dim, hidden_dim, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
        self.dense3 = LinearRWF(hidden_dim, hidden_dim, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
        
        # Trainable α parameter (initialized to 0 for identity at init)
        self.alpha = nn.Parameter(torch.zeros(1))
    
    def _get_activation(self, activation):
        activation_dict = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
        }
        if isinstance(activation, str):
            return activation_dict.get(activation.lower(), nn.Tanh())
        return activation
    
    def forward(self, x: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Forward pass through one residual block."""
        # First dense + gating
        f = self.activation(self.dense1(x))
        z1 = f * U + (1 - f) * V
        
        # Second dense + gating
        g = self.activation(self.dense2(z1))
        z2 = g * U + (1 - g) * V
        
        # Third dense
        h = self.activation(self.dense3(z2))
        
        # Adaptive residual connection
        x_next = self.alpha * h + (1 - self.alpha) * x
        
        return x_next


class PirateNet(nn.Module):
    """
    Physics-Informed Residual AdapTivE Network (PirateNet) for PyTorch.
    
    A novel architecture that addresses initialization issues in PINNs by using
    adaptive residual connections with trainable α parameters initialized to 0.
    At initialization, the network acts as a linear combination of input embeddings.
    Uses Random Weight Factorization (RWF) for improved spectral properties.
    
    Based on Wang et al. "PirateNets: Physics-informed Deep Learning 
    with Residual Adaptive Networks".
    
    Architecture:
    1. Optional input embedding via feature_encoding (e.g., FourierFeatures)
    2. Gate encodings: U = σ(W1·x + b1), V = σ(W2·x + b2)
    3. L residual blocks, each with 3 dense layers and gating operations
    4. Output layer
    
    Args:
        input_dim: Dimension of input coordinates (before any feature encoding)
        output_dim: Dimension of output
        hidden_dim: Width of all hidden layers (must be consistent)
        n_blocks: Number of residual blocks (default: 3). Total depth = 3*n_blocks.
        activation: Activation function name (default: 'tanh')
        normalize_input: Whether to normalize inputs to [-1, 1]
        unnormalize_output: Whether to unnormalize outputs
        input_transform: Optional symmetry transform
        output_transform: Optional hard constraint transform
        feature_encoding: Optional feature encoding (e.g., FourierFeatures)
        rwf_mu: Mean for Random Weight Factorization s initialization (default: 0.5)
        rwf_sigma: Std for Random Weight Factorization s initialization (default: 0.1)
        seed: Random seed for initialization
    
    Example:
        # Without Fourier features
        model = PirateNet(input_dim=2, output_dim=1, hidden_dim=64, n_blocks=3)
        
        # With Fourier features (input_dim is original dimension, not encoded)
        fourier = FourierFeatures(input_dim=2, n_features=64, sigma=1.0)
        model = PirateNet(input_dim=2, output_dim=1, hidden_dim=64,
                         feature_encoding=fourier)
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int,
                 n_blocks: int = 3, activation: str = 'tanh',
                 normalize_input: bool = True, unnormalize_output: bool = True,
                 input_transform=None, output_transform=None,
                 feature_encoding=None, rwf_mu: float = 0.5, rwf_sigma: float = 0.1,
                 seed: int = None):
        super(PirateNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.normalize_input = normalize_input
        self.unnormalize_output = unnormalize_output
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.feature_encoding = feature_encoding
        self.rwf_mu = rwf_mu
        self.rwf_sigma = rwf_sigma
        self.seed = seed
        
        # Determine actual input dimension after feature encoding
        if feature_encoding is not None and hasattr(feature_encoding, 'output_dim'):
            actual_input_dim = feature_encoding.output_dim
        else:
            actual_input_dim = input_dim
        
        # For compatibility with FNN interface
        self.layer_sizes = [input_dim, hidden_dim, output_dim]
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # Input normalization parameters (set by Trainer)
        self.register_buffer('input_min', None)
        self.register_buffer('input_max', None)
        self.register_buffer('output_min', None)
        self.register_buffer('output_max', None)
        
        # U and V gate layers (use actual_input_dim after encoding, using RWF)
        self.U_layer = LinearRWF(actual_input_dim, hidden_dim, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
        self.V_layer = LinearRWF(actual_input_dim, hidden_dim, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
        
        # Input projection to hidden_dim (using RWF)
        self.input_projection = LinearRWF(actual_input_dim, hidden_dim, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            PirateNetBlock(hidden_dim, activation, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma) for _ in range(n_blocks)
        ])
        
        # Output layer (using RWF)
        self.output_layer = LinearRWF(hidden_dim, output_dim, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
        
        # Activation for U, V
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation):
        activation_dict = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
        }
        if isinstance(activation, str):
            return activation_dict.get(activation.lower(), nn.Tanh())
        return activation
    
    def set_input_range(self, xmin, xmax):
        """Set input normalization range. Called by Trainer."""
        device = self.output_layer.V.device
        self.input_min = torch.as_tensor(xmin, dtype=torch.float32, device=device)
        self.input_max = torch.as_tensor(xmax, dtype=torch.float32, device=device)
    
    def set_output_range(self, ymin, ymax):
        """Set output unnormalization range. Called by Trainer."""
        device = self.output_layer.V.device
        self.output_min = torch.as_tensor(ymin, dtype=torch.float32, device=device)
        self.output_max = torch.as_tensor(ymax, dtype=torch.float32, device=device)
    
    def _normalize_input(self, x):
        """Normalize input from [input_min, input_max] to [-1, 1]."""
        if self.input_min is None or not self.normalize_input:
            return x
        return 2.0 * (x - self.input_min) / (self.input_max - self.input_min + 1e-8) - 1.0
    
    def _unnormalize_output(self, y):
        """Unnormalize output from [-1, 1] to [output_min, output_max]."""
        if self.output_min is None or not self.unnormalize_output:
            return y
        return (y + 1.0) / 2.0 * (self.output_max - self.output_min) + self.output_min
    
    def forward(self, x, params=None):
        """Forward pass through PirateNet."""
        x_original = x
        
        # Apply input transform (e.g., symmetry)
        if self.input_transform is not None:
            x = self.input_transform(x, params)
        
        # Normalize input
        x = self._normalize_input(x)
        
        # Apply feature encoding if provided
        if self.feature_encoding is not None:
            x = self.feature_encoding(x, params)
        
        # Compute U and V gates
        U = self.activation(self.U_layer(x))
        V = self.activation(self.V_layer(x))
        
        # Project to hidden_dim
        h = self.input_projection(x)
        
        # Apply residual blocks
        for block in self.blocks:
            h = block(h, U, V)
        
        # Output layer
        y = self.output_layer(h)
        
        # Unnormalize output
        y = self._unnormalize_output(y)
        
        # Apply output transform
        if self.output_transform is not None:
            y = self.output_transform(x_original, y, params)
        
        return y
    
    def predict(self, x_np, params=None):
        """Predict with numpy I/O."""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x_np, device=self.output_layer.V.device,
                                   dtype=self.output_layer.V.dtype)
            y = self.forward(x_tensor, params)
            return y.cpu().numpy()


class FBPINN(nn.Module):
    """
    Finite Basis Physics-Informed Neural Network (FBPINN).
    
    Combines multiple neural networks over a domain decomposition using 
    smooth window functions (partition of unity).
    
    The input normalization (per subdomain) and output unnormalization bounds
    are set by the Trainer based on the problem definition.
    
    Args:
        domain (DomainCubicPartition): DomainCubic partition defining subdomain centers and widths.
                                  The widths are used for the bump window function smoothing.
        networks (nn.Module or list): Either a single network (used as template to create 
                                     one network per subdomain) or a list of networks 
                                     (one per subdomain for active subdomains only).
        active_subdomains (list of bool, list of int, or None): Specifies which subdomains have networks.
                                     - None: All subdomains are active (default)
                                     - list of bool: Boolean mask, True = active (one per subdomain)
                                     - list of int: Indices of active subdomains
                                     Use domain.subdomains to build masks based on subdomain properties.
                                     Inactive subdomains don't create networks (saves memory).
                                     Useful for symmetry where input_transform maps points
                                     from inactive to active regions.
        normalize_input (bool): Whether to normalize inputs per subdomain to [-1, 1] and
                         normalize window functions to form a partition of unity.
                         Both are applied together since they're inherently linked.
                         Default: True
        unnormalize_output (bool): Whether to unnormalize outputs. Bounds set by Trainer.
                           Default: True
        input_transform (callable or None): Optional transformation applied BEFORE normalization.
                                            Function signature: f(x) -> x_transformed
                                            where x is the input tensor.
                                            Useful for enforcing symmetries (e.g., f(x) = |x| for even functions).
                                            Example: lambda x: torch.abs(x)  # Even symmetry
                                            Default: None
        output_transform (callable or None): Optional transformation applied after unnormalization.
                                             Function signature: f(x, y, params) -> y_transformed
                                             where x is the original input, y is the network output,
                                             and params is a dict passed during forward.
                                             Useful for implementing hard constraints.
                                             Example: lambda x, y, p: x[:, 0:1] * y  # y=0 at x=0
                                             Default: None
        device (str or torch.device): Device for tensors. Default: 'cpu'
        dtype (torch.dtype): Data type for tensors. Default: torch.float32
    
    Example:
        import numpy as np
        from pinns.domain import DomainCubicPartition
        from pinns.networks import FNN, FBPINN
        
        # Create domain partition (bounds derived from grid_positions)
        partition = DomainCubicPartition(
            grid_positions=[np.linspace(-1, 1, 5), np.linspace(0, 1, 5)],
            overlap=0.5  # 50% overlap between subdomains
        )
        
        # Add boundary conditions
        partition.add_dirichlet((0, None), value=0.0, component=0, name="left")
        
        # Reflection symmetry: only create networks for x >= 0
        # Use partition.subdomains to inspect subdomain bounds and build a mask
        active_mask = [sub.xmin[0] >= 0 for sub in partition.subdomains]
        
        network = FNN([2, 32, 32, 1])
        fbpinn = FBPINN(
            partition, network,
            active_subdomains=active_mask,
            input_transform=lambda x: torch.cat([torch.abs(x[:, 0:1]), x[:, 1:]], dim=1)
        )
        
        # Forward pass
        x = torch.randn(100, 2)  # 100 points in 2D
        y = fbpinn(x)  # Weighted sum of active network predictions
    """
    
    def __init__(self, domain, networks,
                 active_subdomains=None,
                 normalize_input=True, unnormalize_output=True,
                 input_transform=None, output_transform=None,
                 device='cpu', dtype=torch.float32):
        super(FBPINN, self).__init__()
        
        self.domain = domain
        self.n_subdomains = len(domain)
        self.n_dims = domain.n_dims
        self.normalize_input = normalize_input
        self.unnormalize_output = unnormalize_output
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.device = device
        self.dtype = dtype
        
        # Process active_subdomains parameter
        self._active_indices = self._compute_active_indices(active_subdomains)
        self.n_active_subdomains = len(self._active_indices)
        
        # Create mapping from subdomain index to network index (None if inactive)
        self._subdomain_to_network = {}
        for net_idx, sub_idx in enumerate(self._active_indices):
            self._subdomain_to_network[sub_idx] = net_idx
        
        # Boolean mask for active subdomains
        active_mask = torch.zeros(self.n_subdomains, dtype=torch.bool, device=device)
        active_mask[self._active_indices] = True
        self.register_buffer('active_mask', active_mask)
        
        # Get subdomain centers and widths as tensors (for window functions)
        centers, widths_lower, widths_upper = domain.to_torch(device=device, dtype=dtype)
        self.register_buffer('centers', centers)  # (n_subdomains, n_dims)
        self.register_buffer('widths_lower', widths_lower)  # (n_subdomains, n_dims) - at xmin boundary
        self.register_buffer('widths_upper', widths_upper)  # (n_subdomains, n_dims) - at xmax boundary
        
        # For backward compatibility, keep average widths
        widths_avg = (widths_lower + widths_upper) / 2
        self.register_buffer('widths', widths_avg)
        
        # Get actual subdomain bounds from partition positions (for input scaling)
        lower_bounds_np, upper_bounds_np = domain.get_subdomain_bounds()
        self.register_buffer('lower_bounds', torch.tensor(lower_bounds_np, device=device, dtype=dtype))
        self.register_buffer('upper_bounds', torch.tensor(upper_bounds_np, device=device, dtype=dtype))
        
        # Pre-compute extended bounds for point mask computation (overlap_factor=1.0 for safety margin)
        self.register_buffer('extended_lower', self.lower_bounds - 10*widths_lower)
        self.register_buffer('extended_upper', self.upper_bounds + 10*widths_upper)
        
        # Input scaling target range (subdomain inputs scaled to this range)
        self.register_buffer('input_range_min', torch.tensor(-1.0, device=device, dtype=dtype))
        self.register_buffer('input_range_max', torch.tensor(1.0, device=device, dtype=dtype))
        
        # Output unnormalization parameters (set by Trainer)
        self.register_buffer('output_range_min', None)
        self.register_buffer('output_range_max', None)
        
        # Create networks only for active subdomains
        if isinstance(networks, nn.Module):
            # Single network template - create copies for each active subdomain
            # Each copy gets reinitialized with a unique seed for varied weights
            import copy
            self.networks = nn.ModuleList()
            for net_idx, sub_idx in enumerate(self._active_indices):
                net_copy = copy.deepcopy(networks)
                # Reinitialize with unique seed if network has _initialize_weights
                if hasattr(net_copy, '_initialize_weights'):
                    net_copy.seed = sub_idx  # Use subdomain index for consistent seeding
                    net_copy._initialize_weights()
                self.networks.append(net_copy)
        elif isinstance(networks, (list, tuple)):
            if len(networks) != self.n_active_subdomains:
                raise ValueError(
                    f"Number of networks ({len(networks)}) must match "
                    f"number of active subdomains ({self.n_active_subdomains})"
                )
            self.networks = nn.ModuleList(networks)
        else:
            raise ValueError("networks must be an nn.Module or a list of nn.Module")
        
        # Deactivate input/output scaling on individual networks
        # (FBPINN handles scaling at the FBPINN level)
        for net in self.networks:
            if hasattr(net, 'normalize_input'):
                net.normalize_input = False
            if hasattr(net, 'unnormalize_output'):
                net.unnormalize_output = False
        
        # Check if all networks have the same architecture for batched mode
        self._use_batched = self._check_same_architecture()
        if self._use_batched:
            self._setup_batched_forward()
    
    def to(self, *args, **kwargs):
        """Override to() to update device attribute."""
        result = super().to(*args, **kwargs)
        # Update device attribute based on first buffer/parameter
        for buf in self.buffers():
            self.device = buf.device
            break
        for param in self.parameters():
            self.dtype = param.dtype
            break
        return result
    
    def _compute_active_indices(self, active_subdomains):
        """
        Compute list of active subdomain indices from various input formats.
        
        Args:
            active_subdomains: None, list of bool, or list of int
            
        Returns:
            List of active subdomain indices (sorted)
        """
        if active_subdomains is None:
            # All subdomains active
            return list(range(self.n_subdomains))
        
        if isinstance(active_subdomains, (list, tuple)):
            if len(active_subdomains) == 0:
                raise ValueError("active_subdomains cannot be empty")
            
            # Check if it's a boolean mask or list of indices
            if isinstance(active_subdomains[0], (bool, np.bool_)):
                # Boolean mask
                if len(active_subdomains) != self.n_subdomains:
                    raise ValueError(
                        f"Boolean mask length ({len(active_subdomains)}) must match "
                        f"number of subdomains ({self.n_subdomains})"
                    )
                return [i for i, active in enumerate(active_subdomains) if active]
            else:
                # List of indices
                indices = list(active_subdomains)
                for idx in indices:
                    if idx < 0 or idx >= self.n_subdomains:
                        raise ValueError(
                            f"Subdomain index {idx} out of range [0, {self.n_subdomains-1}]"
                        )
                return sorted(set(indices))
        
        raise ValueError(
            "active_subdomains must be None, list of bool, or list of int"
        )
    
    def _check_same_architecture(self):
        """Check if all networks have the same layer sizes."""
        if not all(hasattr(net, 'layer_sizes') for net in self.networks):
            return False
        
        first_sizes = self.networks[0].layer_sizes
        return all(net.layer_sizes == first_sizes for net in self.networks)
    
    def _setup_batched_forward(self):
        """Setup parameters for batched forward pass."""
        # Get layer sizes from first network
        self._layer_sizes = self.networks[0].layer_sizes
        self._n_layers = len(self._layer_sizes) - 1
        
        # Get activation function
        self._activation = self.networks[0].activation
    
    def set_output_range(self, ymin, ymax):
        """Set output unnormalization range. Called by Trainer."""
        # Use device of existing buffers to ensure consistency
        device = self.centers.device
        self.output_range_min = torch.as_tensor(ymin, dtype=self.dtype, device=device)
        self.output_range_max = torch.as_tensor(ymax, dtype=self.dtype, device=device)
    
    def compute_windows(self, x):
        """
        Compute window function values for all subdomains at given points.
        
        Delegates to the domain partition's compute_windows method.
        
        Args:
            x: Input tensor of shape (batch_size, n_dims)
            
        Returns:
            Tensor of shape (batch_size, n_subdomains) with window values
        """
        return self.domain.compute_windows(x, normalize=self.normalize_input)
    
    def scale_input(self, x, subdomain_idx):
        """
        Scale input to the specified range within a subdomain.
        
        Maps x from [lower_bounds[i], upper_bounds[i]] to [input_range_min, input_range_max]
        
        Args:
            x: Input tensor of shape (batch_size, n_dims)
            subdomain_idx: Index of the subdomain
            
        Returns:
            Scaled input tensor of shape (batch_size, n_dims)
        """
        if not self.normalize_input:
            return x
        
        lower = self.lower_bounds[subdomain_idx]  # (n_dims,)
        upper = self.upper_bounds[subdomain_idx]  # (n_dims,)
        
        # Normalize to [0, 1]
        x_normalized = (x - lower) / (upper - lower + 1e-8)
        
        # Scale to [input_range_min, input_range_max]
        x_scaled = x_normalized * (self.input_range_max - self.input_range_min) + self.input_range_min
        
        return x_scaled
    
    def _unnormalize_output(self, y):
        """
        Unnormalize output from input_range to output_range.
        
        Maps y from [input_range_min, input_range_max] to [output_range_min, output_range_max]
        
        Args:
            y: Output tensor of shape (batch_size, output_size)
            
        Returns:
            Unnormalized output tensor of shape (batch_size, output_size)
        """
        if self.output_range_min is None or not self.unnormalize_output:
            return y
        
        # Normalize to [0, 1]
        y_normalized = (y - self.input_range_min) / (self.input_range_max - self.input_range_min + 1e-8)
        
        # Scale to [output_range_min, output_range_max]
        y_scaled = y_normalized * (self.output_range_max - self.output_range_min) + self.output_range_min
        
        return y_scaled
    
    def get_point_masks(self, x):
        """
        Compute masks indicating which points are relevant for each subdomain.
        
        A point is relevant for a subdomain if it falls within the subdomain's
        extended bounds (subdomain bounds + widths on each side).
        
        Vectorized implementation for efficiency.
        
        Args:
            x: Input tensor of shape (batch_size, n_dims)
            
        Returns:
            Boolean tensor of shape (batch_size, n_subdomains) indicating which 
            points are relevant for each subdomain.
        """
        # x: (batch_size, n_dims) -> (batch_size, 1, n_dims)
        x_expanded = x.unsqueeze(1)
        
        # extended_lower/upper: (n_subdomains, n_dims) -> (1, n_subdomains, n_dims)
        ext_lower = self.extended_lower.unsqueeze(0)
        ext_upper = self.extended_upper.unsqueeze(0)
        
        # Check bounds for all subdomains at once: (batch_size, n_subdomains, n_dims)
        in_bounds = (x_expanded >= ext_lower) & (x_expanded <= ext_upper)
        
        # All dimensions must be in bounds: (batch_size, n_subdomains)
        masks = in_bounds.all(dim=-1)
        
        return masks
    
    def forward(self, x, params=None):
        """
        Forward pass: compute weighted sum of all network predictions.
        
        Uses batched computation when all networks have the same architecture,
        otherwise falls back to loop-based computation.
        
        Args:
            x: Input tensor of shape (batch_size, n_dims)
            params: Optional dict of parameters passed to output_transform
            
        Returns:
            Tensor of shape (batch_size, output_size) with weighted predictions
        """
        if self._use_batched:
            return self._forward_batched(x, params)
        else:
            return self._forward_loop(x, params)
    
    def predict(self, x_np: np.ndarray, params_dict=None) -> np.ndarray:
        """
        Predict with numpy I/O.
        
        Args:
            x_np: Input numpy array of shape (batch_size, n_inputs)
            params_dict: Optional dict passed to transforms
            
        Returns:
            Output numpy array of shape (batch_size, n_outputs)
        """
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x_np, device=next(self.parameters()).device,
                                   dtype=next(self.parameters()).dtype)
            y = self.forward(x_tensor, params_dict)
            return y.cpu().numpy()
    
    def _forward_batched(self, x, params=None):
        """
        Batched forward pass - all active networks evaluated in parallel.
        Only works when all networks have the same architecture.
        """
        # Store original x for output_transform
        x_original = x
        
        # Apply input transform (before normalization, e.g., for symmetries)
        if self.input_transform is not None:
            x = self.input_transform(x, params)
        
        # Compute window weights AFTER input_transform (so symmetry works)
        windows = self.compute_windows(x)  # (batch_size, n_subdomains)
        
        # Zero out windows for inactive subdomains
        windows = windows * self.active_mask.float().unsqueeze(0)
        
        # Scale inputs for active subdomains only
        x_scaled = self._scale_input_batched(x)  # (batch_size, n_active_subdomains, n_dims)
        
        # Batched forward pass through all active networks
        pred = self._batched_forward(x_scaled)  # (batch_size, n_active_subdomains, output_size)
        
        # Get windows for active subdomains only
        active_windows = windows[:, self._active_indices]  # (batch_size, n_active_subdomains)
        
        # Weight by windows and sum across subdomains
        weighted = pred * active_windows.unsqueeze(-1)
        output = weighted.sum(dim=1)  # (batch_size, output_size)
        
        # Unnormalize output to physical range
        output = self._unnormalize_output(output)
        
        # Apply output transform for hard constraints
        if self.output_transform is not None:
            output = self.output_transform(x_original, output, params)
        
        return output
    
    def _forward_loop(self, x, params=None):
        """
        Loop-based forward pass - evaluates each active network sequentially.
        Works with networks of different architectures.
        """
        batch_size = x.shape[0]
        
        # Store original x for output_transform
        x_original = x
        
        # Apply input transform (before normalization, e.g., for symmetries)
        if self.input_transform is not None:
            x = self.input_transform(x, params)
        
        # Compute window weights AFTER input_transform (so symmetry works)
        windows = self.compute_windows(x)  # (batch_size, n_subdomains)
        
        # Zero out windows for inactive subdomains
        windows = windows * self.active_mask.float().unsqueeze(0)
        
        # Get output size from first network
        first_sub_idx = self._active_indices[0]
        with torch.no_grad():
            x_test = self.scale_input(x[:1], first_sub_idx)
            output_size = self.networks[0](x_test).shape[-1]
        
        # Accumulate weighted outputs
        output = torch.zeros(batch_size, output_size, device=x.device, dtype=x.dtype)
        
        for net_idx, sub_idx in enumerate(self._active_indices):
            network = self.networks[net_idx]
            x_scaled = self.scale_input(x, sub_idx)
            pred = network(x_scaled)  # (batch_size, output_size)
            output = output + pred * windows[:, sub_idx:sub_idx+1]
        
        # Unnormalize output to physical range
        output = self._unnormalize_output(output)
        
        # Apply output transform for hard constraints
        if self.output_transform is not None:
            output = self.output_transform(x_original, output, params)
        
        return output
    
    def _scale_input_batched(self, x):
        """
        Scale input to all active subdomain ranges at once.
        
        Args:
            x: Input tensor of shape (batch_size, n_dims)
            
        Returns:
            Scaled inputs of shape (batch_size, n_active_subdomains, n_dims)
        """
        if not self.normalize_input:
            # Repeat x for each active subdomain
            return x.unsqueeze(1).expand(-1, self.n_active_subdomains, -1)
        
        # x: (batch_size, n_dims) -> (batch_size, 1, n_dims)
        x_expanded = x.unsqueeze(1)
        
        # Get bounds for active subdomains only
        # lower_bounds, upper_bounds: (n_active_subdomains, n_dims) -> (1, n_active_subdomains, n_dims)
        lower = self.lower_bounds[self._active_indices].unsqueeze(0)
        upper = self.upper_bounds[self._active_indices].unsqueeze(0)
        
        # Normalize to [0, 1] then scale to [-1, 1]
        x_normalized = (x_expanded - lower) / (upper - lower + 1e-8)
        x_scaled = x_normalized * (self.input_range_max - self.input_range_min) + self.input_range_min
        
        return x_scaled  # (batch_size, n_active_subdomains, n_dims)
    
    def _batched_forward(self, x):
        """
        Batched forward pass through all active networks using stacked weights.
        
        Args:
            x: Input tensor of shape (batch_size, n_active_subdomains, n_dims)
            
        Returns:
            Output tensor of shape (batch_size, n_active_subdomains, output_size)
        """
        # x: (batch_size, n_active_subdomains, n_dims)
        h = x
        
        for layer_idx in range(self._n_layers):
            # Stack weights on-the-fly to ensure gradients flow correctly
            W = torch.stack([
                net.layers[layer_idx].weight for net in self.networks
            ], dim=0)  # (n_active_subdomains, out_features, in_features)
            b = torch.stack([
                net.layers[layer_idx].bias for net in self.networks
            ], dim=0)  # (n_subdomains, out_features)
            
            # Batched matrix multiplication:
            # h: (batch_size, n_subdomains, in_features)
            # W: (n_subdomains, out_features, in_features)
            # Result: (batch_size, n_subdomains, out_features)
            h = torch.einsum('bsi,soi->bso', h, W) + b.unsqueeze(0)
            
            # Apply activation (except for last layer)
            if layer_idx < self._n_layers - 1:
                h = self._activation(h)
        
        return h  # (batch_size, n_active_subdomains, output_size)
    
    def get_subdomain_predictions(self, x):
        """
        Get individual predictions from each active subdomain network.
        
        Args:
            x: Input tensor of shape (batch_size, n_dims)
            
        Returns:
            tuple: (predictions, windows, active_indices)
                - predictions: Tensor of shape (batch_size, n_active_subdomains, output_size)
                - windows: Tensor of shape (batch_size, n_subdomains) - full window weights
                - active_indices: List of active subdomain indices
        """
        windows = self.compute_windows(x)
        
        # Apply input transform if present
        x_for_net = x
        if self.input_transform is not None:
            x_for_net = self.input_transform(x)
        
        predictions = []
        for net_idx, sub_idx in enumerate(self._active_indices):
            network = self.networks[net_idx]
            # Scale input to subdomain range
            x_scaled = self.scale_input(x_for_net, sub_idx)
            pred = network(x_scaled)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=1)
        
        return predictions, windows, self._active_indices

    def precompute_training_data(self, x: torch.Tensor, threshold: float = 1e-6,
                                  params=None) -> Dict:
        """
        Precompute sparse training data for efficient forward passes.
        
        Computes which points are "active" (window > threshold) for each subdomain,
        and precomputes scaled inputs and window weights. This data can be reused
        across all training epochs since collocation points don't change.
        
        Args:
            x: Input tensor of shape (batch_size, n_dims)
            threshold: Minimum window value to consider a point active (default: 1e-6)
            params: Optional dict passed to input transform
        
        Returns:
            Dict containing precomputed data for forward_precomputed()
        """
        x_original = x
        
        # Apply input transform if present
        if self.input_transform is not None:
            x = self.input_transform(x, params)
        
        batch_size = x.shape[0]
        
        # Compute windows for all points: (batch_size, n_subdomains)
        windows = self.compute_windows(x)
        
        # For each active subdomain, find points where window > threshold
        precomputed = {
            'batch_size': batch_size,
            'x_original': x_original,
            'subdomains': {},
            'device': x.device,
            'dtype': x.dtype,
        }
        
        for net_idx, sub_idx in enumerate(self._active_indices):
            w = windows[:, sub_idx]
            mask = w > threshold
            indices = torch.nonzero(mask, as_tuple=True)[0]
            
            if len(indices) > 0:
                # Get active points and their window weights
                x_active = x[indices]
                weights_active = w[indices].unsqueeze(1)  # (n_active_pts, 1)
                
                # Pre-scale inputs for this subdomain
                x_scaled = self.scale_input(x_active, sub_idx)
                
                precomputed['subdomains'][net_idx] = {
                    'indices': indices,           # Which points to process
                    'x_scaled': x_scaled,         # Pre-scaled inputs
                    'weights': weights_active,    # Window weights
                    'n_points': len(indices),
                }
        
        return precomputed
    
    def forward_precomputed(self, precomputed: Dict, params=None) -> torch.Tensor:
        """
        Forward pass using precomputed sparse training data.
        
        Only evaluates points that have non-negligible window values for each
        subdomain, significantly reducing computation for sparse window overlaps.
        
        Args:
            precomputed: Output from precompute_training_data()
            params: Optional dict passed to output transform
        
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch_size = precomputed['batch_size']
        x_original = precomputed['x_original']
        device = precomputed['device']
        dtype = precomputed['dtype']
        
        # Get output size from first network
        output_size = self.networks[0].layers[-1].out_features
        
        # Initialize output
        output = torch.zeros(batch_size, output_size, device=device, dtype=dtype)
        
        # Process each subdomain's active points
        for net_idx, data in precomputed['subdomains'].items():
            indices = data['indices']
            x_scaled = data['x_scaled']
            weights = data['weights']
            
            # Forward pass through this subdomain
            network = self.networks[net_idx]
            pred = network(x_scaled)
            
            # Weight by window and scatter back
            weighted_pred = pred * weights
            output.index_add_(0, indices, weighted_pred)
        
        # Unnormalize output
        output = self._unnormalize_output(output)
        
        # Apply output transform
        if self.output_transform is not None:
            output = self.output_transform(x_original, output, params)
        
        return output
