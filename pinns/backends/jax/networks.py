"""
JAX/Flax implementation of neural networks for PINNS.

Provides FNN (Fully-connected Neural Network) and FBPINN (Finite Basis PINN)
implemented using Flax's linen module.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.core import freeze, unfreeze
from typing import Sequence, Callable, Optional, Tuple, List, Dict, Any
from dataclasses import field


def get_activation(name: str) -> Callable:
    """Get activation function by name."""
    activations = {
        'relu': nn.relu,
        'tanh': nn.tanh,
        'sigmoid': nn.sigmoid,
        'gelu': nn.gelu,
        'silu': nn.silu,
        'leaky_relu': nn.leaky_relu,
        'elu': nn.elu,
        'softplus': nn.softplus,
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
    return activations[name.lower()]


class FourierFeatures:
    """
    Random Fourier Feature encoding to mitigate spectral bias.
    
    Maps input coordinates into high-frequency signals before passing through MLP.
    Based on Tancik et al. "Fourier Features Let Networks Learn High Frequency
    Functions in Low Dimensional Domains" (NeurIPS 2020).
    
    The encoding γ: R^d → R^(2m) is defined by:
        γ(x) = [cos(2π B x), sin(2π B x)]
    where B ∈ R^(m×d) has entries sampled from N(0, σ²).
    
    JIT-compatible: B matrix is a constant JAX array created at initialization.
    
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
        key = jax.random.PRNGKey(seed)
        self.B = jax.random.normal(key, (n_features, input_dim)) * sigma
        
        # Output dimension
        self.output_dim = 2 * n_features + (input_dim if include_input else 0)
    
    def __call__(self, x: jnp.ndarray, params_dict: Optional[Dict] = None) -> jnp.ndarray:
        """
        Apply Fourier feature encoding.
        
        Args:
            x: Input array of shape (batch_size, input_dim)
            params_dict: Ignored (for API compatibility with input_transform)
        
        Returns:
            Encoded array of shape (batch_size, output_dim)
        """
        # x: (batch, input_dim)
        # B: (n_features, input_dim)
        # Bx = x @ B.T: (batch, n_features)
        Bx = x @ self.B.T
        
        # γ(x) = [cos(2π Bx), sin(2π Bx)]
        cos_features = jnp.cos(2 * jnp.pi * Bx)
        sin_features = jnp.sin(2 * jnp.pi * Bx)
        features = jnp.concatenate([cos_features, sin_features], axis=-1)
        
        # Optionally include original input
        if self.include_input:
            features = jnp.concatenate([x, features], axis=-1)
        
        return features
    
    def transform(self, x: jnp.ndarray, params_dict: Optional[Dict] = None) -> jnp.ndarray:
        """Alias for __call__ for explicit usage."""
        return self.__call__(x, params_dict)


class DenseRWF(nn.Module):
    """
    Dense layer with Random Weight Factorization (RWF).
    
    Implements W = diag(exp(s)) · V where s and V are trainable.
    Based on Wang et al. "On the eigenvector bias of Fourier feature networks".
    
    Attributes:
        features: Number of output features
        rwf_mu: Mean for initializing s from N(mu, sigma*I). Recommended: 0.5 or 1.0
        rwf_sigma: Std for initializing s from N(mu, sigma*I). Recommended: 0.1
    """
    features: int
    rwf_mu: float = 0.5
    rwf_sigma: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass: y = (diag(exp(s)) · V) @ x + b"""
        input_features = x.shape[-1]
        
        # Initialize V using Glorot/Xavier
        V = self.param(
            'V',
            nn.initializers.glorot_normal(),
            (self.features, input_features)
        )
        
        # Initialize s from N(mu, sigma*I)
        s = self.param(
            's',
            lambda key, shape: self.rwf_mu + self.rwf_sigma * jax.random.normal(key, shape),
            (self.features,)
        )
        
        # Bias
        b = self.param(
            'b',
            nn.initializers.zeros,
            (self.features,)
        )
        
        # Compute W = diag(exp(s)) · V
        # exp(s): (features,) -> (features, 1) for broadcasting
        W = jnp.exp(s)[:, None] * V  # (features, input_features)
        
        # Linear transform: y = x @ W.T + b
        return x @ W.T + b


class FNNModule(nn.Module):
    """
    Internal Flax module for FNN.
    
    This is the actual neural network - use FNN class for the full API.
    """
    layer_sizes: Sequence[int]
    activation: str = 'tanh'
    output_activation: Optional[str] = None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the network (raw, no normalization)."""
        act_fn = get_activation(self.activation)
        
        # Hidden layers
        for i, size in enumerate(self.layer_sizes[1:-1]):
            x = nn.Dense(size, name=f'hidden_{i}')(x)
            x = act_fn(x)
        
        # Output layer
        x = nn.Dense(self.layer_sizes[-1], name='output')(x)
        
        # Output activation
        if self.output_activation is not None:
            out_act = get_activation(self.output_activation)
            x = out_act(x)
        
        return x


class WFFNNModule(nn.Module):
    """
    Internal Flax module for Weight-Factorized FNN.
    
    Uses Random Weight Factorization (RWF) where W = diag(exp(s)) · V.
    Based on Wang et al. "On the eigenvector bias of Fourier feature networks".
    """
    layer_sizes: Sequence[int]
    activation: str = 'tanh'
    output_activation: Optional[str] = None
    rwf_mu: float = 0.5
    rwf_sigma: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the network (raw, no normalization)."""
        act_fn = get_activation(self.activation)
        
        # Hidden layers with RWF
        for i, size in enumerate(self.layer_sizes[1:-1]):
            x = DenseRWF(size, rwf_mu=self.rwf_mu, rwf_sigma=self.rwf_sigma, name=f'hidden_{i}')(x)
            x = act_fn(x)
        
        # Output layer with RWF
        x = DenseRWF(self.layer_sizes[-1], rwf_mu=self.rwf_mu, rwf_sigma=self.rwf_sigma, name='output')(x)
        
        # Output activation
        if self.output_activation is not None:
            out_act = get_activation(self.output_activation)
            x = out_act(x)
        
        return x


class FNN:
    """
    Fully-connected Neural Network for JAX.
    
    Wrapper class providing PyTorch-compatible API with input/output normalization.
    
    Example:
        model = FNN([2, 64, 64, 1], activation='tanh', normalize_input=True)
        params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 2)))
        y = model.apply(params, x)
    """
    
    def __init__(self, 
                 layer_sizes: Sequence[int],
                 activation: str = 'tanh',
                 output_activation: Optional[str] = None,
                 normalize_input: bool = True,
                 unnormalize_output: bool = True,
                 input_transform: Optional[Callable] = None,
                 output_transform: Optional[Callable] = None,
                 feature_encoding: Optional[Callable] = None):
        """
        Initialize FNN wrapper.
        
        Args:
            layer_sizes: Network architecture [input, hidden..., output]
                         Note: If using feature_encoding, layer_sizes[0] should match
                         the encoding's output_dim, not the original input dimension.
            activation: Activation function name
            output_activation: Optional output activation
            normalize_input: Whether to normalize inputs to [-1, 1]
            unnormalize_output: Whether to unnormalize outputs
            input_transform: Optional symmetry transform (applied before normalization)
            output_transform: Optional hard constraint transform
            feature_encoding: Optional feature encoding (e.g., FourierFeatures).
                             Applied after normalization, before network forward pass.
        """
        self.layer_sizes = list(layer_sizes)
        self.activation = activation
        self.output_activation = output_activation
        self.normalize_input = normalize_input
        self.unnormalize_output = unnormalize_output
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.feature_encoding = feature_encoding
        
        # Bounds (set by trainer)
        self.input_min = None
        self.input_max = None
        self.output_min = None
        self.output_max = None
        
        # Internal Flax module
        self._module = FNNModule(
            layer_sizes=layer_sizes,
            activation=activation,
            output_activation=output_activation
        )
    
    def init(self, rng: jax.random.PRNGKey, dummy_input: jnp.ndarray = None) -> Dict:
        """Initialize network parameters."""
        if dummy_input is None:
            dummy_input = jnp.ones((1, self.layer_sizes[0]))
        return self._module.init(rng, dummy_input)
    
    def set_input_range(self, xmin: np.ndarray, xmax: np.ndarray):
        """Set input normalization range."""
        self.input_min = jnp.array(xmin)
        self.input_max = jnp.array(xmax)
    
    def set_output_range(self, ymin: np.ndarray, ymax: np.ndarray):
        """Set output unnormalization range."""
        self.output_min = jnp.array(ymin)
        self.output_max = jnp.array(ymax)
    
    def apply(self, params: Dict, x: jnp.ndarray, params_dict: Optional[Dict] = None) -> jnp.ndarray:
        """
        Forward pass through the network.
        
        Args:
            params: Network parameters from init()
            x: Input tensor of shape (batch_size, n_inputs)
            params_dict: Optional params passed to transforms
        
        Returns:
            Output tensor of shape (batch_size, n_outputs)
        """
        x_original = x
        
        # Apply input transform (e.g., symmetry)
        if self.input_transform is not None:
            x = self.input_transform(x, params_dict)
        
        # Normalize input to [-1, 1]
        if self.normalize_input and self.input_min is not None:
            x = 2.0 * (x - self.input_min) / (self.input_max - self.input_min + 1e-8) - 1.0
        
        # Apply feature encoding (e.g., Fourier features)
        if self.feature_encoding is not None:
            x = self.feature_encoding(x, params_dict)
        
        # Forward through network
        y = self._module.apply(params, x)
        
        # Unnormalize output
        if self.unnormalize_output and self.output_min is not None:
            y = (y + 1.0) / 2.0 * (self.output_max - self.output_min) + self.output_min
        
        # Apply output transform (e.g., hard constraints)
        if self.output_transform is not None:
            y = self.output_transform(x_original, y, params_dict)
        
        return y
    
    def forward(self, x: jnp.ndarray, params_dict: Optional[Dict] = None) -> jnp.ndarray:
        """
        Forward pass using stored parameters.
        
        Provides PyTorch-compatible API for inference.
        
        Args:
            x: Input tensor of shape (batch_size, n_inputs)
            params_dict: Optional dict passed to transforms
        
        Returns:
            Output tensor of shape (batch_size, n_outputs)
        """
        return self.apply(self.params, x, params_dict)
    
    def predict(self, x_np: np.ndarray, params_dict: Optional[Dict] = None) -> np.ndarray:
        """
        Predict with numpy I/O.
        
        Args:
            x_np: Input numpy array of shape (batch_size, n_inputs)
            params_dict: Optional dict passed to transforms
            
        Returns:
            Output numpy array of shape (batch_size, n_outputs)
        """
        x_jax = jnp.array(x_np)
        y = self.forward(x_jax, params_dict)
        return np.array(y)
    
    def to(self, device: str = None, dtype=None, seed: int = 0) -> 'FNN':
        """
        Initialize parameters and move to specified device.
        
        This provides PyTorch-compatible API. In JAX, device placement is handled
        automatically, but this method initializes params and stores them.
        
        Args:
            device: Device string ('cpu', 'gpu', 'tpu'). Default: auto-detect.
            dtype: Data type (e.g., jnp.float32). Default: jnp.float32.
            seed: Random seed for initialization.
        
        Returns:
            self (for method chaining)
        """
        # Auto-detect device if not specified
        if device is None:
            device = jax.devices()[0].platform  # 'cpu', 'gpu', 'tpu'
        
        # Default dtype
        if dtype is None:
            dtype = jnp.float32
        
        self.device = device
        self.dtype = dtype
        
        # Initialize parameters on the default device (JAX handles placement)
        dummy_input = jnp.ones((1, self.layer_sizes[0]), dtype=dtype)
        self.params = self._module.init(jax.random.PRNGKey(seed), dummy_input)
        
        return self


class WFFNN:
    """
    Weight-Factorized Fully-connected Neural Network for JAX.
    
    Uses Random Weight Factorization (RWF): W = diag(exp(s)) · V
    Based on Wang et al. "On the eigenvector bias of Fourier feature networks".
    
    Recommended settings: rwf_mu=0.5 or 1.0, rwf_sigma=0.1
    
    Example:
        model = WFFNN([2, 64, 64, 1], activation='tanh', rwf_mu=0.5, rwf_sigma=0.1)
        params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 2)))
        y = model.apply(params, x)
    """
    
    def __init__(self, 
                 layer_sizes: Sequence[int],
                 activation: str = 'tanh',
                 output_activation: Optional[str] = None,
                 normalize_input: bool = True,
                 unnormalize_output: bool = True,
                 input_transform: Optional[Callable] = None,
                 output_transform: Optional[Callable] = None,
                 feature_encoding: Optional[Callable] = None,
                 rwf_mu: float = 0.5,
                 rwf_sigma: float = 0.1):
        """
        Initialize Weight-Factorized FNN wrapper.
        
        Args:
            layer_sizes: Network architecture [input, hidden..., output]
            activation: Activation function name
            output_activation: Optional output activation
            normalize_input: Whether to normalize inputs to [-1, 1]
            unnormalize_output: Whether to unnormalize outputs
            input_transform: Optional symmetry transform
            output_transform: Optional hard constraint transform
            feature_encoding: Optional feature encoding (e.g., FourierFeatures)
            rwf_mu: Mean for initializing s ~ N(mu, sigma*I). Recommended: 0.5 or 1.0
            rwf_sigma: Std for initializing s ~ N(mu, sigma*I). Recommended: 0.1
        """
        self.layer_sizes = list(layer_sizes)
        self.activation = activation
        self.output_activation = output_activation
        self.normalize_input = normalize_input
        self.unnormalize_output = unnormalize_output
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.feature_encoding = feature_encoding
        self.rwf_mu = rwf_mu
        self.rwf_sigma = rwf_sigma
        
        # Bounds (set by trainer)
        self.input_min = None
        self.input_max = None
        self.output_min = None
        self.output_max = None
        
        # Internal Flax module with weight factorization
        self._module = WFFNNModule(
            layer_sizes=layer_sizes,
            activation=activation,
            output_activation=output_activation,
            rwf_mu=rwf_mu,
            rwf_sigma=rwf_sigma
        )
    
    def init(self, rng: jax.random.PRNGKey, dummy_input: jnp.ndarray = None) -> Dict:
        """Initialize network parameters."""
        if dummy_input is None:
            dummy_input = jnp.ones((1, self.layer_sizes[0]))
        return self._module.init(rng, dummy_input)
    
    def set_input_range(self, xmin: np.ndarray, xmax: np.ndarray):
        """Set input normalization range."""
        self.input_min = jnp.array(xmin)
        self.input_max = jnp.array(xmax)
    
    def set_output_range(self, ymin: np.ndarray, ymax: np.ndarray):
        """Set output unnormalization range."""
        self.output_min = jnp.array(ymin)
        self.output_max = jnp.array(ymax)
    
    def apply(self, params: Dict, x: jnp.ndarray, params_dict: Optional[Dict] = None) -> jnp.ndarray:
        """
        Forward pass through the network.
        
        Args:
            params: Network parameters from init()
            x: Input tensor of shape (batch_size, n_inputs)
            params_dict: Optional params passed to transforms
        
        Returns:
            Output tensor of shape (batch_size, n_outputs)
        """
        x_original = x
        
        # Apply input transform (e.g., symmetry)
        if self.input_transform is not None:
            x = self.input_transform(x, params_dict)
        
        # Normalize input to [-1, 1]
        if self.normalize_input and self.input_min is not None:
            x = 2.0 * (x - self.input_min) / (self.input_max - self.input_min + 1e-8) - 1.0
        
        # Apply feature encoding (e.g., Fourier features)
        if self.feature_encoding is not None:
            x = self.feature_encoding(x, params_dict)
        
        # Forward through network
        y = self._module.apply(params, x)
        
        # Unnormalize output
        if self.unnormalize_output and self.output_min is not None:
            y = (y + 1.0) / 2.0 * (self.output_max - self.output_min) + self.output_min
        
        # Apply output transform (e.g., hard constraints)
        if self.output_transform is not None:
            y = self.output_transform(x_original, y, params_dict)
        
        return y
    
    def forward(self, x: jnp.ndarray, params_dict: Optional[Dict] = None) -> jnp.ndarray:
        """Forward pass using stored parameters."""
        return self.apply(self.params, x, params_dict)
    
    def predict(self, x_np: np.ndarray, params_dict: Optional[Dict] = None) -> np.ndarray:
        """Predict with numpy I/O."""
        x_jax = jnp.array(x_np)
        y = self.forward(x_jax, params_dict)
        return np.array(y)
    
    def to(self, device: str = None, dtype=None, seed: int = 0) -> 'WFFNN':
        """
        Initialize parameters and move to specified device.
        
        Args:
            device: Device string ('cpu', 'gpu', 'tpu'). Default: auto-detect.
            dtype: Data type. Default: jnp.float32.
            seed: Random seed for initialization.
        
        Returns:
            self (for method chaining)
        """
        if device is None:
            device = jax.devices()[0].platform
        
        if dtype is None:
            dtype = jnp.float32
        
        self.device = device
        self.dtype = dtype
        
        dummy_input = jnp.ones((1, self.layer_sizes[0]), dtype=dtype)
        self.params = self._module.init(jax.random.PRNGKey(seed), dummy_input)
        
        return self


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
    hidden_dim: int
    activation: str = 'tanh'
    rwf_mu: float = 0.5
    rwf_sigma: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, U: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through one residual block."""
        act_fn = get_activation(self.activation)
        
        # Trainable α parameter (initialized to 0 for identity at init)
        alpha = self.param('alpha', nn.initializers.zeros, ())
        
        # First dense + gating (using RWF)
        f = act_fn(DenseRWF(self.hidden_dim, rwf_mu=self.rwf_mu, rwf_sigma=self.rwf_sigma, name='dense1')(x))
        z1 = f * U + (1 - f) * V
        
        # Second dense + gating (using RWF)
        g = act_fn(DenseRWF(self.hidden_dim, rwf_mu=self.rwf_mu, rwf_sigma=self.rwf_sigma, name='dense2')(z1))
        z2 = g * U + (1 - g) * V
        
        # Third dense (using RWF)
        h = act_fn(DenseRWF(self.hidden_dim, rwf_mu=self.rwf_mu, rwf_sigma=self.rwf_sigma, name='dense3')(z2))
        
        # Adaptive residual connection
        x_next = alpha * h + (1 - alpha) * x
        
        return x_next


class PirateNetModule(nn.Module):
    """
    Internal Flax module for PirateNet.
    
    Physics-Informed Residual AdapTivE Networks (PirateNet) with Random Weight Factorization.
    Based on Wang et al. "PirateNets: Physics-informed Deep Learning 
    with Residual Adaptive Networks".
    """
    input_dim: int
    output_dim: int
    hidden_dim: int
    n_blocks: int = 3
    activation: str = 'tanh'
    rwf_mu: float = 0.5
    rwf_sigma: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through PirateNet."""
        act_fn = get_activation(self.activation)
        
        # Step 1: Compute U and V gates from input (using RWF)
        U = act_fn(DenseRWF(self.hidden_dim, rwf_mu=self.rwf_mu, rwf_sigma=self.rwf_sigma, name='U_layer')(x))
        V = act_fn(DenseRWF(self.hidden_dim, rwf_mu=self.rwf_mu, rwf_sigma=self.rwf_sigma, name='V_layer')(x))
        
        # Step 2: Project input to hidden_dim for residual blocks (using RWF)
        h = DenseRWF(self.hidden_dim, rwf_mu=self.rwf_mu, rwf_sigma=self.rwf_sigma, name='input_projection')(x)
        
        # Step 3: Apply L residual blocks
        for i in range(self.n_blocks):
            h = PirateNetBlock(
                hidden_dim=self.hidden_dim,
                activation=self.activation,
                rwf_mu=self.rwf_mu,
                rwf_sigma=self.rwf_sigma,
                name=f'block_{i}'
            )(h, U, V)
        
        # Step 4: Final output layer (using RWF)
        output = DenseRWF(self.output_dim, rwf_mu=self.rwf_mu, rwf_sigma=self.rwf_sigma, name='output')(h)
        
        return output


class PirateNet:
    """
    Physics-Informed Residual AdapTivE Network (PirateNet) for JAX.
    
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
    
    Example:
        # Without Fourier features
        model = PirateNet(input_dim=2, output_dim=1, hidden_dim=64, n_blocks=3)
        
        # With Fourier features (input_dim is original dimension, not encoded)
        fourier = FourierFeatures(input_dim=2, n_features=64, sigma=1.0)
        model = PirateNet(input_dim=2, output_dim=1, hidden_dim=64, 
                         feature_encoding=fourier)
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 n_blocks: int = 3,
                 activation: str = 'tanh',
                 normalize_input: bool = True,
                 unnormalize_output: bool = True,
                 input_transform: Optional[Callable] = None,
                 output_transform: Optional[Callable] = None,
                 feature_encoding: Optional[Callable] = None,
                 rwf_mu: float = 0.5,
                 rwf_sigma: float = 0.1):
        """
        Initialize PirateNet.
        
        Args:
            input_dim: Dimension of input coordinates (before any feature encoding)
            output_dim: Dimension of output
            hidden_dim: Width of all hidden layers (must be consistent)
            n_blocks: Number of residual blocks (default: 3). Total depth = 3*n_blocks.
            activation: Activation function name (default: 'tanh')
            normalize_input: Whether to normalize inputs to [-1, 1]
            unnormalize_output: Whether to unnormalize outputs
            input_transform: Optional symmetry transform (applied before normalization)
            output_transform: Optional hard constraint transform
            feature_encoding: Optional feature encoding (e.g., FourierFeatures).
                             Applied after normalization, before network forward pass.
            rwf_mu: Mean for Random Weight Factorization s initialization (default: 0.5)
            rwf_sigma: Std for Random Weight Factorization s initialization (default: 0.1)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.activation = activation
        self.normalize_input = normalize_input
        self.unnormalize_output = unnormalize_output
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.feature_encoding = feature_encoding
        self.rwf_mu = rwf_mu
        self.rwf_sigma = rwf_sigma
        
        # Bounds (set by trainer)
        self.input_min = None
        self.input_max = None
        self.output_min = None
        self.output_max = None
        
        # For compatibility with FNN interface
        self.layer_sizes = [input_dim, hidden_dim, output_dim]
        
        # Internal module
        self._module = PirateNetModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            activation=activation,
            rwf_mu=rwf_mu,
            rwf_sigma=rwf_sigma
        )
    
    def init(self, rng: jax.random.PRNGKey, dummy_input: jnp.ndarray = None) -> Dict:
        """Initialize network parameters."""
        if dummy_input is None:
            dummy_input = jnp.ones((1, self.input_dim))
        
        # Apply same transforms as in apply() to get correct input shape
        if self.feature_encoding is not None:
            dummy_input = self.feature_encoding(dummy_input, None)
        
        return self._module.init(rng, dummy_input)
    
    def set_input_range(self, xmin: np.ndarray, xmax: np.ndarray):
        """Set input normalization range."""
        self.input_min = jnp.array(xmin)
        self.input_max = jnp.array(xmax)
    
    def set_output_range(self, ymin: np.ndarray, ymax: np.ndarray):
        """Set output unnormalization range."""
        self.output_min = jnp.array(ymin)
        self.output_max = jnp.array(ymax)
    
    def apply(self, params: Dict, x: jnp.ndarray, params_dict: Optional[Dict] = None) -> jnp.ndarray:
        """
        Forward pass through the network.
        
        Args:
            params: Network parameters from init()
            x: Input tensor of shape (batch_size, input_dim)
            params_dict: Optional params passed to transforms
        
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        x_original = x
        
        # Apply input transform (e.g., symmetry)
        if self.input_transform is not None:
            x = self.input_transform(x, params_dict)
        
        # Normalize input to [-1, 1]
        if self.normalize_input and self.input_min is not None:
            x = 2.0 * (x - self.input_min) / (self.input_max - self.input_min + 1e-8) - 1.0
        
        # Apply external feature encoding if provided
        if self.feature_encoding is not None:
            x = self.feature_encoding(x, params_dict)
        
        # Forward through network
        y = self._module.apply(params, x)
        
        # Unnormalize output
        if self.unnormalize_output and self.output_min is not None:
            y = (y + 1.0) / 2.0 * (self.output_max - self.output_min) + self.output_min
        
        # Apply output transform (e.g., hard constraints)
        if self.output_transform is not None:
            y = self.output_transform(x_original, y, params_dict)
        
        return y
    
    def forward(self, x: jnp.ndarray, params_dict: Optional[Dict] = None) -> jnp.ndarray:
        """Forward pass using stored parameters."""
        return self.apply(self.params, x, params_dict)
    
    def predict(self, x_np: np.ndarray, params_dict: Optional[Dict] = None) -> np.ndarray:
        """Predict with numpy I/O."""
        x_jax = jnp.array(x_np)
        y = self.forward(x_jax, params_dict)
        return np.array(y)
    
    def to(self, device: str = None, dtype=None, seed: int = 0) -> 'PirateNet':
        """
        Initialize parameters and move to specified device.
        
        Args:
            device: Device string ('cpu', 'gpu', 'tpu'). Default: auto-detect.
            dtype: Data type. Default: jnp.float32.
            seed: Random seed for initialization.
        
        Returns:
            self (for method chaining)
        """
        if device is None:
            device = jax.devices()[0].platform
        
        if dtype is None:
            dtype = jnp.float32
        
        self.device = device
        self.dtype = dtype
        
        dummy_input = jnp.ones((1, self.input_dim), dtype=dtype)
        
        # Apply same transforms as in apply() to get correct input shape
        if self.feature_encoding is not None:
            dummy_input = self.feature_encoding(dummy_input, None)
        
        self.params = self._module.init(jax.random.PRNGKey(seed), dummy_input)
        
        return self


def create_fnn(layer_sizes: Sequence[int], 
               activation: str = 'tanh',
               seed: int = 0) -> Tuple[FNN, Dict]:
    """
    Create an FNN model and initialize its parameters.
    
    Args:
        layer_sizes: Network architecture [input, hidden..., output]
        activation: Activation function name
        seed: Random seed for initialization
    
    Returns:
        Tuple of (model, params)
        
    Example:
        model, params = create_fnn([2, 64, 64, 1], activation='tanh', seed=42)
        y = model.apply(params, x)
    """
    model = FNN(layer_sizes=layer_sizes, activation=activation)
    dummy_input = jnp.ones((1, layer_sizes[0]))
    params = model.init(jax.random.PRNGKey(seed), dummy_input)
    return model, params


class FBPINNModule(nn.Module):
    """
    Finite Basis Physics-Informed Neural Network (FBPINN) - single subdomain network.
    
    This module represents ONE subdomain's network. The full FBPINN combines
    multiple such networks with window functions.
    """
    layer_sizes: Sequence[int]
    activation: str = 'tanh'
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass for single subdomain."""
        act_fn = get_activation(self.activation)
        
        for i, size in enumerate(self.layer_sizes[1:-1]):
            x = nn.Dense(size, name=f'hidden_{i}')(x)
            x = act_fn(x)
        
        x = nn.Dense(self.layer_sizes[-1], name='output')(x)
        return x


class FBPINN:
    """
    Finite Basis Physics-Informed Neural Network (FBPINN) for JAX.
    
    This is a functional implementation that combines multiple subnet predictions
    using smooth window functions (partition of unity).
    
    Unlike PyTorch's nn.Module, this is a pure Python class that holds configuration
    and provides methods to create params and compute forward passes.
    
    Attributes:
        domain: DomainCubicPartition defining subdomain geometry
        layer_sizes: Network architecture for each subdomain
        activation: Activation function name
        active_indices: List of active subdomain indices
        n_active: Number of active subdomains
    
    Example:
        fbpinn = FBPINN(domain, layer_sizes=[3, 64, 3], activation='tanh')
        params = fbpinn.init(jax.random.PRNGKey(0))
        y = fbpinn.apply(params, x)
    """
    
    def __init__(self, 
                 domain,
                 layer_sizes_or_network,
                 activation: str = 'tanh',
                 active_subdomains: Optional[List] = None,
                 normalize_input: bool = True,
                 unnormalize_output: bool = True,
                 input_transform: Optional[Callable] = None,
                 output_transform: Optional[Callable] = None):
        """
        Initialize FBPINN.
        
        Args:
            domain: DomainCubicPartition with subdomain geometry
            layer_sizes_or_network: Network architecture [input, hidden..., output] or FNN instance
            activation: Activation function name (ignored if FNN passed)
            active_subdomains: None (all active), boolean mask, or index list
            normalize_input: Whether to normalize inputs per subdomain
            unnormalize_output: Whether to unnormalize outputs
            input_transform: Optional symmetry transform
            output_transform: Optional hard constraint transform
        """
        self.domain = domain
        
        # Accept either layer_sizes list or FNN instance
        if isinstance(layer_sizes_or_network, FNN):
            base_network = layer_sizes_or_network
            self.layer_sizes = base_network.layer_sizes
            self.activation = base_network.activation
        else:
            self.layer_sizes = list(layer_sizes_or_network)
            self.activation = activation
        
        self.normalize_input = normalize_input
        self.unnormalize_output = unnormalize_output
        self.input_transform = input_transform
        self.output_transform = output_transform
        
        self.n_subdomains = len(domain)
        self.n_dims = domain.n_dims
        
        # Compute active subdomain indices
        self.active_indices = self._compute_active_indices(active_subdomains)
        self.n_active = len(self.active_indices)
        
        # Get subdomain geometry as numpy arrays
        centers, widths_lower, widths_upper = domain.to_numpy()
        self.centers = centers
        self.widths_lower = widths_lower
        self.widths_upper = widths_upper
        
        lower_bounds, upper_bounds = domain.get_subdomain_bounds()
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        
        # Create active mask
        self.active_mask = np.zeros(self.n_subdomains, dtype=bool)
        self.active_mask[self.active_indices] = True
        
        # Output range (set by trainer)
        self.output_min = None
        self.output_max = None
    
    def _compute_active_indices(self, active_subdomains) -> List[int]:
        """Compute list of active subdomain indices."""
        if active_subdomains is None:
            return list(range(self.n_subdomains))
        
        if isinstance(active_subdomains, (list, tuple)):
            if len(active_subdomains) == 0:
                raise ValueError("active_subdomains cannot be empty")
            
            if isinstance(active_subdomains[0], (bool, np.bool_)):
                # Boolean mask
                return [i for i, active in enumerate(active_subdomains) if active]
            else:
                # Index list
                return sorted(set(active_subdomains))
        
        raise ValueError("active_subdomains must be None, list of bool, or list of int")
    
    def init(self, rng: jax.random.PRNGKey, dummy_input=None) -> Dict:
        """
        Initialize parameters for all active subdomain networks.
        
        Args:
            rng: JAX random key
            dummy_input: Ignored (for API compatibility with FNN)
        
        Returns:
            Dict containing params for each active subdomain
        """
        params = {}
        dummy = jnp.ones((1, self.layer_sizes[0]))
        
        for i, sub_idx in enumerate(self.active_indices):
            # Split key for each subdomain
            rng, subkey = jax.random.split(rng)
            
            # Create and init subdomain network
            subnet = FBPINNModule(
                layer_sizes=self.layer_sizes,
                activation=self.activation
            )
            subnet_params = subnet.init(subkey, dummy)
            params[f'subnet_{sub_idx}'] = subnet_params
        
        return freeze(params)
    
    def set_eval_mode(self, mode: str):
        """Deprecated: sparse mode removed. This method does nothing."""
        pass  # No-op for backwards compatibility
    
    def set_output_range(self, ymin: np.ndarray, ymax: np.ndarray):
        """Set output unnormalization range."""
        self.output_min = jnp.array(ymin)
        self.output_max = jnp.array(ymax)
    
    def compute_windows(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute window function values for all subdomains.
        
        Uses sigmoid-based bump functions (same as PyTorch version).
        
        Args:
            x: Input array of shape (batch_size, n_dims)
        
        Returns:
            Window values of shape (batch_size, n_subdomains)
        """
        # Get subdomain bounds
        lower_bounds = jnp.array(self.lower_bounds)  # (n_subdomains, n_dims)
        upper_bounds = jnp.array(self.upper_bounds)  # (n_subdomains, n_dims)
        widths_lower = jnp.array(self.widths_lower)  # (n_subdomains, n_dims)
        widths_upper = jnp.array(self.widths_upper)  # (n_subdomains, n_dims)
        
        # Extend bounds by widths (for overlap regions)
        extended_lower = lower_bounds - widths_lower
        extended_upper = upper_bounds + widths_upper
        
        # Sigmoid-based bump function (numerically stable)
        # x: (batch_size, n_dims) -> (batch_size, 1, n_dims)
        x_expanded = x[:, None, :]
        
        # bounds: (n_subdomains, n_dims) -> (1, n_subdomains, n_dims)
        extended_lower = extended_lower[None, :, :]
        extended_upper = extended_upper[None, :, :]
        widths_lower_exp = widths_lower[None, :, :] + 1e-8
        widths_upper_exp = widths_upper[None, :, :] + 1e-8
        
        # Compute sigmoid arguments with clamping to avoid overflow
        lower_arg = jnp.clip((x_expanded - extended_lower) / widths_lower_exp, -10, 10)
        upper_arg = jnp.clip((x_expanded - extended_upper) / widths_upper_exp, -10, 10)
        
        # Sigmoid values
        lower_sigmoid = 1.0 / (1.0 + jnp.exp(-lower_arg))  # 1 when x >> lower
        upper_sigmoid = 1.0 / (1.0 + jnp.exp(upper_arg))   # 1 when x << upper
        
        # Per-dimension window: (batch_size, n_subdomains, n_dims)
        dim_windows = lower_sigmoid * upper_sigmoid
        
        # Product across dimensions: (batch_size, n_subdomains)
        windows = jnp.prod(dim_windows, axis=-1)
        
        # Normalize to partition of unity
        windows = windows / (jnp.sum(windows, axis=1, keepdims=True) + 1e-8)
        
        return windows
    
    def inside_points(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Find which points are inside which subdomains (FBPINNs-style optimization).
        
        Only points within the extended bounds (including overlap) are considered
        "inside" a subdomain. This allows us to only evaluate relevant points
        through each subdomain network.
        
        JIT-compatible: uses fixed max_size for jnp.nonzero.
        
        Args:
            x: Input array of shape (batch_size, n_dims)
        
        Returns:
            n_take: Point indices for each (point, subdomain) pair inside
            m_take: Subdomain indices for each pair
            inside_ims: Which subdomains have at least one point inside
        """
        n_points = x.shape[0]
        n_dims = x.shape[1]
        
        # Get extended bounds (bounds + overlap widths)
        lower_bounds = jnp.array(self.lower_bounds)  # (n_subdomains, n_dims)
        upper_bounds = jnp.array(self.upper_bounds)  # (n_subdomains, n_dims)
        widths_lower = jnp.array(self.widths_lower)  # (n_subdomains, n_dims)
        widths_upper = jnp.array(self.widths_upper)  # (n_subdomains, n_dims)
        
        extended_lower = lower_bounds - widths_lower
        extended_upper = upper_bounds + widths_upper
        
        # Check which points are inside each subdomain
        # x: (n, d) -> (n, 1, d)
        # bounds: (m, d) -> (1, m, d)
        x_expanded = x[:, None, :]
        extended_lower_exp = extended_lower[None, :, :]
        extended_upper_exp = extended_upper[None, :, :]
        
        # inside: (n, m, d) -> (n, m) after all()
        inside = (x_expanded >= extended_lower_exp) & (x_expanded <= extended_upper_exp)
        inside = jnp.all(inside, axis=-1)  # (n_points, n_subdomains)
        
        # Only consider active subdomains
        active_mask = jnp.array(self.active_mask)
        inside = inside & active_mask[None, :]
        
        # JIT-compatible: use fixed max_size (worst case: all points in all active subdomains)
        # For typical overlaps (~50%), actual pairs is ~2-3x n_points
        max_pairs = n_points * self.n_active
        
        # Get indices of (point, subdomain) pairs that are inside
        n_take, m_take = jnp.nonzero(inside, size=max_pairs, fill_value=-1)
        
        # Get which subdomains have at least one point
        inside_ims = jnp.nonzero(jnp.any(inside, axis=0), size=self.n_active, fill_value=-1)[0]
        
        return n_take, m_take, inside_ims
    
    def scale_input(self, x: jnp.ndarray, subdomain_idx: int) -> jnp.ndarray:
        """Scale input to [-1, 1] within subdomain bounds."""
        if not self.normalize_input:
            return x
        
        lower = jnp.array(self.lower_bounds[subdomain_idx])
        upper = jnp.array(self.upper_bounds[subdomain_idx])
        
        x_norm = (x - lower) / (upper - lower + 1e-8)
        return 2.0 * x_norm - 1.0
    
    def scale_input_vectorized(self, x: jnp.ndarray, active_indices: jnp.ndarray) -> jnp.ndarray:
        """
        Scale input for all active subdomains at once.
        
        Args:
            x: Input array of shape (batch_size, n_dims)
            active_indices: Array of active subdomain indices
        
        Returns:
            Scaled inputs of shape (n_active, batch_size, n_dims)
        """
        if not self.normalize_input:
            # Broadcast x to (n_active, batch, dims)
            return jnp.broadcast_to(x, (len(active_indices),) + x.shape)
        
        # Get bounds for active subdomains: (n_active, n_dims)
        lower = jnp.array(self.lower_bounds)[active_indices]
        upper = jnp.array(self.upper_bounds)[active_indices]
        
        # Broadcast: x (batch, dims) -> (1, batch, dims)
        # bounds (n_active, dims) -> (n_active, 1, dims)
        x_expanded = x[None, :, :]  # (1, batch, dims)
        lower_expanded = lower[:, None, :]  # (n_active, 1, dims)
        upper_expanded = upper[:, None, :]  # (n_active, 1, dims)
        
        # Scale: result is (n_active, batch, dims)
        x_norm = (x_expanded - lower_expanded) / (upper_expanded - lower_expanded + 1e-8)
        return 2.0 * x_norm - 1.0
    
    def _unnormalize_output(self, y: jnp.ndarray) -> jnp.ndarray:
        """Unnormalize output from [-1, 1] to physical range."""
        if self.output_min is None or not self.unnormalize_output:
            return y
        
        y_norm = (y + 1.0) / 2.0
        return y_norm * (self.output_max - self.output_min) + self.output_min
    
    def apply(self, params: Dict, x: jnp.ndarray, 
              params_dict: Optional[Dict] = None) -> jnp.ndarray:
        """
        Forward pass: compute weighted sum of subdomain predictions.
        
        Uses vmap to vectorize over subdomains for efficient GPU execution.
        
        Args:
            params: Network parameters from init()
            x: Input array of shape (batch_size, n_dims)
            params_dict: Optional dict passed to transforms
        
        Returns:
            Output array of shape (batch_size, n_outputs)
        """
        x_original = x
        
        # Apply input transform (e.g., symmetry)
        if self.input_transform is not None:
            x = self.input_transform(x, params_dict)
        
        batch_size = x.shape[0]
        
        # Compute windows for all subdomains: (batch_size, n_subdomains)
        windows = self.compute_windows(x)
        
        # Get windows for active subdomains: (n_active, batch_size)
        active_indices_jax = jnp.array(self.active_indices)
        active_windows = windows[:, active_indices_jax].T  # (n_active, batch_size)
        
        # Stack params for active subdomains
        # {subnet_0: ..., subnet_1: ...} -> stacked arrays (n_active, ...)
        subnet_params_list = [params[f'subnet_{sub_idx}'] for sub_idx in self.active_indices]
        stacked_params = jax.tree_util.tree_map(
            lambda *leaves: jnp.stack(leaves, axis=0),
            *subnet_params_list
        )
        
        # Scale inputs for all active subdomains: (n_active, batch_size, n_dims)
        x_scaled = self.scale_input_vectorized(x, active_indices_jax)
        
        # Create subnet module for forward pass
        subnet = FBPINNModule(
            layer_sizes=self.layer_sizes,
            activation=self.activation
        )
        
        # vmap over subdomains: process all subdomains in parallel
        # in_axes: (0, 0) means vmap over first axis of both params and x
        def forward_single_subdomain(subnet_params, x_sub):
            return subnet.apply(subnet_params, x_sub)
        
        vmapped_forward = jax.vmap(forward_single_subdomain, in_axes=(0, 0))
        
        # Forward pass: (n_active, batch_size, n_outputs)
        all_preds = vmapped_forward(stacked_params, x_scaled)
        
        # Weight by windows and sum over subdomains
        # all_preds: (n_active, batch, outputs)
        # active_windows: (n_active, batch) -> (n_active, batch, 1)
        weighted_preds = all_preds * active_windows[:, :, None]
        output = jnp.sum(weighted_preds, axis=0)  # (batch, outputs)
        
        # Unnormalize output
        output = self._unnormalize_output(output)
        
        # Apply output transform
        if self.output_transform is not None:
            output = self.output_transform(x_original, output, params_dict)
        
        return output
    
    def forward(self, x: jnp.ndarray, params_dict: Optional[Dict] = None) -> jnp.ndarray:
        """
        Forward pass using stored parameters.
        
        Provides PyTorch-compatible API for inference.
        
        Args:
            x: Input tensor of shape (batch_size, n_inputs)
            params_dict: Optional dict passed to transforms
        
        Returns:
            Output tensor of shape (batch_size, n_outputs)
        """
        return self.apply(self.params, x, params_dict)
    
    def predict(self, x_np: np.ndarray, params_dict: Optional[Dict] = None) -> np.ndarray:
        """
        Predict with numpy I/O.
        
        Args:
            x_np: Input numpy array of shape (batch_size, n_inputs)
            params_dict: Optional dict passed to transforms
            
        Returns:
            Output numpy array of shape (batch_size, n_outputs)
        """
        x_jax = jnp.array(x_np)
        y = self.forward(x_jax, params_dict)
        return np.array(y)
    
    def precompute_sparse_indices_jit(self, x: jnp.ndarray, threshold: float = 1e-6,
                                      params_dict: Optional[Dict] = None) -> Dict:
        """
        Precompute sparse index structure for efficient differentiable forward passes.
        
        Only precomputes INDICES (which points belong to which subdomain).
        This is JIT-compatible and allows differentiation w.r.t. x since the
        actual computation still traces through x.
        
        Args:
            x: Input array of shape (batch_size, n_dims)
            threshold: Minimum window value to consider a point active
            params_dict: Optional dict passed to input transform
        
        Returns:
            Dict with index arrays for apply_sparse_differentiable()
        """
        x_transformed = x
        if self.input_transform is not None:
            x_transformed = self.input_transform(x, params_dict)
        
        batch_size = x.shape[0]
        n_dims = x.shape[1]
        output_size = self.layer_sizes[-1]
        n_active = len(self.active_indices)
        
        # Compute windows
        windows = self.compute_windows(x_transformed)
        
        # Find max active points across all subdomains
        max_pts_list = []
        for sub_idx in self.active_indices:
            mask = windows[:, sub_idx] > threshold
            max_pts_list.append(int(jnp.sum(mask)))
        max_pts = max(max_pts_list) if max_pts_list else 1
        
        # Precompute indices only (padded for JIT)
        all_indices = jnp.zeros((n_active, max_pts), dtype=jnp.int32)
        all_n_valid = jnp.zeros(n_active, dtype=jnp.int32)
        
        for i, sub_idx in enumerate(self.active_indices):
            w = windows[:, sub_idx]
            mask = w > threshold
            indices = jnp.where(mask, size=max_pts, fill_value=0)[0]
            n_valid = jnp.sum(mask)
            
            all_indices = all_indices.at[i].set(indices)
            all_n_valid = all_n_valid.at[i].set(n_valid)
        
        return {
            'batch_size': batch_size,
            'output_size': output_size,
            'max_pts': max_pts,
            'n_active': n_active,
            'all_indices': all_indices,  # (n_active, max_pts)
            'all_n_valid': all_n_valid,  # (n_active,)
            'threshold': threshold,
        }
    
    def apply_sparse_differentiable(self, params: Dict, x: jnp.ndarray,
                                    sparse_indices: Dict,
                                    params_dict: Optional[Dict] = None) -> jnp.ndarray:
        """
        Sparse forward pass that maintains differentiability w.r.t. x.
        
        Uses precomputed indices to know which points belong to which subdomain,
        but re-computes scaling and window values from x to maintain the gradient
        trace for dy/dx (needed for PDE residuals).
        
        Args:
            params: Network parameters from init()
            x: Input array of shape (batch_size, n_dims)
            sparse_indices: Output from precompute_sparse_indices_jit()
            params_dict: Optional dict passed to transforms
        
        Returns:
            Output array of shape (batch_size, n_outputs) with gradient trace
        """
        x_original = x
        x_transformed = x
        if self.input_transform is not None:
            x_transformed = self.input_transform(x, params_dict)
        
        batch_size = sparse_indices['batch_size']
        output_size = sparse_indices['output_size']
        max_pts = sparse_indices['max_pts']
        all_indices = sparse_indices['all_indices']
        all_n_valid = sparse_indices['all_n_valid']
        
        # Recompute windows from x (to maintain gradient trace)
        windows = self.compute_windows(x_transformed)
        
        # Get jax arrays for geometry
        lower_bounds = jnp.array(self.lower_bounds)
        upper_bounds = jnp.array(self.upper_bounds)
        
        # Stack params for vmap
        subnet_params_list = [params[f'subnet_{sub_idx}'] for sub_idx in self.active_indices]
        stacked_params = jax.tree_util.tree_map(
            lambda *leaves: jnp.stack(leaves, axis=0),
            *subnet_params_list
        )
        
        # Create subnet module
        subnet = FBPINNModule(
            layer_sizes=self.layer_sizes,
            activation=self.activation
        )
        
        # For each subdomain, gather active points, scale, forward, weight
        # Using vmap for efficiency
        active_sub_indices = jnp.array(self.active_indices)
        
        def forward_one_subdomain(sub_local_idx, subnet_params, indices, n_valid):
            """Process one subdomain (vmapped over subdomains)."""
            sub_idx = active_sub_indices[sub_local_idx]
            
            # Gather points (indices are precomputed)
            x_active = x_transformed[indices]  # (max_pts, n_dims)
            
            # Scale inputs (traced through x)
            lb = lower_bounds[sub_idx]
            ub = upper_bounds[sub_idx]
            x_scaled = 2.0 * (x_active - lb) / (ub - lb + 1e-8) - 1.0
            
            # Forward through network
            pred = subnet.apply(subnet_params, x_scaled)  # (max_pts, output)
            
            # Get window weights (traced through x)
            w = windows[indices, sub_idx][:, None]  # (max_pts, 1)
            
            # Weight predictions
            weighted = pred * w  # (max_pts, output)
            
            # Mask out padded entries
            point_idx = jnp.arange(max_pts)
            valid_mask = point_idx < n_valid
            weighted = weighted * valid_mask[:, None]
            
            return weighted, indices
        
        # vmap over subdomains
        n_active = len(self.active_indices)
        sub_local_indices = jnp.arange(n_active)
        
        vmapped_forward = jax.vmap(
            forward_one_subdomain,
            in_axes=(0, 0, 0, 0)
        )
        
        all_weighted, all_scatter_indices = vmapped_forward(
            sub_local_indices, stacked_params, all_indices, all_n_valid
        )  # (n_active, max_pts, output)
        
        # Scatter-add to output
        output = jnp.zeros((batch_size, output_size))
        
        # Flatten for scatter
        flat_indices = all_scatter_indices.reshape(-1)
        flat_weighted = all_weighted.reshape(-1, output_size)
        
        output = output.at[flat_indices].add(flat_weighted)
        
        # Unnormalize output
        output = self._unnormalize_output(output)
        
        # Apply output transform
        if self.output_transform is not None:
            output = self.output_transform(x_original, output, params_dict)
        
        return output

    def precompute_training_data(self, x: jnp.ndarray, threshold: float = 1e-6,
                                  params_dict: Optional[Dict] = None) -> Dict:
        """
        Precompute sparse training data for efficient forward passes.
        
        Computes which points are "active" (window > threshold) for each subdomain,
        and precomputes scaled inputs and window weights. This data can be reused
        across all training epochs since collocation points don't change.
        
        Args:
            x: Input array of shape (batch_size, n_dims)
            threshold: Minimum window value to consider a point active (default: 1e-6)
            params_dict: Optional dict passed to input transform
        
        Returns:
            Dict containing precomputed data for apply_precomputed()
        """
        x_original = x
        
        # Apply input transform if present
        if self.input_transform is not None:
            x = self.input_transform(x, params_dict)
        
        batch_size = x.shape[0]
        output_size = self.layer_sizes[-1]
        
        # Compute windows for all points: (batch_size, n_subdomains)
        windows = self.compute_windows(x)
        
        # For each active subdomain, find points where window > threshold
        precomputed = {
            'batch_size': batch_size,
            'output_size': output_size,
            'x_original': x_original,
            'subdomains': {}
        }
        
        for i, sub_idx in enumerate(self.active_indices):
            w = windows[:, sub_idx]
            mask = w > threshold
            indices = jnp.where(mask)[0]
            
            if len(indices) > 0:
                # Get active points and their window weights
                x_active = x[indices]
                weights_active = w[indices, None]  # (n_active_pts, 1)
                
                # Pre-scale inputs for this subdomain
                x_scaled = self.scale_input(x_active, sub_idx)
                
                precomputed['subdomains'][sub_idx] = {
                    'indices': indices,           # Which points to process
                    'x_scaled': x_scaled,         # Pre-scaled inputs
                    'weights': weights_active,    # Window weights
                    'n_points': len(indices),
                }
        
        return precomputed
    
    def apply_precomputed(self, params: Dict, precomputed: Dict,
                          params_dict: Optional[Dict] = None) -> jnp.ndarray:
        """
        Forward pass using precomputed sparse training data.
        
        Only evaluates points that have non-negligible window values for each
        subdomain, significantly reducing computation for sparse window overlaps.
        
        Note: This uses a Python loop and is NOT jit-compatible. For JIT training,
        use the standard apply() method. This is useful when:
        - Training points are fixed (typical for PINNs)
        - You want to avoid recomputing indices every step
        
        Args:
            params: Network parameters from init()
            precomputed: Output from precompute_training_data()
            params_dict: Optional dict passed to output transform
        
        Returns:
            Output array of shape (batch_size, n_outputs)
        """
        batch_size = precomputed['batch_size']
        output_size = precomputed['output_size']
        x_original = precomputed['x_original']
        
        # Initialize output
        output = jnp.zeros((batch_size, output_size))
        
        # Create subnet module
        subnet = FBPINNModule(
            layer_sizes=self.layer_sizes,
            activation=self.activation
        )
        
        # Process each subdomain's active points
        for sub_idx, data in precomputed['subdomains'].items():
            indices = data['indices']
            x_scaled = data['x_scaled']
            weights = data['weights']
            
            # Forward pass through this subdomain
            subnet_params = params[f'subnet_{sub_idx}']
            pred = subnet.apply(subnet_params, x_scaled)
            
            # Weight by window and scatter back
            weighted_pred = pred * weights
            output = output.at[indices].add(weighted_pred)
        
        # Unnormalize output
        output = self._unnormalize_output(output)
        
        # Apply output transform
        if self.output_transform is not None:
            output = self.output_transform(x_original, output, params_dict)
        
        return output
    
    def precompute_training_data_jit(self, x: jnp.ndarray, threshold: float = 1e-6,
                                      params_dict: Optional[Dict] = None) -> Dict:
        """
        Precompute sparse training data in JIT-compatible format.
        
        Pads all subdomain data to the same shape for use with vmap.
        Returns arrays instead of dicts for efficient JIT tracing.
        
        Args:
            x: Input array of shape (batch_size, n_dims)
            threshold: Minimum window value to consider a point active
            params_dict: Optional dict passed to input transform
        
        Returns:
            Dict with padded arrays for JIT-compatible apply_precomputed_jit()
        """
        x_original = x
        
        # Apply input transform if present
        if self.input_transform is not None:
            x = self.input_transform(x, params_dict)
        
        batch_size = x.shape[0]
        n_dims = x.shape[1]
        output_size = self.layer_sizes[-1]
        n_active = len(self.active_indices)
        
        # Compute windows: (batch_size, n_subdomains)
        windows = self.compute_windows(x)
        
        # Find max active points across all subdomains
        max_pts_list = []
        for sub_idx in self.active_indices:
            mask = windows[:, sub_idx] > threshold
            max_pts_list.append(int(jnp.sum(mask)))
        max_pts = max(max_pts_list) if max_pts_list else 1
        
        # Allocate padded arrays
        # Shape: (n_active, max_pts, n_dims) for x_scaled
        # Shape: (n_active, max_pts, 1) for weights
        # Shape: (n_active, max_pts) for indices
        # Shape: (n_active,) for n_valid
        all_x_scaled = jnp.zeros((n_active, max_pts, n_dims))
        all_weights = jnp.zeros((n_active, max_pts, 1))
        all_indices = jnp.zeros((n_active, max_pts), dtype=jnp.int32)
        all_n_valid = jnp.zeros(n_active, dtype=jnp.int32)
        
        for i, sub_idx in enumerate(self.active_indices):
            w = windows[:, sub_idx]
            mask = w > threshold
            indices = jnp.where(mask, size=max_pts, fill_value=0)[0]
            n_valid = jnp.sum(mask)
            
            # Gather active points
            x_active = x[indices]
            weights_active = w[indices, None]
            
            # Scale inputs
            x_scaled = self.scale_input(x_active, sub_idx)
            
            # Store in padded arrays
            all_x_scaled = all_x_scaled.at[i].set(x_scaled)
            all_weights = all_weights.at[i].set(weights_active)
            all_indices = all_indices.at[i].set(indices)
            all_n_valid = all_n_valid.at[i].set(n_valid)
        
        return {
            'batch_size': batch_size,
            'output_size': output_size,
            'max_pts': max_pts,
            'x_original': x_original,
            'all_x_scaled': all_x_scaled,      # (n_active, max_pts, n_dims)
            'all_weights': all_weights,        # (n_active, max_pts, 1)
            'all_indices': all_indices,        # (n_active, max_pts)
            'all_n_valid': all_n_valid,        # (n_active,)
        }
    
    def apply_precomputed_jit(self, params: Dict, precomputed: Dict,
                               params_dict: Optional[Dict] = None) -> jnp.ndarray:
        """
        JIT-compatible forward pass using precomputed sparse training data.
        
        Uses vmap over subdomains with padded arrays for fixed shapes.
        
        Args:
            params: Network parameters from init()
            precomputed: Output from precompute_training_data_jit()
            params_dict: Optional dict passed to output transform
        
        Returns:
            Output array of shape (batch_size, n_outputs)
        """
        batch_size = precomputed['batch_size']
        output_size = precomputed['output_size']
        x_original = precomputed['x_original']
        all_x_scaled = precomputed['all_x_scaled']
        all_weights = precomputed['all_weights']
        all_indices = precomputed['all_indices']
        all_n_valid = precomputed['all_n_valid']
        
        # Stack params for vmap
        subnet_params_list = [params[f'subnet_{sub_idx}'] for sub_idx in self.active_indices]
        stacked_params = jax.tree_util.tree_map(
            lambda *leaves: jnp.stack(leaves, axis=0),
            *subnet_params_list
        )
        
        # Create subnet module
        subnet = FBPINNModule(
            layer_sizes=self.layer_sizes,
            activation=self.activation
        )
        
        # vmap forward pass over subdomains
        def forward_one_subdomain(subnet_params, x_scaled):
            return subnet.apply(subnet_params, x_scaled)
        
        vmapped_forward = jax.vmap(forward_one_subdomain, in_axes=(0, 0))
        all_preds = vmapped_forward(stacked_params, all_x_scaled)  # (n_active, max_pts, output)
        
        # Weight predictions
        all_weighted = all_preds * all_weights  # (n_active, max_pts, output)
        
        # Create validity mask to zero out padded entries
        # Shape: (n_active, max_pts, 1)
        max_pts = precomputed['max_pts']
        point_idx = jnp.arange(max_pts)[None, :, None]  # (1, max_pts, 1)
        valid_mask = point_idx < all_n_valid[:, None, None]  # (n_active, max_pts, 1)
        all_weighted = all_weighted * valid_mask
        
        # Scatter-add to output
        output = jnp.zeros((batch_size, output_size))
        
        # Flatten for scatter
        n_active = len(self.active_indices)
        flat_indices = all_indices.reshape(-1)  # (n_active * max_pts,)
        flat_weighted = all_weighted.reshape(-1, output_size)  # (n_active * max_pts, output)
        flat_valid = valid_mask.reshape(-1, 1)  # (n_active * max_pts, 1)
        
        # Zero out invalid entries before scatter
        flat_weighted = flat_weighted * flat_valid
        
        # Use segment_sum for efficient scatter-add
        output = output.at[flat_indices].add(flat_weighted)
        
        # Unnormalize output
        output = self._unnormalize_output(output)
        
        # Apply output transform
        if self.output_transform is not None:
            output = self.output_transform(x_original, output, params_dict)
        
        return output
    
    def _stack_params(self, params: Dict) -> Dict:
        """
        Stack subdomain params into arrays indexed by subdomain.
        
        Converts from {subnet_0: {params: {...}}, subnet_1: ...}
        to stacked arrays that can be indexed by subdomain index.
        
        JIT-compatible: all operations use static shapes.
        
        This enables efficient indexing by subdomain index.
        """
        # Get params for each active subdomain
        subnet_params_list = [params[f'subnet_{sub_idx}'] for sub_idx in self.active_indices]
        
        # Stack them - use tree_map to stack corresponding leaves
        # Result: stacked arrays indexed 0..n_active-1
        stacked = jax.tree_util.tree_map(
            lambda *leaves: jnp.stack(leaves, axis=0),
            *subnet_params_list
        )
        
        # Create index map: original subdomain index -> position in stacked array
        # For invalid indices (inactive subdomains), map to 0
        idx_map = np.zeros(self.n_subdomains, dtype=np.int32)
        for i, sub_idx in enumerate(self.active_indices):
            idx_map[sub_idx] = i
        idx_map = jnp.array(idx_map)
        
        # Reindex using the map: stacked[idx_map[m_take]] gives correct params
        def reindex_by_map(arr):
            """Expand array so arr[subdomain_idx] works directly."""
            return arr[idx_map]  # (n_subdomains, ...)
        
        return jax.tree_util.tree_map(reindex_by_map, stacked)
    
    def to(self, device: str = None, dtype=None, seed: int = 0) -> 'FBPINN':
        """
        Initialize parameters and move to specified device.
        
        This provides PyTorch-compatible API. In JAX, device placement is handled
        automatically, but this method initializes params and stores them.
        
        Args:
            device: Device string ('cpu', 'gpu', 'tpu'). Default: auto-detect.
            dtype: Data type (e.g., jnp.float32). Default: jnp.float32.
            seed: Random seed for initialization.
        
        Returns:
            self (for method chaining)
        """
        # Auto-detect device if not specified
        if device is None:
            device = jax.devices()[0].platform  # 'cpu', 'gpu', 'tpu'
        
        # Default dtype
        if dtype is None:
            dtype = jnp.float32
        
        self.device = device
        self.dtype = dtype
        
        # Initialize parameters for all active subdomains (JAX handles placement)
        self.params = self.init(jax.random.PRNGKey(seed))
        
        return self
    
    def __call__(self, params: Dict, x: jnp.ndarray, 
                 params_dict: Optional[Dict] = None) -> jnp.ndarray:
        """Alias for apply()."""
        return self.apply(params, x, params_dict)


def create_fbpinn(domain,
                  layer_sizes: Sequence[int],
                  activation: str = 'tanh',
                  active_subdomains: Optional[List] = None,
                  seed: int = 0,
                  **kwargs) -> Tuple[FBPINN, Dict]:
    """
    Create an FBPINN model and initialize its parameters.
    
    Args:
        domain: DomainCubicPartition
        layer_sizes: Network architecture
        activation: Activation function name
        active_subdomains: Active subdomain specification
        seed: Random seed
        **kwargs: Additional arguments passed to FBPINN
    
    Returns:
        Tuple of (model, params)
    
    Example:
        model, params = create_fbpinn(domain, [3, 64, 3], seed=42)
        y = model.apply(params, x)
    """
    model = FBPINN(
        domain=domain,
        layer_sizes=layer_sizes,
        activation=activation,
        active_subdomains=active_subdomains,
        **kwargs
    )
    params = model.init(jax.random.PRNGKey(seed))
    return model, params
