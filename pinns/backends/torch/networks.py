import numpy as np
import torch
import torch.nn as nn
from pinns.domain import DomainCubicPartition

class FNN(nn.Module):
    """
    A fully-connected neural network with configurable layer sizes and activation function.
    
    Includes optional input normalization and output unnormalization layers.
    The normalization bounds are set by the Trainer based on the problem definition.
    
    Args:
        layer_sizes (list): List of integers specifying the size of each layer.
                           First element is input size, last element is output size.
                           Example: [2, 64, 64, 32, 1] creates a network with:
                           - Input layer: 2 features
                           - Hidden layers: 64, 64, 32 neurons
                           - Output layer: 1 feature
        activation (nn.Module or str): Activation function to use between layers.
                                       Can be a string ('relu', 'tanh', 'sigmoid', 'gelu', 'silu')
                                       or an nn.Module instance.
                                       Default: 'tanh'
        output_activation (nn.Module or None): Optional activation for the output layer.
                                               Default: None (no activation)
        normalize_input (bool): Whether to normalize inputs. Bounds set by Trainer. Default: True
        unnormalize_output (bool): Whether to unnormalize outputs. Bounds set by Trainer. Default: True
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
        seed (int): Random seed for weight initialization. If None, uses current RNG state. Default: None
    """
    
    def __init__(self, layer_sizes, activation='tanh', output_activation=None,
                 normalize_input=True, unnormalize_output=True, 
                 input_transform=None, output_transform=None, seed=None):
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
