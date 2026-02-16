"""
Base Trainer class with shared functionality for all backends.

This module contains the common code shared between PyTorch and JAX trainers,
including plotting, history management, parameter building, and utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Union, Any
from abc import ABC, abstractmethod


def is_notebook():
    """Check if running in a Jupyter notebook."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal IPython
        else:
            return False
    except (NameError, AttributeError):
        return False


class BaseTrainer(ABC):
    """
    Abstract base class for PINN trainers.
    
    Provides common functionality for:
    - Training history management
    - Parameter building
    - Plotting (1D, 2D, N-D solutions, residuals, errors, losses)
    - Utility methods for BC names, input/output names
    
    Subclasses must implement:
    - __init__: Initialize backend-specific components
    - compile: Configure training parameters (optimizer differs per backend)
    - train: Training loop (autodiff differs per backend)
    - predict: Inference (tensor handling differs per backend)
    - _sample_interior, _sample_boundary: Point sampling
    - _compute_pde_loss, _compute_bc_loss: Loss computation
    """
    
    def __init__(self, problem, network, device=None):
        """
        Initialize common trainer attributes.
        
        Args:
            problem: Problem instance defining PDE and boundary conditions.
            network: Neural network to train.
            device: Device to use. If None, auto-detect using backend.
        """
        self.problem = problem
        
        # Training history
        self.history = {
            'epoch': [],
            'loss': [],
            'train_loss': [],  # Alias for compatibility
            'loss_pde': [],
            'loss_bcs': [],
            'test_loss': [],
            'solution_error': [],
            'epoch_times': [],
        }
        
        # Global epoch counter (accumulated across train() calls)
        self._global_epoch = 0
        
        # Figure for persistent plotting
        self._fig = None
        self._axes = None
        self._display_handle = None
        self._colorbars = []
        
        # Random number generator
        self.rng = np.random.default_rng()
        
        # Auto-detect device if not specified
        if device is None:
            device = self._auto_detect_device()
        
        # Initialize network and move to device
        self.network = network.to(device)
        self.device = getattr(self.network, 'device', device)
        self.dtype = getattr(self.network, 'dtype', None)
        
        # Set normalization bounds on the network from the problem
        self._setup_network_normalization()
        
        # Training configuration defaults
        n_bcs = len(problem.boundary_conditions)
        expected_len = 1 + n_bcs
        self.train_samples = [100] + [10] * n_bcs
        self.test_samples = [100] + [0] * n_bcs
        self.weights = [1.0] * expected_len
        self.learning_rate = 1e-3
        self.optimizer_name = "adam"
        self.optimizer = None
        self.opt_state = None
        
        # Sampled data (dict format for internal use)
        self._train_data = None
        self._test_data = None
        
        # Sparse FBPINN precomputation (common to all backends)
        self._use_sparse_fbpinn = True
        self._sparse_threshold = 1e-6
        self._precomputed_pde = None
        self._precomputed_bcs = {}
        
        # Training configuration (set by compile)
        self._epochs = 1000
        self._print_each = 100
        self._show_plots = False
        self._save_plots = None
        self._show_subdomains = {'solution': False, 'residuals': False, 'zoom': False}
        self._show_sampling_points = {'solution': False, 'residuals': False, 'zoom': False}
        self._plot_regions = []
        self._plot_n_points = 200
        self._batch_size = None
        self._compiled = False
    
    def _setup_network_normalization(self):
        """Set up input/output normalization on the network from problem definition."""
        xmin = np.array(self.problem.xmin)
        xmax = np.array(self.problem.xmax)
        
        # Set input range from domain bounds (only if normalization is enabled)
        if hasattr(self.network, 'set_input_range'):
            if getattr(self.network, 'normalize_input', True):
                self.network.set_input_range(xmin, xmax)
        
        # Set output range from problem definition (only if unnormalization is enabled)
        if hasattr(self.network, 'set_output_range'):
            if getattr(self.network, 'unnormalize_output', True):
                if self.problem.output_range is not None:
                    output_range = self.problem.output_range
                    if isinstance(output_range, list):
                        ymin = np.array([r[0] if r is not None else -1.0 for r in output_range])
                        ymax = np.array([r[1] if r is not None else 1.0 for r in output_range])
                    else:
                        ymin, ymax = output_range
                        ymin = np.array([ymin] * self.problem.n_outputs)
                        ymax = np.array([ymax] * self.problem.n_outputs)
                else:
                    ymin = -np.ones(self.problem.n_outputs)
                    ymax = np.ones(self.problem.n_outputs)
                self.network.set_output_range(ymin, ymax)
    
    # ==================== Abstract Methods ====================
    
    @abstractmethod
    def _auto_detect_device(self) -> str:
        """Auto-detect the best available device. Must be implemented by subclass."""
        pass
    
    @abstractmethod
    def _create_optimizer(self):
        """Create optimizer. Must be implemented by subclass (backend-specific)."""
        pass
    
    @abstractmethod
    def _init_optimizer_state(self):
        """Initialize optimizer state after creating optimizer. Override if needed (e.g., JAX/optax)."""
        pass
    
    def _after_compile_hook(self):
        """Sample initial training and test data. Override for additional behavior."""
        self._sample_train_data()
        if any(s > 0 for s in self.test_samples):
            self._sample_test_data()
    
    # ==================== Compile ====================
    
    def compile(
        self,
        train_samples: Union[List[int], Dict[str, int]] = None,
        test_samples: Union[List[int], Dict[str, int]] = None,
        weights: Union[List[float], Dict[str, float]] = None,
        optimizer: str = None,
        learning_rate: float = None,
        epochs: int = 1000,
        batch_size: int = None,
        print_each: int = 100,
        show_plots: bool = False,
        save_plots: str = None,
        show_subdomains=False,
        show_sampling_points=False,
        plot_regions: List[tuple] = None,
        plot_n_points: int = 200,
        # L-BFGS specific parameters
        lbfgs_max_iter: int = 5,
        lbfgs_history_size: int = 50,
        lbfgs_tolerance: float = 1e-9,
        lbfgs_line_search: str = "strong_wolfe",
    ):
        """
        Configure training parameters.
        
        Args:
            train_samples: Number of samples for each loss term (list or dict).
            test_samples: Number of test samples for each loss term.
            weights: Weights for each loss term.
            optimizer: Optimizer name ('adam', 'sgd', 'lbfgs').
            learning_rate: Learning rate.
            epochs: Number of training epochs.
            batch_size: Batch size (if applicable).
            print_each: Print progress every N epochs.
            show_plots: Show plots during training.
            save_plots: Path prefix for saving plots.
            show_subdomains: Show subdomain boundaries in plots.
            show_sampling_points: Show sampling points in plots.
            plot_regions: List of zoom regions for additional plots.
            plot_n_points: Number of points for plotting.
            lbfgs_max_iter: Max iterations per L-BFGS step (default: 5).
            lbfgs_history_size: History size for L-BFGS (default: 50).
            lbfgs_tolerance: Tolerance for gradient convergence (default: 1e-9).
            lbfgs_line_search: Line search method - 'strong_wolfe' or None (default: 'strong_wolfe').
        """
        n_bcs = len(self.problem.boundary_conditions)
        expected_len = 1 + n_bcs
        
        # Store L-BFGS parameters
        self._lbfgs_max_iter = lbfgs_max_iter
        self._lbfgs_history_size = lbfgs_history_size
        self._lbfgs_tolerance = lbfgs_tolerance
        self._lbfgs_line_search = lbfgs_line_search
        
        if train_samples is not None:
            train_samples = self._convert_dict_to_list(train_samples, 'train_samples')
            if len(train_samples) != expected_len:
                raise ValueError(
                    f"train_samples must have {expected_len} elements "
                    f"(1 interior + {n_bcs} BCs), got {len(train_samples)}"
                )
            self.train_samples = train_samples
        
        if test_samples is not None:
            test_samples = self._convert_dict_to_list(test_samples, 'test_samples')
            if len(test_samples) != expected_len:
                raise ValueError(
                    f"test_samples must have {expected_len} elements, got {len(test_samples)}"
                )
            self.test_samples = test_samples
        
        if weights is not None:
            weights = self._convert_dict_to_list(weights, 'weights')
            if len(weights) != expected_len:
                raise ValueError(
                    f"weights must have {expected_len} elements, got {len(weights)}"
                )
            self.weights = weights
        
        optimizer_changed = False
        if optimizer is not None and optimizer.lower() != self.optimizer_name:
            self.optimizer_name = optimizer.lower()
            optimizer_changed = True
        
        if learning_rate is not None and learning_rate != self.learning_rate:
            self.learning_rate = learning_rate
            optimizer_changed = True
        
        if self.optimizer is None or optimizer_changed:
            self.optimizer = self._create_optimizer()
            self._init_optimizer_state()
        
        self._epochs = epochs
        self._print_each = print_each
        self._show_plots = show_plots
        self._save_plots = save_plots
        
        if isinstance(show_subdomains, bool):
            self._show_subdomains = {'solution': show_subdomains, 'residuals': show_subdomains, 'zoom': show_subdomains}
        else:
            self._show_subdomains = {'solution': False, 'residuals': False, 'zoom': False}
            self._show_subdomains.update(show_subdomains)
        
        if isinstance(show_sampling_points, bool):
            self._show_sampling_points = {'solution': show_sampling_points, 'residuals': show_sampling_points, 'zoom': show_sampling_points}
        else:
            self._show_sampling_points = {'solution': False, 'residuals': False, 'zoom': False}
            self._show_sampling_points.update(show_sampling_points)
        
        self._plot_regions = plot_regions if plot_regions is not None else []
        self._plot_n_points = plot_n_points
        self._batch_size = batch_size
        
        # Backend-specific hook (e.g., presampling in JAX)
        self._after_compile_hook()
        
        self._compiled = True
    
    # ==================== Abstract Methods Continued ====================
    
    @abstractmethod
    def train(self):
        """Run training loop. Must be implemented by subclass."""
        pass
    
    def predict(self, x: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Predict output for given input points.
        
        Delegates to self.network.predict() which handles numpy ↔ tensor conversion.
        
        Args:
            x: Input points as numpy array of shape (n_points, n_inputs)
            batch_size: Optional batch size for large inputs
            
        Returns:
            Predictions as numpy array of shape (n_points, n_outputs)
        """
        if batch_size is None:
            batch_size = getattr(self, '_batch_size', None)
        
        params_dict = self._build_params()
        
        if batch_size is None or batch_size >= len(x):
            return self.network.predict(x, params_dict)
        else:
            results = []
            for start in range(0, len(x), batch_size):
                end = min(start + batch_size, len(x))
                y_batch = self.network.predict(x[start:end], params_dict)
                results.append(y_batch)
            return np.vstack(results)
    
    @abstractmethod
    def _to_tensor(self, np_array: np.ndarray):
        """Convert numpy array to backend tensor (jnp.array or torch.Tensor)."""
        pass
    
    @abstractmethod
    def _get_pde_residual_tensor(self, x, y, params_dict: Dict[str, Any]):
        """
        Compute PDE residual tensor using backend-specific autodiff.
        
        Args:
            x: Input tensor (backend-specific)
            y: Network output tensor
            params_dict: Parameters dict from _build_params()
            
        Returns:
            Residual tensor or list of residual tensors (backend-specific)
        """
        pass
    
    @abstractmethod
    def _mean_squared(self, tensor) -> float:
        """Compute mean squared value of a tensor. Returns backend tensor scalar."""
        pass
    
    @abstractmethod
    def _compute_directional_derivative(self, x, component: int, dim: int, params_dict: Dict[str, Any]):
        """
        Compute directional derivative of network output w.r.t. input dimension.
        
        Args:
            x: Input tensor (backend-specific)
            component: Which output component to differentiate
            dim: Which input dimension to differentiate w.r.t.
            params_dict: Parameters dict from _build_params()
            
        Returns:
            Tensor of shape (batch_size,) containing du/dx_dim
        """
        pass
    
    def _get_bc_target(self, bc, x):
        """
        Get BC target value as backend tensor.
        
        Args:
            bc: Boundary condition object
            x: Input points (backend-specific tensor)
            
        Returns:
            Target tensor (backend-specific)
        """
        from pinns.boundary import PointsetBC
        
        if isinstance(bc, PointsetBC):
            return self._to_tensor(np.array(bc.values))
        elif callable(bc.value):
            x_np = np.asarray(x)  # Convert to numpy for user function
            result = bc.value(x_np)
            return self._to_tensor(np.asarray(result))
        else:
            return self._to_tensor(np.full((x.shape[0],), bc.value))
    
    # ==================== PDE Loss (Common Implementation) ====================
    
    def _compute_pde_loss(self, x, params_dict: Dict[str, Any], pde_weights=None):
        """
        Compute PDE loss - common logic, backend-specific tensor ops.
        
        Args:
            x: Input tensor (backend-specific)
            params_dict: Parameters dict from _build_params()
            pde_weights: Optional per-equation weights (list/tuple or scalar)
            
        Returns:
            Tuple of (total_loss, individual_losses_list)
        """
        y = self.network.forward(x, params_dict)
        residual = self._get_pde_residual_tensor(x, y, params_dict)
        
        if isinstance(residual, (list, tuple)):
            individual = [self._mean_squared(r) for r in residual]
            if pde_weights is not None and isinstance(pde_weights, (list, tuple)):
                if len(pde_weights) != len(residual):
                    raise ValueError(
                        f"Number of PDE weights ({len(pde_weights)}) must match "
                        f"number of PDE equations ({len(residual)})"
                    )
                total = sum(w * l for w, l in zip(pde_weights, individual))
            else:
                total = sum(individual) / len(individual)
            return total, individual
        else:
            loss = self._mean_squared(residual)
            return loss, [loss]
    
    # ==================== BC Loss (Common Implementation) ====================
    
    def _compute_bc_loss(self, bc, x, params_dict: Dict[str, Any]):
        """
        Compute boundary condition loss - common logic, backend-specific gradient.
        
        Args:
            bc: Boundary condition object
            x: Input tensor (backend-specific)
            params_dict: Parameters dict from _build_params()
            
        Returns:
            Loss tensor (backend-specific scalar)
        """
        from pinns.boundary import DirichletBC, NeumannBC, RobinBC, PointsetBC
        
        y = self.network.forward(x, params_dict)
        
        if isinstance(bc, DirichletBC):
            target = self._get_bc_target(bc, x)
            diff = y[:, bc.component] - target
            return self._mean_squared(diff)
        
        elif isinstance(bc, NeumannBC):
            normal_dim, normal_sign = bc.get_normal_direction()
            du_dn = self._compute_directional_derivative(x, bc.component, normal_dim, params_dict)
            target = self._get_bc_target(bc, x)
            diff = normal_sign * du_dn - target
            return self._mean_squared(diff)
        
        elif isinstance(bc, RobinBC):
            normal_dim, normal_sign = bc.get_normal_direction()
            du_dn = self._compute_directional_derivative(x, bc.component, normal_dim, params_dict)
            alpha, beta, gamma = bc.get_coefficients(x)
            residual = alpha * y[:, bc.component] + beta * normal_sign * du_dn - gamma
            return self._mean_squared(residual)
        
        elif isinstance(bc, PointsetBC):
            target = self._get_bc_target(bc, x)
            diff = y[:, bc.component] - target
            return self._mean_squared(diff)
        
        else:
            raise ValueError(f"Unknown BC type: {type(bc)}")
    
    def _compute_total_loss(self, data: Dict, params_dict: Dict[str, Any], weights_dict: Dict):
        """
        Compute total weighted loss from data dict.
        
        Common implementation for both JAX and Torch.
        
        Args:
            data: Dict with 'pde' and BC name keys mapping to input tensors
            params_dict: Parameters dict from _build_params()
            weights_dict: Dict mapping loss names to weights
            
        Returns:
            Tuple of (total_loss, losses_dict)
        """
        total_loss = None
        losses = {}
        
        # PDE loss
        if 'pde' in data:
            x_pde = data['pde']
            pde_weights = weights_dict.get('pde') if isinstance(weights_dict.get('pde'), (list, tuple)) else None
            pde_loss, pde_individual = self._compute_pde_loss(x_pde, params_dict, pde_weights)
            
            # Apply scalar weight if not per-equation
            if not isinstance(weights_dict.get('pde'), (list, tuple)):
                pde_loss = weights_dict.get('pde', 1.0) * pde_loss
            
            losses['pde'] = pde_loss
            losses['pde_individual'] = pde_individual
            total_loss = pde_loss
        
        # BC losses
        bc_names = self._get_bc_names()
        losses['bcs'] = []
        for i, bc in enumerate(self.problem.boundary_conditions):
            name = bc_names[i]
            if name in data:
                x_bc = data[name]
                bc_loss = self._compute_bc_loss(bc, x_bc, params_dict)
                weighted_bc_loss = weights_dict.get(name, 1.0) * bc_loss
                losses['bcs'].append(weighted_bc_loss)
                losses[name] = weighted_bc_loss
                
                if total_loss is None:
                    total_loss = weighted_bc_loss
                else:
                    total_loss = total_loss + weighted_bc_loss
        
        return total_loss, losses
    
    def _compute_residuals(self, x_np: np.ndarray, batch_size: Optional[int] = None) -> List[np.ndarray]:
        """
        Compute PDE residuals at given points.
        
        Args:
            x_np: Input points as numpy array of shape (n_points, n_inputs).
            batch_size: Optional batch size for large inputs to avoid OOM.
            
        Returns:
            List of numpy arrays, one per residual equation.
        """
        if batch_size is None:
            batch_size = getattr(self, '_batch_size', None)
        
        # If no batching needed, compute directly
        if batch_size is None or batch_size >= len(x_np):
            x = self._to_tensor(x_np)
            params_dict = self._build_params()
            y = self.network.forward(x, params_dict)
            residual = self._get_pde_residual_tensor(x, y, params_dict)
            
            if isinstance(residual, (list, tuple)):
                return [self._to_numpy(r).flatten() for r in residual]
            else:
                return [self._to_numpy(residual).flatten()]
        
        # Batched computation
        params_dict = self._build_params()
        all_residuals = None
        
        for start in range(0, len(x_np), batch_size):
            end = min(start + batch_size, len(x_np))
            x_batch = self._to_tensor(x_np[start:end])
            y_batch = self.network.forward(x_batch, params_dict)
            residual_batch = self._get_pde_residual_tensor(x_batch, y_batch, params_dict)
            
            if isinstance(residual_batch, (list, tuple)):
                batch_res = [self._to_numpy(r).flatten() for r in residual_batch]
            else:
                batch_res = [self._to_numpy(residual_batch).flatten()]
            
            if all_residuals is None:
                all_residuals = [[] for _ in batch_res]
            
            for i, r in enumerate(batch_res):
                all_residuals[i].append(r)
        
        return [np.concatenate(res_list) for res_list in all_residuals]
    
    @abstractmethod
    def _to_numpy(self, tensor) -> np.ndarray:
        """Convert backend tensor to numpy array."""
        pass
    
    # ==================== Sampling Methods ====================
    
    def _get_bc_by_name(self, name: str):
        """Get boundary condition by name."""
        for bc in self.problem.boundary_conditions:
            if hasattr(bc, 'name') and bc.name == name:
                return bc
        return None
    
    def _sample_interior_np(self, n_points: int) -> np.ndarray:
        """Sample interior points and return as numpy array."""
        from pinns.boundary import PointsetBC
        params = self._build_params()
        return self.problem.domain.sample_interior(n_points, rng=self.rng, params=params)
    
    def _sample_boundary_np(self, bc, n_points: int) -> np.ndarray:
        """Sample boundary points for a specific boundary condition."""
        from pinns.boundary import PointsetBC
        
        if isinstance(bc, PointsetBC):
            return bc.points
        
        boundary = bc.boundary
        domain = self.problem.domain
        params = self._build_params()
        
        method = getattr(bc, 'sampling_method', 'uniform')
        transform = getattr(bc, 'sampling_transform', None)
        
        for dim, side in enumerate(boundary):
            if side is not None:
                points = domain.sample_boundary(
                    n_points, dim, side, rng=self.rng,
                    method=method, transform=transform, params=params
                )
                
                subdomain = getattr(bc, 'subdomain', None)
                if subdomain is not None:
                    sub_min, sub_max = subdomain
                    for d in range(len(boundary)):
                        if boundary[d] is None:
                            orig_min = domain.xmin[d]
                            orig_max = domain.xmax[d]
                            orig_extent = orig_max - orig_min
                            normalized = (points[:, d] - orig_min) / orig_extent
                            points[:, d] = sub_min + normalized * (sub_max - sub_min)
                
                return points
        
        raise ValueError(f"Invalid boundary specification: {boundary}")
    
    def _sample_points_np(self, name: str, n_samples: int) -> np.ndarray:
        """Sample points for a given loss term (returns numpy)."""
        if name == 'pde':
            return self._sample_interior_np(n_samples)
        else:
            bc = self._get_bc_by_name(name)
            if bc is not None:
                return self._sample_boundary_np(bc, n_samples)
            else:
                raise ValueError(f"Unknown loss term: {name}")
    
    def _list_to_dict_samples(self, samples_list: List[int]) -> Dict[str, int]:
        """Convert list format samples to dict format."""
        bc_names = self._get_bc_names()
        result = {'pde': samples_list[0]}
        for i, name in enumerate(bc_names):
            result[name] = samples_list[i + 1]
        return result
    
    def _list_to_dict_weights(self, weights_list: List[float]) -> Dict[str, float]:
        """Convert list format weights to dict format."""
        bc_names = self._get_bc_names()
        result = {'pde': weights_list[0]}
        for i, name in enumerate(bc_names):
            result[name] = weights_list[i + 1]
        return result
    
    def _sample_train_data(self):
        """Sample training data and store as backend tensors."""
        self._train_data = {}
        samples_dict = self._list_to_dict_samples(self.train_samples)
        for name, n in samples_dict.items():
            if n > 0:
                np_data = self._sample_points_np(name, n)
                self._train_data[name] = self._to_tensor(np_data)
    
    def _sample_test_data(self):
        """Sample test data and store as backend tensors."""
        self._test_data = {}
        samples_dict = self._list_to_dict_samples(self.test_samples)
        for name, n in samples_dict.items():
            if n > 0:
                np_data = self._sample_points_np(name, n)
                self._test_data[name] = self._to_tensor(np_data)

    def _get_n_batches(self) -> int:
        """Get number of mini-batches based on PDE data size and batch_size."""
        if self._batch_size is None or self._batch_size <= 0:
            return 1
        if 'pde' not in self._train_data:
            return 1
        n_pde = len(self._train_data['pde'])
        return max(1, (n_pde + self._batch_size - 1) // self._batch_size)

    def _get_batch_indices(self, n_points: int, batch_idx: int, n_batches: int):
        """Get start and end indices for a given batch.
        
        Args:
            n_points: Total number of points
            batch_idx: Current batch index (0-based)
            n_batches: Total number of batches
            
        Returns:
            Tuple of (start_idx, end_idx)
        """
        batch_size = max(1, (n_points + n_batches - 1) // n_batches)
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_points)
        return start_idx, end_idx

    # ==================== Utility Methods ====================

    def _get_bc_names(self) -> List[str]:
        """Get list of boundary condition names."""
        names = []
        for i, bc in enumerate(self.problem.boundary_conditions):
            if hasattr(bc, 'name') and bc.name is not None:
                names.append(bc.name)
            else:
                names.append(f'bc_{i}')
        return names
    
    def _get_output_name(self, output_idx: int) -> str:
        """Get the name of an output by index."""
        if hasattr(self.problem, 'output_names') and self.problem.output_names is not None:
            return self.problem.output_names[output_idx]
        return f'u{output_idx}'
    
    def _get_input_name(self, input_idx: int) -> str:
        """Get the name of an input by index."""
        if hasattr(self.problem, 'input_names') and self.problem.input_names is not None:
            return self.problem.input_names[input_idx]
        return f'x{input_idx}'
    
    def _build_params(self, internal: Dict[str, Any] = None) -> Dict[str, Any]:
        """Build structured params dictionary for PDE and solution functions.
        
        Args:
            internal: Internal training state. If None, uses current epoch/step 0.
            
        Returns:
            Dict with 'fixed', 'infer', and 'internal' keys.
        """
        if internal is None:
            internal = {'global_step': self._global_epoch, 'step': 0}
        return {
            "fixed": self.problem.params,
            "infer": {},  # Reserved for future inverse problem support
            "internal": internal
        }
    
    def _convert_dict_to_list(self, data: Union[List, Dict], param_name: str) -> List:
        """Convert a dictionary of samples/weights to list format."""
        if isinstance(data, dict):
            bc_names = self._get_bc_names()
            result = []
            
            # First element is 'pde'
            if 'pde' not in data:
                raise ValueError(f"{param_name} dict must contain 'pde' key")
            result.append(data['pde'])
            
            # Then BC values in order
            for i, bc_name in enumerate(bc_names):
                if bc_name in data:
                    result.append(data[bc_name])
                elif f'bc_{i}' in data:
                    result.append(data[f'bc_{i}'])
                else:
                    raise ValueError(
                        f"{param_name} dict missing key '{bc_name}' (or 'bc_{i}'). "
                        f"Available BC names: {bc_names}"
                    )
            return result
        return list(data)
    
    def _get_colormap(self, output_idx: int) -> str:
        """Get colormap based on output range symmetry.
        
        Returns a diverging colormap if the output range is symmetric
        around zero, otherwise returns 'inferno'.
        """
        if self.problem.output_range is None:
            return 'inferno'
        elif self.problem.output_range[output_idx][0] == -self.problem.output_range[output_idx][1]:
            return 'managua_r'
        else:
            return 'inferno'
    
    def reset(self):
        """
        Reset training history and epoch counter.
        
        Call this to start fresh training while keeping the same problem/network.
        """
        self.history = {
            'epoch': [],
            'loss': [],
            'train_loss': [],
            'loss_pde': [],
            'loss_bcs': [],
            'test_loss': [],
            'solution_error': [],
            'epoch_times': [],
        }
        self._global_epoch = 0
        self._fig = None
        self._axes = None
        self._display_handle = None
        self._colorbars = []
        self._compiled = False
    
    # ==================== Region Parsing ====================
    
    def _parse_region_nd(self, region):
        """Parse an N-dimensional region specification.
        
        Args:
            region: Tuple with one element per dimension. Each element can be:
                - None: use full range for this dimension (free dimension)
                - (min, max): range for this dimension (free dimension, zoomed)
                - scalar: fix this dimension at that value (sliced dimension)
                
        Returns:
            tuple: (free_dims, free_ranges, fixed_dims, fixed_values)
        """
        n_dims = self.problem.n_dims
        
        if region is None:
            region = [None] * n_dims
        
        free_dims = []
        free_ranges = []
        fixed_dims = []
        fixed_values = []
        
        for i, spec in enumerate(region):
            if spec is None:
                free_dims.append(i)
                free_ranges.append((self.problem.xmin[i], self.problem.xmax[i]))
            elif isinstance(spec, (list, tuple)) and len(spec) == 2:
                free_dims.append(i)
                free_ranges.append((spec[0], spec[1]))
            else:
                fixed_dims.append(i)
                fixed_values.append(float(spec))
        
        return free_dims, free_ranges, fixed_dims, fixed_values
    
    # ==================== Plotting Methods ====================
    
    def _clear_colorbars(self):
        """Remove all stored colorbars."""
        if hasattr(self, '_colorbars'):
            for cbar in self._colorbars:
                try:
                    cbar.remove()
                except:
                    pass
        self._colorbars = []
    
    def _create_figure(self):
        """Create figure and axes for plotting."""
        n_dims = self.problem.n_dims
        n_outputs = self.problem.n_outputs
        has_solution = self.problem.solution is not None
        n_regions = len(getattr(self, '_plot_regions', []))
        
        if n_dims == 1:
            if has_solution:
                n_cols = 3  # solution, residuals, error
            else:
                n_cols = 2  # solution, residuals
            
            n_rows = 1 + n_outputs + n_regions
            fig = plt.figure(figsize=(5 * n_cols, 3.5 * n_rows))
            gs = fig.add_gridspec(n_rows, n_cols)
            
            axes = {}
            axes['losses'] = fig.add_subplot(gs[0, :])
            
            for i in range(n_outputs):
                axes[f'sol_{i}'] = fig.add_subplot(gs[1 + i, 0])
                axes[f'res_{i}'] = fig.add_subplot(gs[1 + i, 1])
                if has_solution:
                    axes[f'err_{i}'] = fig.add_subplot(gs[1 + i, 2])
            
            # Region plots
            for r in range(n_regions):
                axes[f'region_{r}'] = fig.add_subplot(gs[1 + n_outputs + r, :])
        
        elif n_dims == 2:
            if has_solution:
                n_cols = 4  # predicted, true, residuals, error
            else:
                n_cols = 2  # predicted, residuals
            
            n_rows = 1 + n_outputs + n_regions
            fig = plt.figure(figsize=(4 * n_cols, 3.5 * n_rows))
            gs = fig.add_gridspec(n_rows, n_cols)
            
            axes = {}
            axes['losses'] = fig.add_subplot(gs[0, :])
            
            for i in range(n_outputs):
                axes[f'sol_{i}'] = fig.add_subplot(gs[1 + i, 0])
                if has_solution:
                    axes[f'true_{i}'] = fig.add_subplot(gs[1 + i, 1])
                    axes[f'res_{i}'] = fig.add_subplot(gs[1 + i, 2])
                    axes[f'err_{i}'] = fig.add_subplot(gs[1 + i, 3])
                else:
                    axes[f'res_{i}'] = fig.add_subplot(gs[1 + i, 1])
            
            for r in range(n_regions):
                axes[f'region_{r}'] = fig.add_subplot(gs[1 + n_outputs + r, :])
        
        else:
            # For 3D+: loss plot + region slices for all outputs with residuals
            n_cols = 2 * n_outputs  # Two columns per output (solution + residual)
            n_rows = 1 + n_regions  # 1 for loss + one row per region
            fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
            gs = fig.add_gridspec(n_rows, n_cols)
            
            axes = {}
            axes['losses'] = fig.add_subplot(gs[0, :])
            
            for r in range(n_regions):
                for i in range(n_outputs):
                    axes[f'region_{r}_{i}'] = fig.add_subplot(gs[1 + r, 2*i])
                    axes[f'region_res_{r}_{i}'] = fig.add_subplot(gs[1 + r, 2*i + 1])
        
        self._colorbars = []
        return fig, axes
    
    def _plot_losses(self, ax):
        """Plot loss curves on given axes."""
        epochs = self.history['epoch']
        if not epochs:
            return
        
        # Total loss (use 'loss' or 'train_loss')
        loss_data = self.history.get('loss', self.history.get('train_loss', []))
        if loss_data:
            ax.semilogy(epochs, loss_data, 'k-', label='Total', linewidth=2)
        
        # PDE losses
        pde_losses = self.history.get('loss_pde', [])
        if len(pde_losses) > 0:
            if isinstance(pde_losses[0], (list, tuple)):
                pde_array = np.array(pde_losses)
                for i in range(pde_array.shape[1]):
                    ax.semilogy(epochs, pde_array[:, i], '--', label=f'PDE eq{i+1}')
            else:
                ax.semilogy(epochs, pde_losses, '--', label='PDE')
        
        # BC losses with names
        bc_names = self._get_bc_names()
        bc_losses = self.history.get('loss_bcs', [])
        if bc_losses and len(bc_losses) > 0:
            bc_losses_array = np.array(bc_losses)
            if bc_losses_array.ndim == 2:
                for i in range(bc_losses_array.shape[1]):
                    bc_label = bc_names[i] if i < len(bc_names) else f'BC {i+1}'
                    ax.semilogy(epochs, bc_losses_array[:, i], '--', label=bc_label)
        
        # Test loss if available
        test_loss = self.history.get('test_loss', [])
        if len(test_loss) > 0:
            n_test = len(test_loss)
            test_epochs = np.linspace(epochs[0], epochs[-1], n_test).astype(int) if n_test > 1 else [epochs[-1]]
            ax.semilogy(test_epochs, test_loss, 'r:', marker='o', markersize=4, label='Test', linewidth=2)
        
        # Solution error if available
        sol_error = self.history.get('solution_error', [])
        if len(sol_error) > 0:
            n_err = len(sol_error)
            err_epochs = np.linspace(epochs[0], epochs[-1], n_err).astype(int) if n_err > 1 else [epochs[-1]]
            ax.semilogy(err_epochs, sol_error, 'm-', marker='s', markersize=4, label='Solution Error', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_solution_1d(self, ax, output_idx, n_points=200):
        """Plot 1D solution on given axes."""
        x = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points).reshape(-1, 1)
        y = self.predict(x)
        
        # Plot true solution if available
        if self.problem.solution is not None:
            y_true = self.problem.solution(x, self._build_params())
            if isinstance(y_true, (list, tuple)):
                y_true = np.concatenate([np.atleast_2d(yt).T if yt.ndim == 1 else yt for yt in y_true], axis=1)
            elif y_true.ndim == 1:
                y_true = y_true.reshape(-1, 1)
            ax.plot(x, y_true[:, output_idx], 'r-', linewidth=2, label='True')
            ax.plot(x, y[:, output_idx], 'b--', linewidth=2, label='Predicted')
            ax.legend(loc='best', fontsize=8)
        else:
            ax.plot(x, y[:, output_idx], 'b-', linewidth=2)
        
        output_name = self._get_output_name(output_idx)
        input_name = self._get_input_name(0)
        ax.set_xlabel(input_name)
        ax.set_ylabel(output_name)
        ax.set_title(f'Solution ({output_name})')
        ax.grid(True, alpha=0.3)
    
    def _plot_solution_2d(self, ax, output_idx, n_points=50):
        """Plot 2D solution as heatmap on given axes."""
        x0 = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points)
        x1 = np.linspace(self.problem.xmin[1], self.problem.xmax[1], n_points)
        X0, X1 = np.meshgrid(x0, x1)
        
        x_flat = np.column_stack([X0.ravel(), X1.ravel()])
        y = self.predict(x_flat)
        Y = y[:, output_idx].reshape(X0.shape)
        
        extent = [x0.min(), x0.max(), x1.min(), x1.max()]
        cmap = self._get_colormap(output_idx)
        im = ax.imshow(Y, extent=extent, origin='lower', aspect='equal', cmap=cmap)
        cbar = plt.colorbar(im, ax=ax)
        self._colorbars.append(cbar)
        
        output_name = self._get_output_name(output_idx)
        ax.set_title(f'Predicted ({output_name})')
        ax.set_xlabel(self._get_input_name(0))
        ax.set_ylabel(self._get_input_name(1))
    
    def _plot_true_solution_2d(self, ax, output_idx, n_points=50):
        """Plot 2D true solution as heatmap on given axes."""
        if self.problem.solution is None:
            return
        
        x0 = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points)
        x1 = np.linspace(self.problem.xmin[1], self.problem.xmax[1], n_points)
        X0, X1 = np.meshgrid(x0, x1)
        
        x_flat = np.column_stack([X0.ravel(), X1.ravel()])
        y_true = self.problem.solution(x_flat, self._build_params())
        
        if isinstance(y_true, (list, tuple)):
            y_true = np.concatenate([np.atleast_2d(yt).T if yt.ndim == 1 else yt for yt in y_true], axis=1)
        elif y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        
        Y_true = y_true[:, output_idx].reshape(X0.shape)
        
        extent = [x0.min(), x0.max(), x1.min(), x1.max()]
        cmap = self._get_colormap(output_idx)
        im = ax.imshow(Y_true, extent=extent, origin='lower', aspect='equal', cmap=cmap)
        cbar = plt.colorbar(im, ax=ax)
        self._colorbars.append(cbar)
        
        output_name = self._get_output_name(output_idx)
        ax.set_title(f'True Solution ({output_name})')
        ax.set_xlabel(self._get_input_name(0))
        ax.set_ylabel(self._get_input_name(1))
    
    def _plot_error_1d(self, ax, output_idx, n_points=200):
        """Plot 1D absolute error."""
        if self.problem.solution is None:
            return
        
        x = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points).reshape(-1, 1)
        y = self.predict(x)
        
        y_true = self.problem.solution(x, self._build_params())
        if isinstance(y_true, (list, tuple)):
            y_true = np.concatenate([np.atleast_2d(yt).T if yt.ndim == 1 else yt for yt in y_true], axis=1)
        elif y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        
        error = np.abs(y[:, output_idx] - y_true[:, output_idx])
        ax.plot(x, error, 'r-', linewidth=2)
        
        output_name = self._get_output_name(output_idx)
        input_name = self._get_input_name(0)
        ax.set_xlabel(input_name)
        ax.set_ylabel(f'|Error| ({output_name})')
        ax.set_title(f'Absolute Error ({output_name})')
        ax.grid(True, alpha=0.3)
    
    def _plot_error_2d(self, ax, output_idx, n_points=50):
        """Plot 2D absolute error as heatmap."""
        if self.problem.solution is None:
            return
        
        x0 = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points)
        x1 = np.linspace(self.problem.xmin[1], self.problem.xmax[1], n_points)
        X0, X1 = np.meshgrid(x0, x1)
        
        x_flat = np.column_stack([X0.ravel(), X1.ravel()])
        y = self.predict(x_flat)
        y_true = self.problem.solution(x_flat, self._build_params())
        
        if isinstance(y_true, (list, tuple)):
            y_true = np.concatenate([np.atleast_2d(yt).T if yt.ndim == 1 else yt for yt in y_true], axis=1)
        elif y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        
        error = np.abs(y[:, output_idx] - y_true[:, output_idx]).reshape(X0.shape)
        
        extent = [x0.min(), x0.max(), x1.min(), x1.max()]
        im = ax.imshow(error, extent=extent, origin='lower', aspect='equal', cmap='Reds')
        cbar = plt.colorbar(im, ax=ax)
        self._colorbars.append(cbar)
        
        output_name = self._get_output_name(output_idx)
        ax.set_title(f'Absolute Error ({output_name})')
        ax.set_xlabel(self._get_input_name(0))
        ax.set_ylabel(self._get_input_name(1))
    
    def _plot_region_nd(self, ax, output_idx, region, n_points=50):
        """Plot a region of an N-dimensional solution as 1D or 2D."""
        free_dims, free_ranges, fixed_dims, fixed_values = self._parse_region_nd(region)
        n_free = len(free_dims)
        
        if n_free == 0:
            # No free dimensions - just show the single point value
            x_point = np.zeros((1, self.problem.n_dims))
            for i, val in zip(fixed_dims, fixed_values):
                x_point[0, i] = val
            y = self.predict(x_point)
            ax.text(0.5, 0.5, f'u={y[0, output_idx]:.4f}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(self._get_output_name(output_idx))
            return
        
        elif n_free == 1:
            # 1D plot
            dim = free_dims[0]
            x_range = free_ranges[0]
            x_vals = np.linspace(x_range[0], x_range[1], n_points)
            
            x_full = np.zeros((n_points, self.problem.n_dims))
            x_full[:, dim] = x_vals
            for i, val in zip(fixed_dims, fixed_values):
                x_full[:, i] = val
            
            y = self.predict(x_full)
            
            ax.plot(x_vals, y[:, output_idx], linewidth=2)
            ax.set_xlabel(self._get_input_name(dim))
            ax.set_ylabel(self._get_output_name(output_idx))
            
            title_parts = [self._get_output_name(output_idx)]
            if fixed_dims:
                fixed_str = ', '.join([f'{self._get_input_name(d)}={v:.3g}' 
                                       for d, v in zip(fixed_dims, fixed_values)])
                title_parts.append(f'at {fixed_str}')
            ax.set_title(' '.join(title_parts))
            ax.grid(True, alpha=0.3)
            
        elif n_free == 2:
            # 2D plot
            dim0, dim1 = free_dims[0], free_dims[1]
            x0_range, x1_range = free_ranges[0], free_ranges[1]
            
            x0 = np.linspace(x0_range[0], x0_range[1], n_points)
            x1 = np.linspace(x1_range[0], x1_range[1], n_points)
            X0, X1 = np.meshgrid(x0, x1)
            
            n_total = X0.size
            x_full = np.zeros((n_total, self.problem.n_dims))
            x_full[:, dim0] = X0.ravel()
            x_full[:, dim1] = X1.ravel()
            for i, val in zip(fixed_dims, fixed_values):
                x_full[:, i] = val
            
            y = self.predict(x_full)
            Y = y[:, output_idx].reshape(X0.shape)
            
            extent = [x0.min(), x0.max(), x1.min(), x1.max()]
            cmap = self._get_colormap(output_idx)
            im = ax.imshow(Y, extent=extent, origin='lower', aspect='equal', cmap=cmap)
            cbar = plt.colorbar(im, ax=ax)
            self._colorbars.append(cbar)
            
            ax.set_xlabel(self._get_input_name(dim0))
            ax.set_ylabel(self._get_input_name(dim1))
            
            output_name = self._get_output_name(output_idx)
            if fixed_dims:
                fixed_str = ', '.join([f'{self._get_input_name(d)}={v:.3g}' 
                                       for d, v in zip(fixed_dims, fixed_values)])
                ax.set_title(f'{output_name} at {fixed_str}')
            else:
                ax.set_title(output_name)
        else:
            ax.text(0.5, 0.5, f'Cannot plot {n_free}D (max 2D)',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(self._get_output_name(output_idx))
    
    def _plot_region_residuals_nd(self, ax, residual_idx, region, n_points=50):
        """Plot residuals for a region of an N-dimensional problem as 1D or 2D."""
        free_dims, free_ranges, fixed_dims, fixed_values = self._parse_region_nd(region)
        n_free = len(free_dims)
        
        if n_free == 0:
            # No free dimensions - just show the single point residual value
            x_point = np.zeros((1, self.problem.n_dims))
            for i, val in zip(fixed_dims, fixed_values):
                x_point[0, i] = val
            residuals = self._compute_residuals(x_point)
            if residual_idx < len(residuals):
                res_val = np.abs(residuals[residual_idx][0])
            else:
                res_val = 0.0
            ax.text(0.5, 0.5, f'|R|={res_val:.4e}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'Residual eq{residual_idx+1}')
            return
        
        elif n_free == 1:
            # 1D plot
            dim = free_dims[0]
            x_range = free_ranges[0]
            x_vals = np.linspace(x_range[0], x_range[1], n_points)
            
            x_full = np.zeros((n_points, self.problem.n_dims))
            x_full[:, dim] = x_vals
            for i, val in zip(fixed_dims, fixed_values):
                x_full[:, i] = val
            
            residuals = self._compute_residuals(x_full)
            if residual_idx < len(residuals):
                res = np.abs(residuals[residual_idx])
            else:
                res = np.zeros(n_points)
            
            ax.plot(x_vals, res, 'm-', linewidth=2)
            ax.set_xlabel(self._get_input_name(dim))
            ax.set_ylabel(f'|Residual eq{residual_idx+1}|')
            
            title_parts = [f'Residual eq{residual_idx+1}']
            if fixed_dims:
                fixed_str = ', '.join([f'{self._get_input_name(d)}={v:.3g}' 
                                       for d, v in zip(fixed_dims, fixed_values)])
                title_parts.append(f'at {fixed_str}')
            ax.set_title(' '.join(title_parts))
            ax.grid(True, alpha=0.3)
            
        elif n_free == 2:
            # 2D plot
            dim0, dim1 = free_dims[0], free_dims[1]
            x0_range, x1_range = free_ranges[0], free_ranges[1]
            
            x0 = np.linspace(x0_range[0], x0_range[1], n_points)
            x1 = np.linspace(x1_range[0], x1_range[1], n_points)
            X0, X1 = np.meshgrid(x0, x1)
            
            n_total = X0.size
            x_full = np.zeros((n_total, self.problem.n_dims))
            x_full[:, dim0] = X0.ravel()
            x_full[:, dim1] = X1.ravel()
            for i, val in zip(fixed_dims, fixed_values):
                x_full[:, i] = val
            
            residuals = self._compute_residuals(x_full)
            if residual_idx < len(residuals):
                Res = np.abs(residuals[residual_idx]).reshape(X0.shape)
            else:
                Res = np.zeros(X0.shape)
            
            extent = [x0.min(), x0.max(), x1.min(), x1.max()]
            im = ax.imshow(Res, extent=extent, origin='lower', aspect='equal', cmap='viridis')
            cbar = plt.colorbar(im, ax=ax, label='|Residual|')
            self._colorbars.append(cbar)
            
            ax.set_xlabel(self._get_input_name(dim0))
            ax.set_ylabel(self._get_input_name(dim1))
            
            if fixed_dims:
                fixed_str = ', '.join([f'{self._get_input_name(d)}={v:.3g}' 
                                       for d, v in zip(fixed_dims, fixed_values)])
                ax.set_title(f'Res. eq{residual_idx+1} at {fixed_str}')
            else:
                ax.set_title(f'Residual eq{residual_idx+1}')
        else:
            ax.text(0.5, 0.5, f'Cannot plot {n_free}D (max 2D)',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Residual eq{residual_idx+1}')
    
    def _update_figure(self, fig, axes, n_points=200):
        """Update existing figure with current data."""
        n_dims = self.problem.n_dims
        n_outputs = self.problem.n_outputs
        has_solution = self.problem.solution is not None
        
        self._clear_colorbars()
        
        for key, ax in axes.items():
            if hasattr(ax, 'clear'):
                ax.clear()
        
        self._plot_losses(axes['losses'])
        
        if n_dims == 1:
            for i in range(n_outputs):
                if f'sol_{i}' in axes:
                    self._plot_solution_1d(axes[f'sol_{i}'], i, n_points)
                if f'res_{i}' in axes:
                    self._plot_residuals_1d(axes[f'res_{i}'], i, n_points)
                if f'err_{i}' in axes and has_solution:
                    self._plot_error_1d(axes[f'err_{i}'], i, n_points)
        
        elif n_dims == 2:
            for i in range(n_outputs):
                if f'sol_{i}' in axes:
                    self._plot_solution_2d(axes[f'sol_{i}'], i, n_points)
                if f'true_{i}' in axes and has_solution:
                    self._plot_true_solution_2d(axes[f'true_{i}'], i, n_points)
                if f'res_{i}' in axes:
                    self._plot_residuals_2d(axes[f'res_{i}'], i, n_points)
                if f'err_{i}' in axes and has_solution:
                    self._plot_error_2d(axes[f'err_{i}'], i, n_points)
        
        # Plot regions (for any dimension)
        regions = getattr(self, '_plot_regions', [])
        n_outputs = self.problem.n_outputs
        for r, region in enumerate(regions):
            # Check if we have per-output region axes (3D+ case)
            if f'region_{r}_0' in axes:
                for i in range(n_outputs):
                    if f'region_{r}_{i}' in axes:
                        self._plot_region_nd(axes[f'region_{r}_{i}'], i, region, n_points)
                    if f'region_res_{r}_{i}' in axes:
                        self._plot_region_residuals_nd(axes[f'region_res_{r}_{i}'], i, region, n_points)
            elif f'region_{r}' in axes:
                # 1D/2D case: single axis per region (plots first output)
                self._plot_region_nd(axes[f'region_{r}'], 0, region, n_points)
        
        fig.tight_layout()
    
    def plot_progress(self, save_path=None, n_points=200, fig=None, axes=None, display_handle=None):
        """
        Generate a figure with loss curves and solution plots.
        
        Args:
            save_path: Path to save the figure.
            n_points: Number of points for solution plots.
            fig: Existing figure to update.
            axes: Existing axes to update.
            display_handle: IPython display handle for in-place updates.
            
        Returns:
            tuple: (fig, axes, display_handle)
        """
        if fig is None or axes is None:
            fig, axes = self._create_figure()
        
        self._update_figure(fig, axes, n_points)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if is_notebook():
            from IPython.display import display, update_display
            if display_handle is None:
                display_handle = display(fig, display_id=True)
            else:
                display_handle.update(fig)
        # Script mode: no interactive display, just save to file
        
        return fig, axes, display_handle
    
# ==================== Residual Plotting ====================

    def _plot_residuals_1d(self, ax, output_idx, n_points=200):
        """Plot 1D PDE residuals."""
        x_np = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points).reshape(-1, 1)
        
        residuals = self._compute_residuals(x_np)
        
        if output_idx < len(residuals):
            res = np.abs(residuals[output_idx]).flatten()
        else:
            res = np.zeros(n_points)
        
        ax.plot(x_np, res, 'm-', linewidth=2)
        
        output_name = self._get_output_name(output_idx)
        input_name = self._get_input_name(0)
        ax.set_xlabel(input_name)
        ax.set_ylabel(f'|Residual| ({output_name})')
        ax.set_title(f'PDE Residual ({output_name})')
        ax.grid(True, alpha=0.3)

    def _plot_residuals_2d(self, ax, output_idx, n_points=50):
        """Plot 2D PDE residuals as heatmap."""
        x0 = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points)
        x1 = np.linspace(self.problem.xmin[1], self.problem.xmax[1], n_points)
        X0, X1 = np.meshgrid(x0, x1)
        
        x_np = np.column_stack([X0.ravel(), X1.ravel()])
        
        residuals = self._compute_residuals(x_np)
        
        if output_idx < len(residuals):
            res = np.abs(residuals[output_idx]).flatten()
        else:
            res = np.zeros(x_np.shape[0])
        
        Res = res.reshape(X0.shape)
        extent = [x0.min(), x0.max(), x1.min(), x1.max()]
        im = ax.imshow(Res, extent=extent, origin='lower', aspect='equal', cmap='viridis')
        cbar = plt.colorbar(im, ax=ax, label='|Residual|')
        self._colorbars.append(cbar)
        
        output_name = self._get_output_name(output_idx)
        ax.set_title(f'PDE Residual ({output_name})')
        ax.set_xlabel(self._get_input_name(0))
        ax.set_ylabel(self._get_input_name(1))

    # ==================== FBPINN-specific Plotting ====================

    def _get_subdomain_predictions_np(self, x_np: np.ndarray):
        """
        Get subdomain predictions and windows for FBPINN networks.
        
        Args:
            x_np: Input points as numpy array.
            
        Returns:
            Tuple of (predictions, windows) as numpy arrays, or (None, None) if not FBPINN.
            predictions shape: (n_points, n_subdomains, n_outputs)
            windows shape: (n_points, n_subdomains)
        """
        # Default: not supported. Override in subclass for FBPINN support.
        return None, None

    def _plot_fbpinn_subdomains_1d(self, ax, output_idx, n_points=200):
        """Plot individual FBPINN network predictions with their windows."""
        x_np = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points).reshape(-1, 1)
        
        predictions, windows = self._get_subdomain_predictions_np(x_np)
        
        if predictions is None:
            return  # Not an FBPINN network
        
        if hasattr(self.network, 'domain'):
            lower_bounds, upper_bounds = self.network.domain.get_subdomain_bounds()
            n_subdomains = self.network.n_subdomains
            colors = plt.cm.tab10(np.linspace(0, 1, min(n_subdomains, 10)))
            
            for i in range(n_subdomains):
                pred_i = predictions[:, i, output_idx]
                
                # Unnormalize if needed
                if hasattr(self.network, 'output_range_min') and self.network.output_range_min is not None:
                    if hasattr(self.network, 'unnormalize_output') and self.network.unnormalize_output:
                        y_min = np.array(self.network.output_range_min[output_idx])
                        y_max = np.array(self.network.output_range_max[output_idx])
                        # Handle tensor conversion if needed
                        if hasattr(y_min, 'cpu'):
                            y_min = y_min.cpu().numpy()
                            y_max = y_max.cpu().numpy()
                        pred_i = (pred_i + 1.0) / 2.0 * (y_max - y_min) + y_min
                
                window_i = windows[:, i]
                mask = window_i > 0.01
                if mask.any():
                    color = colors[i % len(colors)]
                    ax.plot(x_np[mask], pred_i[mask], '-', color=color,
                           alpha=0.4, linewidth=1.5, label=f'Net {i}' if i < 10 else None)

    def _plot_subdomain_boundaries_2d(self, ax):
        """Plot 2D subdomain boundaries as rectangles."""
        from matplotlib.patches import Rectangle
        
        if hasattr(self.network, 'domain') and hasattr(self.network.domain, 'get_subdomain_bounds'):
            lower_bounds, upper_bounds = self.network.domain.get_subdomain_bounds()
            n_subdomains = self.network.n_subdomains
            
            for i in range(n_subdomains):
                lb = lower_bounds[i]
                ub = upper_bounds[i]
                # Convert tensors to numpy if needed
                if hasattr(lb, 'cpu'):
                    lb = lb.cpu().numpy()
                    ub = ub.cpu().numpy()
                elif hasattr(lb, '__array__'):
                    lb = np.asarray(lb)
                    ub = np.asarray(ub)
                width = ub[0] - lb[0]
                height = ub[1] - lb[1]
                rect = Rectangle((lb[0], lb[1]), width, height,
                                linewidth=0.5, edgecolor="white",
                                facecolor='none', alpha=1, linestyle='--')
                ax.add_patch(rect)

    # ==================== Sampling Point Plotting ====================

    def _plot_sampling_points_1d(self, ax, cmap='viridis'):
        """Plot training sampling points on 1D axes."""
        train_color = '#15B01A'
        
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        
        if self.train_samples[0] > 0:
            x_train = self._sample_interior_np(self.train_samples[0])
            n_train = len(x_train)
            train_size = max(5, min(50, 1000 / n_train))
            y_train = np.full(n_train, y_min + 0.02 * y_range)
            ax.scatter(x_train[:, 0], y_train, s=train_size, c=train_color,
                      alpha=1, marker='|', label=f'Train ({n_train})', zorder=5)

    def _plot_sampling_points_2d(self, ax, cmap='viridis'):
        """Plot training sampling points on 2D axes."""
        train_color = '#15B01A'
        bc_color = '#15B01A'
        
        if self.train_samples[0] > 0:
            x_train = self._sample_interior_np(self.train_samples[0])
            n_train = len(x_train)
            train_size = max(1, min(20, 500 / n_train))
            ax.scatter(x_train[:, 0], x_train[:, 1], s=train_size, c=train_color,
                      alpha=1, marker='.', label=f'Train ({n_train})', zorder=5)
        
        for i, bc in enumerate(self.problem.boundary_conditions):
            if self.train_samples[i + 1] > 0:
                x_bc = self._sample_boundary_np(bc, self.train_samples[i + 1])
                n_bc = len(x_bc)
                bc_size = max(2, min(30, 300 / n_bc))
                ax.scatter(x_bc[:, 0], x_bc[:, 1], s=bc_size, c=bc_color,
                          alpha=1, marker='x', zorder=6)
    
    # ==================== Solution Error Computation ====================
    
    def _compute_solution_error(self, n_points: int = 1000) -> Optional[float]:
        """
        Compute L2 relative error between predicted and true solution.
        
        Args:
            n_points: Number of points to sample for error estimation.
            
        Returns:
            Relative L2 error as a float, or None if no solution available.
        """
        if self.problem.solution is None:
            return None
        
        # Sample points (uses abstract method - implemented by subclass)
        x = np.random.uniform(
            low=self.problem.xmin,
            high=self.problem.xmax,
            size=(n_points, self.problem.n_dims)
        )
        
        y_pred = self.predict(x)
        y_true = self.problem.solution(x, self._build_params())
        
        if isinstance(y_true, (list, tuple)):
            y_true = np.concatenate([np.atleast_2d(y).T if y.ndim == 1 else y for y in y_true], axis=1)
        elif y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        
        # Relative L2 error
        error = np.sqrt(np.mean((y_pred - y_true) ** 2))
        norm = np.sqrt(np.mean(y_true ** 2)) + 1e-10
        return float(error / norm)
