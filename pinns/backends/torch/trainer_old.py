import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Union, Any
from pinns.problem import Problem
from pinns.boundary import DirichletBC, NeumannBC, RobinBC, PointsetBC
from .functional import derivative


def _is_notebook():
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


class Trainer:
    """
    Trainer for Physics-Informed Neural Networks.
    
    Combines a problem definition with a network and handles training with
    automatic sampling from the domain and boundary conditions.
    
    Args:
        problem (Problem): The problem to solve (domain, PDE, BCs, params).
        network (nn.Module): The neural network to train (e.g., FBPINN, VanillaNetwork).
        device (str): Device to use ('cpu', 'cuda', 'mps'). Default: auto-detect.
        
    Example:
        trainer = pinns.Trainer(problem, network)
        
        trainer.compile(
            train_samples=[1000, 100, 100],
            test_samples=[1000, 0, 0],
            weights=[1.0, 1.0, 1.0],
            optimizer="adam",
            learning_rate=1e-3,
            epochs=5000,
            print_each=100,
            show_plots=True,
        )
        
        trainer.train()
        
        # Continue training with L-BFGS
        trainer.compile(optimizer="lbfgs", epochs=500)
        trainer.train()
    """
    
    def __init__(
        self,
        problem: Problem,
        network: nn.Module,
        device: str = None,
    ):
        # Auto-detect best device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon
            else:
                device = "cpu"
        
        self.problem = problem
        self.network = network.to(device)
        self.device = device
        self.dtype = torch.float32
        
        # Set normalization bounds on the network from the problem
        self._setup_network_normalization()
        
        # Default training parameters (can be overridden by compile())
        n_bcs = len(problem.boundary_conditions)
        expected_len = 1 + n_bcs
        self.train_samples = [100] + [10] * n_bcs
        self.test_samples = [100] + [0] * n_bcs
        self.weights = [1.0] * expected_len
        self.learning_rate = 1e-3
        self.optimizer_name = "adam"
        self.optimizer = None
        
        # Training run parameters (set by compile())
        self._epochs = 1000
        self._print_each = 100
        self._show_plots = False
        self._save_plots = None
        self._show_subdomains = {'solution': False, 'residuals': False, 'zoom': False}
        self._show_sampling_points = {'solution': False, 'residuals': False, 'zoom': False}
        self._profile = False
        
        # Training history (accumulated across train() calls)
        self.history = {
            'epoch': [],
            'loss': [],
            'loss_pde': [],  # List of per-equation losses or single value
            'loss_bcs': [],
            'test_loss': [],
            'solution_error': [],
        }
        
        # Global epoch counter (accumulated across train() calls)
        self._global_epoch = 0
        
        # Figure for persistent plotting
        self._fig = None
        self._axes = None
        self._display_handle = None
        
        # Random number generator
        self.rng = np.random.default_rng()
        
        # Compiled flag
        self._compiled = False
    
    def _get_bc_names(self) -> List[str]:
        """Get list of boundary condition names."""
        names = []
        for i, bc in enumerate(self.problem.boundary_conditions):
            if hasattr(bc, 'name') and bc.name is not None:
                names.append(bc.name)
            else:
                names.append(f'bc_{i}')
        return names
    
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
        show_subdomains = False,
        show_sampling_points = False,
        plot_regions: List[tuple] = None,
        plot_n_points: int = 200,
        profile: bool = False,
    ):
        """
        Configure training parameters.
        
        Can be called multiple times to change optimizer, learning rate, etc.
        Training history is preserved across compile() calls.
        
        Args:
            train_samples: Number of training samples. Can be either:
                - list: [n_interior, n_bc1, n_bc2, ...] 
                - dict: {'pde': n, 'bc_name1': n, 'bc_name2': n, ...} using BC names
            test_samples: Number of test samples with same structure as train_samples.
            weights: Loss weights. Can be either:
                - list: [w_pde, w_bc1, w_bc2, ...]
                - dict: {'pde': w, 'bc_name1': w, 'bc_name2': w, ...} using BC names
            optimizer (str): Optimizer name ('adam', 'lbfgs', 'sgd').
            learning_rate (float): Learning rate.
            epochs (int): Number of training epochs for next train() call.
            print_each (int): Print progress every N epochs.
            show_plots (bool): Whether to display plots.
            save_plots (str): Path for saving plots. Behavior depends on value:
                - None (default): In notebooks, displays inline. In scripts with show_plots=True,
                  auto-saves to './pinn_progress_X.png' where X is auto-incremented to avoid
                  overwriting previous runs (updated each print_each epochs).
                - String path (e.g., './results'): Saves epoch-suffixed files like
                  'results_epoch00500.png' for training history.
            show_subdomains (bool or dict): For FBPINN, show subdomain predictions.
                Can be bool (applies to all) or dict with keys 'solution', 'residuals', 'zoom'.
            show_sampling_points (bool or dict): Show training/test sampling points on plots.
                Can be bool (applies to all) or dict with keys 'solution', 'residuals', 'zoom'.
            plot_regions (list): List of regions to show as additional plots.
                                Each region is an N-element tuple (one per input dimension).
                                Each element can be:
                                  - None: use full domain range (free dimension for plotting)
                                  - (min, max): specific range (free dimension, zoomed)
                                  - scalar: fix dimension at this value (slice)
                                The resulting plot is 1D or 2D based on the number of free dimensions.
                                
                                Examples for 1D problem:
                                  [(0.0, 0.5)] - zoom to x in [0, 0.5]
                                
                                Examples for 2D problem:
                                  [((-.1, .1), None)] - zoom x to [-.1, .1], full y range
                                  
                                Examples for 3D problem (x, y, t):
                                  [(None, None, 0.05)] - x-y plane at t=0.05
                                  [(0.0, None, None)] - y-t plane at x=0.0
                                  [((-0.5, 0.5), None, 0.1)] - zoomed x, full y, at t=0.1
                                
                                Examples for 4D+ problems:
                                  [(None, None, 0.5, 0.0)] - 2D slice fixing dims 2 and 3
                                  [(None, 0.0, 0.0, 0.0)] - 1D slice fixing dims 1, 2, 3
                                
                                For N>=3D, if no plot_regions specified, only loss curves are shown.
            plot_n_points (int): Number of points per dimension for plots. For 2D problems,
                                creates a grid of n x n points. Default 200.
            profile (bool): Print timing breakdown after training.
            batch_size (int): Maximum mini-batch size for gradient accumulation. If None, 
                             processes all samples in a single forward/backward pass. When set,
                             the number of batches is determined by the largest sample count
                             (PDE or any BC) divided by batch_size. All sample types (PDE and BCs)
                             are then distributed evenly across these batches, ensuring no single
                             batch exceeds batch_size for any sample type. This reduces peak GPU 
                             memory usage by freeing computation graphs after each batch's backward
                             pass. Gradients are accumulated across all batches before the optimizer
                             step. Default: None (no mini-batching).
                             
                             Example: With batch_size=500, pde=1000, bc=5000:
                               - n_batches = ceil(5000/500) = 10
                               - PDE: 100 samples/batch, BC: 500 samples/batch
            
        Note:
            Interior sampling method and transform are configured on the DomainCubic class.
            Boundary condition sampling is configured on each BC class.
        """
        n_bcs = len(self.problem.boundary_conditions)
        expected_len = 1 + n_bcs
        
        # Update sampling if provided
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
        
        # Update optimizer if changed
        optimizer_changed = False
        if optimizer is not None and optimizer.lower() != self.optimizer_name:
            self.optimizer_name = optimizer.lower()
            optimizer_changed = True
        
        if learning_rate is not None and learning_rate != self.learning_rate:
            self.learning_rate = learning_rate
            optimizer_changed = True
        
        # Create/recreate optimizer if needed
        if self.optimizer is None or optimizer_changed:
            self.optimizer = self._create_optimizer()
        
        # Store training run parameters
        self._epochs = epochs
        self._print_each = print_each
        self._show_plots = show_plots
        self._save_plots = save_plots
        
        # Expand bool to dict for show_subdomains
        if isinstance(show_subdomains, bool):
            self._show_subdomains = {'solution': show_subdomains, 'residuals': show_subdomains, 'zoom': show_subdomains}
        else:
            self._show_subdomains = {'solution': False, 'residuals': False, 'zoom': False}
            self._show_subdomains.update(show_subdomains)
        
        # Expand bool to dict for show_sampling_points
        if isinstance(show_sampling_points, bool):
            self._show_sampling_points = {'solution': show_sampling_points, 'residuals': show_sampling_points, 'zoom': show_sampling_points}
        else:
            self._show_sampling_points = {'solution': False, 'residuals': False, 'zoom': False}
            self._show_sampling_points.update(show_sampling_points)
        
        self._plot_regions = plot_regions if plot_regions is not None else []
        self._plot_n_points = plot_n_points
        self._profile = profile
        self._batch_size = batch_size
        
        self._compiled = True
    
    def _setup_network_normalization(self):
        """Set up input/output normalization on the network from problem definition."""
        from .networks import FNN, FBPINN
        
        # Set input normalization from domain bounds
        xmin = np.array(self.problem.xmin)
        xmax = np.array(self.problem.xmax)
        
        if isinstance(self.network, FNN):
            # FNN: set input range from domain
            if hasattr(self.network, 'normalize_input') and self.network.normalize_input:
                self.network.set_input_range(xmin, xmax)
            
            # Set output range from problem if specified
            if hasattr(self.network, 'unnormalize_output') and self.network.unnormalize_output:
                if self.problem.output_range is not None:
                    output_range = self.problem.output_range
                    if isinstance(output_range, list):
                        # Per-output ranges
                        ymin = np.array([r[0] if r is not None else -1.0 for r in output_range])
                        ymax = np.array([r[1] if r is not None else 1.0 for r in output_range])
                    else:
                        # Single range for all outputs
                        ymin, ymax = output_range
                    self.network.set_output_range(ymin, ymax)
        
        elif isinstance(self.network, FBPINN):
            # FBPINN: input normalization is per-subdomain (handled internally)
            # Set output range from problem if specified
            if hasattr(self.network, 'unnormalize_output') and self.network.unnormalize_output:
                if self.problem.output_range is not None:
                    output_range = self.problem.output_range
                    if isinstance(output_range, list):
                        # Per-output ranges
                        ymin = np.array([r[0] if r is not None else -1.0 for r in output_range])
                        ymax = np.array([r[1] if r is not None else 1.0 for r in output_range])
                    else:
                        # Single range for all outputs
                        ymin, ymax = output_range
                    self.network.set_output_range(ymin, ymax)
    
    def reset(self):
        """
        Reset training history and epoch counter.
        
        Call this to start fresh training while keeping the same problem/network.
        """
        self.history = {
            'epoch': [],
            'loss': [],
            'loss_pde': [],  # List of per-equation losses or single value
            'loss_bcs': [],
            'test_loss': [],
            'solution_error': [],
        }
        self._global_epoch = 0
        self._fig = None
        self._axes = None
        self._display_handle = None
        self._compiled = False
    
    def _create_optimizer(self):
        """Create the optimizer."""
        if self.optimizer_name == "adam":
            return torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "lbfgs":
            return torch.optim.LBFGS(
                self.network.parameters(), 
                lr=self.learning_rate,
                max_iter=5,
                history_size=50,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-9,
                tolerance_change=1e-12
            )
        elif self.optimizer_name == "sgd":
            return torch.optim.SGD(self.network.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
    
    def _sample_interior(self, n_points: int) -> torch.Tensor:
        """Sample interior points from the domain.
        
        Uses the domain's sampling_method and sampling_transform settings.
        
        Args:
            n_points: Number of points to sample.
        """
        # Build params dict for sampler
        params = {
            "fixed": self.problem.params,
            "infer": {},
            "internal": {}
        }
        points = self.problem.domain.sample_interior(n_points, rng=self.rng, params=params)
        return torch.tensor(points, device=self.device, dtype=self.dtype, requires_grad=True)
    
    def _sample_boundary(self, bc: Union[DirichletBC, NeumannBC, RobinBC], 
                         n_points: int) -> torch.Tensor:
        """Sample boundary points for a specific boundary condition.
        
        If bc has a subdomain attribute, samples will be constrained to that subdomain.
        The subdomain is specified as (min_value, max_value) representing the actual
        coordinate limits for the free (non-fixed) dimension of the boundary.
        
        Uses bc.sampling_method and bc.sampling_transform if available.
        """
        if isinstance(bc, PointsetBC):
            # PointsetBC has fixed points
            return torch.tensor(bc.points, device=self.device, dtype=self.dtype, requires_grad=True)
        
        # Get boundary specification
        boundary = bc.boundary
        domain = self.problem.domain
        
        # Get sampling method and transform from BC (or defaults)
        method = getattr(bc, 'sampling_method', 'uniform')
        transform = getattr(bc, 'sampling_transform', None)
        
        # Build params dict for sampler
        params = {
            "fixed": self.problem.params,
            "infer": {},
            "internal": {}
        }
        
        # Find which dimension and side this boundary is on
        for dim, side in enumerate(boundary):
            if side is not None:
                # Sample on this boundary with the BC's sampling method
                points = domain.sample_boundary(
                    n_points, dim, side, rng=self.rng,
                    method=method, transform=transform, params=params
                )
                
                # Apply subdomain constraint if specified
                subdomain = getattr(bc, 'subdomain', None)
                if subdomain is not None:
                    sub_min, sub_max = subdomain
                    # Find the free dimension (the one that is None in boundary spec)
                    for d in range(len(boundary)):
                        if boundary[d] is None:  # This is the free dimension
                            # Get the original sampling range for this dimension
                            orig_min = domain.xmin[d]
                            orig_max = domain.xmax[d]
                            orig_extent = orig_max - orig_min
                            # Rescale: map [orig_min, orig_max] -> [sub_min, sub_max]
                            normalized = (points[:, d] - orig_min) / orig_extent  # in [0, 1]
                            points[:, d] = sub_min + normalized * (sub_max - sub_min)
                
                return torch.tensor(points, device=self.device, dtype=self.dtype, requires_grad=True)
        
        raise ValueError(f"Invalid boundary specification: {boundary}")
    
    def _compute_pde_loss(self, x: torch.Tensor, internal: Dict[str, Any] = None):
        """Compute PDE residual loss.
        
        Args:
            x: Interior points tensor
            internal: Internal training state dict with 'global_step' and 'step'
            
        Returns:
            Tuple of (total_loss, individual_losses) where individual_losses is a list
        """
        params = self._build_params(internal)
        y = self.network(x, params)
        residual = self.problem.compute_pde_residual(x, y, internal)
        
        # Get PDE weights (can be single value or list for multiple equations)
        pde_weight = self.weights[0]
        
        # Handle both single residual tensor and list of residuals (for multiple PDEs)
        if isinstance(residual, (list, tuple)):
            # Compute individual losses
            individual_losses = [torch.mean(r ** 2) for r in residual]
            
            # Apply per-equation weights if provided
            if isinstance(pde_weight, (list, tuple)):
                if len(pde_weight) != len(residual):
                    raise ValueError(
                        f"Number of PDE weights ({len(pde_weight)}) must match "
                        f"number of PDE equations ({len(residual)})"
                    )
                total_loss = sum(w * l for w, l in zip(pde_weight, individual_losses))
            else:
                # Single weight applied to sum of all equations
                total_loss = sum(individual_losses)
            return total_loss, individual_losses
        else:
            # Single equation
            loss = torch.mean(residual ** 2)
            return loss, [loss]
    
    def _compute_bc_loss(self, bc, x: torch.Tensor) -> torch.Tensor:
        """Compute boundary condition loss for a single BC."""
        params = self._build_params()
        y = self.network(x, params)
        
        if isinstance(bc, DirichletBC):
            target = bc.get_value(x)
            return torch.mean((y[:, bc.component] - target) ** 2)
        
        elif isinstance(bc, NeumannBC):
            normal_dim, normal_sign = bc.get_normal_direction()
            u = y[:, bc.component:bc.component + 1]
            grads = torch.autograd.grad(
                u, x, grad_outputs=torch.ones_like(u),
                create_graph=True, retain_graph=True
            )[0]
            du_dn = grads[:, normal_dim]
            target = bc.get_value(x)
            return torch.mean((normal_sign * du_dn - target) ** 2)
        
        elif isinstance(bc, RobinBC):
            normal_dim, normal_sign = bc.get_normal_direction()
            u = y[:, bc.component:bc.component + 1]
            grads = torch.autograd.grad(
                u, x, grad_outputs=torch.ones_like(u),
                create_graph=True, retain_graph=True
            )[0]
            du_dn = grads[:, normal_dim]
            alpha, beta, gamma = bc.get_coefficients(x)
            residual = alpha * y[:, bc.component] + beta * normal_sign * du_dn - gamma
            return torch.mean(residual ** 2)
        
        elif isinstance(bc, PointsetBC):
            target = torch.tensor(bc.values, device=self.device, dtype=self.dtype)
            return torch.mean((y[:, bc.component] - target) ** 2)
        
        else:
            raise ValueError(f"Unknown BC type: {type(bc)}")
    
    def _compute_total_loss(self, x_interior=None, x_bcs=None, internal=None):
        """Compute total weighted loss.
        
        Args:
            x_interior: Pre-sampled interior points (optional, will sample if None)
            x_bcs: Pre-sampled boundary points list (optional, will sample if None)
            internal: Internal training state dict with 'global_step' and 'step'
        """
        losses = {}
        
        # PDE loss
        if self.train_samples[0] > 0:
            if x_interior is None:
                x_interior = self._sample_interior(self.train_samples[0])
            pde_loss, pde_individual = self._compute_pde_loss(x_interior, internal)
            # Apply weight only if it's a scalar (list weights are applied inside _compute_pde_loss)
            if not isinstance(self.weights[0], (list, tuple)):
                pde_loss = self.weights[0] * pde_loss
            losses['pde'] = pde_loss
            losses['pde_individual'] = pde_individual
        else:
            losses['pde'] = torch.tensor(0.0, device=self.device)
            losses['pde_individual'] = [torch.tensor(0.0, device=self.device)]
        
        # BC losses
        losses['bcs'] = []
        for i, bc in enumerate(self.problem.boundary_conditions):
            n_samples = self.train_samples[i + 1]
            if n_samples > 0:
                if x_bcs is None:
                    x_bc = self._sample_boundary(bc, n_samples)
                else:
                    x_bc = x_bcs[i]
                bc_loss = self.weights[i + 1] * self._compute_bc_loss(bc, x_bc)
                losses['bcs'].append(bc_loss)
            else:
                losses['bcs'].append(torch.tensor(0.0, device=self.device))
        
        # Total loss
        total = losses['pde'] + sum(losses['bcs'])
        
        return total, losses
    
    def _compute_test_loss(self):
        """Compute test loss (without gradients)."""
        with torch.no_grad():
            losses = {}
            
            # PDE loss
            if self.test_samples[0] > 0:
                x_interior = self._sample_interior(self.test_samples[0])
                x_interior.requires_grad_(True)
                with torch.enable_grad():
                    pde_total, _ = self._compute_pde_loss(x_interior)
                    losses['pde'] = pde_total.item()
            else:
                losses['pde'] = 0.0
            
            # BC losses
            losses['bcs'] = []
            for i, bc in enumerate(self.problem.boundary_conditions):
                n_samples = self.test_samples[i + 1]
                if n_samples > 0:
                    x_bc = self._sample_boundary(bc, n_samples)
                    x_bc.requires_grad_(True)
                    with torch.enable_grad():
                        bc_loss = self._compute_bc_loss(bc, x_bc).item()
                    losses['bcs'].append(bc_loss)
                else:
                    losses['bcs'].append(0.0)
            
            total = losses['pde'] + sum(losses['bcs'])
            
        return total, losses
    
    def _compute_solution_error(self, n_points: int = 1000) -> float:
        """
        Compute L2 relative error between predicted and true solution.
        
        Args:
            n_points: Number of points to sample for error estimation.
            
        Returns:
            Relative L2 error as a float.
        """
        if self.problem.solution is None:
            return None
        
        with torch.no_grad():
            x = self._sample_interior(n_points)
            params = self._build_params()
            y_pred = self.network(x, params).cpu().numpy()
            x_np = x.cpu().numpy()
            y_true = self.problem.solution(x_np, self._build_params())
            
            # Handle list of outputs (for multi-output problems)
            if isinstance(y_true, (list, tuple)):
                y_true = np.concatenate([np.atleast_2d(y).T if y.ndim == 1 else y for y in y_true], axis=1)
            elif y_true.ndim == 1:
                y_true = y_true.reshape(-1, 1)
            
            # Relative L2 error
            error = np.sqrt(np.mean((y_pred - y_true) ** 2))
            norm = np.sqrt(np.mean(y_true ** 2)) + 1e-10
            return float(error / norm)
    
    def _get_colormap(self, output_idx):
        """Get colormap based on output range symmetry.
        
        Returns a diverging colormap (RdBu_r) if the output range is symmetric
        around zero (e.g., (-1, 1), (-0.5, 0.5)), otherwise returns 'viridis'.
        """
        if self.problem.output_range is None:
            return 'inferno'
        elif self.problem.output_range[output_idx][0] == -self.problem.output_range[output_idx][1]:
            return 'managua_r'
        else:
            return 'inferno'
    
    def _plot_losses(self, ax):
        """Plot loss curves on given axes."""
        epochs = self.history['epoch']
        
        # Total loss
        ax.semilogy(epochs, self.history['loss'], 'k-', label='Total', linewidth=2)
        
        # PDE losses (can be list of per-equation losses or single values)
        pde_losses = self.history['loss_pde']
        if len(pde_losses) > 0:
            # Check if it's a list of lists (per-equation) or list of scalars
            if isinstance(pde_losses[0], (list, tuple)):
                pde_array = np.array(pde_losses)
                for i in range(pde_array.shape[1]):
                    ax.semilogy(epochs, pde_array[:, i], '--', label=f'PDE eq{i+1}')
            else:
                ax.semilogy(epochs, pde_losses, '--', label='PDE')
        
        # BC losses with names
        bc_names = self._get_bc_names()
        bc_losses_array = np.array(self.history['loss_bcs'])
        for i in range(bc_losses_array.shape[1]):
            bc_label = bc_names[i] if i < len(bc_names) else f'BC {i+1}'
            ax.semilogy(epochs, bc_losses_array[:, i], '--', label=bc_label)
        
        # Test loss if available (sampled at different intervals)
        if len(self.history['test_loss']) > 0:
            # Test loss is recorded less frequently, plot with markers
            n_test = len(self.history['test_loss'])
            test_epochs = np.linspace(epochs[0], epochs[-1], n_test).astype(int) if n_test > 1 else [epochs[-1]]
            ax.semilogy(test_epochs, self.history['test_loss'], 'r:', marker='o', markersize=4, label='Test', linewidth=2)
        
        # Solution error if available (sampled at different intervals)
        if len(self.history['solution_error']) > 0:
            n_err = len(self.history['solution_error'])
            err_epochs = np.linspace(epochs[0], epochs[-1], n_err).astype(int) if n_err > 1 else [epochs[-1]]
            ax.semilogy(err_epochs, self.history['solution_error'], 'm-', marker='s', markersize=4, label='Solution Error', linewidth=2)
        
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
            # Handle list of outputs (for multi-output problems)
            if isinstance(y_true, (list, tuple)):
                y_true = np.concatenate([np.atleast_2d(yt).T if yt.ndim == 1 else yt for yt in y_true], axis=1)
            elif y_true.ndim == 1:
                y_true = y_true.reshape(-1, 1)
            ax.plot(x, y_true[:, output_idx], 'r-', linewidth=2, label='True')
            ax.plot(x, y[:, output_idx], 'b--', linewidth=2, label='Predicted')
            ax.legend(loc='best', fontsize=8)
        else:
            ax.plot(x, y[:, output_idx], 'b-', linewidth=2)
        
        # If FBPINN, also plot individual network predictions (if enabled)
        show_subdomains = getattr(self, '_show_subdomains', {})
        if show_subdomains.get('solution', False) and hasattr(self.network, 'get_subdomain_predictions'):
            self._plot_fbpinn_subdomains_1d(ax, output_idx, n_points)
        
        output_name = self._get_output_name(output_idx)
        input_name = self._get_input_name(0)
        ax.set_xlabel(input_name)
        ax.set_ylabel(output_name)
        ax.set_title(f'Solution ({output_name})')
        ax.grid(True, alpha=0.3)
        
        # Show sampling points if enabled
        show_sampling = getattr(self, '_show_sampling_points', {})
        if show_sampling.get('solution', False):
            cmap = self._get_colormap(output_idx)
            self._plot_sampling_points_1d(ax, cmap)
    
    def _plot_fbpinn_subdomains_1d(self, ax, output_idx, n_points=200):
        """Plot individual FBPINN network predictions with their windows."""
        import torch
        
        x_np = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points).reshape(-1, 1)
        x_tensor = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        
        self.network.eval()
        with torch.no_grad():
            predictions, windows = self.network.get_subdomain_predictions(x_tensor)
            # predictions: (n_points, n_subdomains, n_outputs)
            # windows: (n_points, n_subdomains)
            
            # Get subdomain bounds for color coding
            lower_bounds, upper_bounds = self.network.domain.get_subdomain_bounds()
            
            # Use a colormap for different subdomains
            n_subdomains = self.network.n_subdomains
            colors = plt.cm.tab10(np.linspace(0, 1, min(n_subdomains, 10)))
            
            for i in range(n_subdomains):
                # Get raw network output and unnormalize
                pred_i = predictions[:, i, output_idx].cpu().numpy()
                
                # Unnormalize like the FBPINN does
                if self.network.output_range_min is not None and self.network.unnormalize_output:
                    y_min = self.network.output_range_min[output_idx].cpu().numpy()
                    y_max = self.network.output_range_max[output_idx].cpu().numpy()
                    pred_i = (pred_i + 1.0) / 2.0 * (y_max - y_min) + y_min
                
                window_i = windows[:, i].cpu().numpy()
                
                # Only plot where window is significant
                mask = window_i > 0.01
                if mask.any():
                    color = colors[i % len(colors)]
                    # Plot with alpha based on window value
                    ax.plot(x_np[mask], pred_i[mask], '-', color=color, 
                           alpha=0.4, linewidth=1.5, label=f'Net {i}' if i < 10 else None)
    
    def _plot_solution_2d(self, ax, output_idx, n_points=50):
        """Plot 2D solution as contour on given axes."""
        x0 = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points)
        x1 = np.linspace(self.problem.xmin[1], self.problem.xmax[1], n_points)
        X0, X1 = np.meshgrid(x0, x1)
        
        x_flat = np.column_stack([X0.ravel(), X1.ravel()])
        y = self.predict(x_flat)
        Y = y[:, output_idx].reshape(X0.shape)
        
        # Predicted solution as heatmap
        extent = [x0.min(), x0.max(), x1.min(), x1.max()]
        cmap = self._get_colormap(output_idx)
        im = ax.imshow(Y, extent=extent, origin='lower', aspect='auto', cmap=cmap)
        cbar = plt.colorbar(im, ax=ax)
        self._colorbars.append(cbar)
        
        output_name = self._get_output_name(output_idx)
        ax.set_title(f'Predicted ({output_name})')
        
        # Show subdomain partitions if FBPINN and enabled
        show_subdomains = getattr(self, '_show_subdomains', {})
        if show_subdomains.get('solution', False) and hasattr(self.network, 'domain'):
            self._plot_subdomain_boundaries_2d(ax)
        
        # Show sampling points if enabled
        show_sampling = getattr(self, '_show_sampling_points', {})
        if show_sampling.get('solution', False):
            self._plot_sampling_points_2d(ax, cmap)
        
        ax.set_xlabel(self._get_input_name(0))
        ax.set_ylabel(self._get_input_name(1))
    
    def _plot_true_solution_2d(self, ax, output_idx, n_points=50):
        """Plot 2D true solution as contour on given axes."""
        x0 = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points)
        x1 = np.linspace(self.problem.xmin[1], self.problem.xmax[1], n_points)
        X0, X1 = np.meshgrid(x0, x1)
        
        x_flat = np.column_stack([X0.ravel(), X1.ravel()])
        y_true = self.problem.solution(x_flat, self._build_params())
        
        # Handle list of outputs (for multi-output problems)
        if isinstance(y_true, (list, tuple)):
            y_true = np.concatenate([np.atleast_2d(yt).T if yt.ndim == 1 else yt for yt in y_true], axis=1)
        elif y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        
        Y_true = y_true[:, output_idx].reshape(X0.shape)
        
        # True solution as heatmap
        extent = [x0.min(), x0.max(), x1.min(), x1.max()]
        cmap = self._get_colormap(output_idx)
        im = ax.imshow(Y_true, extent=extent, origin='lower', aspect='auto', cmap=cmap)
        cbar = plt.colorbar(im, ax=ax)
        self._colorbars.append(cbar)
        
        output_name = self._get_output_name(output_idx)
        ax.set_title(f'True Solution ({output_name})')
        ax.set_xlabel(self._get_input_name(0))
        ax.set_ylabel(self._get_input_name(1))
    
    def _plot_sampling_points_2d(self, ax, cmap='viridis'):
        """Plot training and test sampling points on 2D axes.
        """
        # Choose colors based on colormap
        train_color = '#15B01A'
        test_color = '#15B01A'
        bc_color = '#15B01A'
        
        # Sample training points
        if self.train_samples[0] > 0:
            x_train = self._sample_interior(self.train_samples[0]).detach().cpu().numpy()
            # Adaptive marker size: smaller for more points
            n_train = len(x_train)
            train_size = max(1, min(20, 500 / n_train))
            ax.scatter(x_train[:, 0], x_train[:, 1], s=train_size, c=train_color, 
                      alpha=1, marker='.', label=f'Train ({n_train})', zorder=5)
        
        # Sample test points
        if self.test_samples[0] > 0:
            x_test = self._sample_interior(self.test_samples[0]).detach().cpu().numpy()
            n_test = len(x_test)
            test_size = max(1, min(20, 500 / n_test))
            ax.scatter(x_test[:, 0], x_test[:, 1], s=test_size, c=test_color, 
                      alpha=1, marker='x', label=f'Test ({n_test})', zorder=5)
        
        # Sample BC points
        for i, bc in enumerate(self.problem.boundary_conditions):
            if self.train_samples[i + 1] > 0:
                x_bc = self._sample_boundary(bc, self.train_samples[i + 1]).detach().cpu().numpy()
                n_bc = len(x_bc)
                bc_size = max(2, min(30, 300 / n_bc))
                ax.scatter(x_bc[:, 0], x_bc[:, 1], s=bc_size, c=bc_color, 
                          alpha=1, marker='x', zorder=6)
    
    def _plot_sampling_points_1d(self, ax, cmap='viridis'):
        """Plot training and test sampling points on 1D axes.
        """
        train_color = '#15B01A'
        test_color = 'white'
        
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        
        # Sample training points
        if self.train_samples[0] > 0:
            x_train = self._sample_interior(self.train_samples[0]).detach().cpu().numpy()
            n_train = len(x_train)
            train_size = max(5, min(50, 1000 / n_train))
            # Plot at bottom of axes
            y_train = np.full(n_train, y_min + 0.02 * y_range)
            ax.scatter(x_train[:, 0], y_train, s=train_size, c=train_color, 
                      alpha=1, marker='|', label=f'Train ({n_train})', zorder=5)
        
        # Sample test points
        if self.test_samples[0] > 0:
            x_test = self._sample_interior(self.test_samples[0]).detach().cpu().numpy()
            n_test = len(x_test)
            test_size = max(5, min(50, 1000 / n_test))
            y_test = np.full(n_test, y_min + 0.05 * y_range)
            ax.scatter(x_test[:, 0], y_test, s=test_size, c=test_color, 
                      alpha=1, marker='|', label=f'Test ({n_test})', zorder=5)
    
    def _plot_subdomain_boundaries_2d(self, ax):
        """Plot 2D subdomain boundaries as rectangles."""
        from matplotlib.patches import Rectangle
        
        lower_bounds, upper_bounds = self.network.domain.get_subdomain_bounds()
        n_subdomains = self.network.n_subdomains
        colors = plt.cm.tab10(np.linspace(0, 1, min(n_subdomains, 10)))
        
        for i in range(n_subdomains):
            lb = lower_bounds[i]
            ub = upper_bounds[i]
            # Handle both torch tensors and numpy arrays
            if hasattr(lb, 'cpu'):
                lb = lb.cpu().numpy()
                ub = ub.cpu().numpy()
            width = ub[0] - lb[0]
            height = ub[1] - lb[1]
            rect = Rectangle((lb[0], lb[1]), width, height, 
                            linewidth=0.5, edgecolor="white", 
                            facecolor='none', alpha=1, linestyle='--')
            ax.add_patch(rect)
    
    def _plot_solution_1d_region(self, ax, output_idx, region, n_points=200):
        """Plot 1D solution in a zoomed region."""
        # Parse region: (xmin, xmax) for 1D
        if region is None:
            xmin, xmax = self.problem.xmin[0], self.problem.xmax[0]
        else:
            xmin, xmax = region
            if xmin is None:
                xmin = self.problem.xmin[0]
            if xmax is None:
                xmax = self.problem.xmax[0]
        
        x = np.linspace(xmin, xmax, n_points).reshape(-1, 1)
        y = self.predict(x)
        
        # Plot predicted
        ax.plot(x, y[:, output_idx], 'b-', linewidth=2, label='Predicted')
        
        # Plot true solution if available
        if self.problem.solution is not None:
            y_true = self.problem.solution(x, self._build_params())
            if isinstance(y_true, (list, tuple)):
                y_true = np.concatenate([np.atleast_2d(yt).T if yt.ndim == 1 else yt for yt in y_true], axis=1)
            elif y_true.ndim == 1:
                y_true = y_true.reshape(-1, 1)
            ax.plot(x, y_true[:, output_idx], 'r--', linewidth=2, label='True')
            ax.legend()
        
        output_name = self._get_output_name(output_idx)
        input_name = self._get_input_name(0)
        ax.set_xlabel(input_name)
        ax.set_ylabel(output_name)
        ax.set_title(f'Zoom [{xmin:.3g}, {xmax:.3g}]')
        ax.grid(True, alpha=0.3)
        
        # Show FBPINN subdomain predictions if enabled
        show_subdomains = getattr(self, '_show_subdomains', {})
        if show_subdomains.get('zoom', False) and hasattr(self.network, 'get_subdomain_predictions'):
            self._plot_fbpinn_subdomains_1d(ax, output_idx, n_points)
        
        # Show sampling points if enabled (preserve zoom limits)
        show_sampling = getattr(self, '_show_sampling_points', {})
        if show_sampling.get('zoom', False):
            cmap = self._get_colormap(output_idx)
            self._plot_sampling_points_1d(ax, cmap)
            # Restore zoom x-limits after plotting (sampling points are from full domain)
            ax.set_xlim(xmin, xmax)
    
    def _parse_region_nd(self, region):
        """Parse an N-dimensional region specification.
        
        Args:
            region: Tuple with one element per dimension. Each element can be:
                - None: use full range for this dimension (free dimension)
                - (min, max): range for this dimension (free dimension, zoomed)
                - scalar: fix this dimension at that value (sliced dimension)
                
        Returns:
            tuple: (free_dims, free_ranges, fixed_dims, fixed_values)
                - free_dims: list of dimension indices that are free
                - free_ranges: list of (min, max) tuples for free dimensions
                - fixed_dims: list of dimension indices that are fixed
                - fixed_values: list of scalar values for fixed dimensions
        """
        n_dims = self.problem.n_dims
        
        # Default: all dimensions free with full range
        if region is None:
            region = [None] * n_dims
        
        free_dims = []
        free_ranges = []
        fixed_dims = []
        fixed_values = []
        
        for i, spec in enumerate(region):
            if spec is None:
                # Full range for this dimension
                free_dims.append(i)
                free_ranges.append((self.problem.xmin[i], self.problem.xmax[i]))
            elif isinstance(spec, (list, tuple)) and len(spec) == 2:
                # Range specified
                free_dims.append(i)
                free_ranges.append((spec[0], spec[1]))
            else:
                # Scalar value - fixed dimension
                fixed_dims.append(i)
                fixed_values.append(float(spec))
        
        return free_dims, free_ranges, fixed_dims, fixed_values
    
    def _plot_region_nd(self, ax, output_idx, region, n_points=50):
        """Plot a region of an N-dimensional solution as 1D or 2D.
        
        Args:
            ax: Matplotlib axes
            output_idx: Which output to plot
            region: N-element tuple specifying the region (see _parse_region_nd)
            n_points: Number of points per free dimension
        """
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
            
            # Build full input array
            x_full = np.zeros((n_points, self.problem.n_dims))
            x_full[:, dim] = x_vals
            for i, val in zip(fixed_dims, fixed_values):
                x_full[:, i] = val
            
            y = self.predict(x_full)
            
            ax.plot(x_vals, y[:, output_idx], linewidth=2)
            ax.set_xlabel(self._get_input_name(dim))
            ax.set_ylabel(self._get_output_name(output_idx))
            
            # Build title showing fixed dimensions
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
            
            # Build full input array
            n_total = X0.size
            x_full = np.zeros((n_total, self.problem.n_dims))
            x_full[:, dim0] = X0.ravel()
            x_full[:, dim1] = X1.ravel()
            for i, val in zip(fixed_dims, fixed_values):
                x_full[:, i] = val
            
            y = self.predict(x_full)
            Y = y[:, output_idx].reshape(X0.shape)
            
            # Plot as heatmap
            extent = [x0.min(), x0.max(), x1.min(), x1.max()]
            cmap = self._get_colormap(output_idx)
            im = ax.imshow(Y, extent=extent, origin='lower', aspect='auto', cmap=cmap)
            cbar = plt.colorbar(im, ax=ax)
            self._colorbars.append(cbar)
            
            ax.set_xlabel(self._get_input_name(dim0))
            ax.set_ylabel(self._get_input_name(dim1))
            
            # Build title showing fixed dimensions
            output_name = self._get_output_name(output_idx)
            if fixed_dims:
                fixed_str = ', '.join([f'{self._get_input_name(d)}={v:.3g}' 
                                       for d, v in zip(fixed_dims, fixed_values)])
                ax.set_title(f'{output_name} at {fixed_str}')
            else:
                ax.set_title(output_name)
        else:
            # More than 2 free dimensions - cannot visualize
            ax.text(0.5, 0.5, f'Cannot plot {n_free}D (max 2D)',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(self._get_output_name(output_idx))
    
    def _plot_solution_3d_slice(self, ax, output_idx, region, n_points=50):
        """Plot 2D slice of 3D solution (legacy wrapper for _plot_region_nd)."""
        self._plot_region_nd(ax, output_idx, region, n_points)
    
    def _plot_solution_2d_region(self, ax, output_idx, region, n_points=50):
        """Plot 2D solution in a zoomed region."""
        # Parse region: ((x0min, x0max), (x1min, x1max)) for 2D
        if region is None:
            x0_range = (self.problem.xmin[0], self.problem.xmax[0])
            x1_range = (self.problem.xmin[1], self.problem.xmax[1])
        else:
            x0_range, x1_range = region
            if x0_range is None:
                x0_range = (self.problem.xmin[0], self.problem.xmax[0])
            if x1_range is None:
                x1_range = (self.problem.xmin[1], self.problem.xmax[1])
        
        x0 = np.linspace(x0_range[0], x0_range[1], n_points)
        x1 = np.linspace(x1_range[0], x1_range[1], n_points)
        X0, X1 = np.meshgrid(x0, x1)
        
        x_flat = np.column_stack([X0.ravel(), X1.ravel()])
        y = self.predict(x_flat)
        Y = y[:, output_idx].reshape(X0.shape)
        
        # Predicted solution as heatmap
        extent = [x0.min(), x0.max(), x1.min(), x1.max()]
        cmap = self._get_colormap(output_idx)
        im = ax.imshow(Y, extent=extent, origin='lower', aspect='auto', cmap=cmap)
        cbar = plt.colorbar(im, ax=ax)
        self._colorbars.append(cbar)
        
        # Show subdomain boundaries if FBPINN and enabled
        show_subdomains = getattr(self, '_show_subdomains', {})
        if show_subdomains.get('zoom', False) and hasattr(self.network, 'domain'):
            self._plot_subdomain_boundaries_2d(ax)
        
        # Show sampling points if enabled (preserve zoom limits)
        show_sampling = getattr(self, '_show_sampling_points', {})
        if show_sampling.get('zoom', False):
            self._plot_sampling_points_2d(ax, cmap)
            # Restore zoom limits after plotting (sampling points are from full domain)
            ax.set_xlim(x0_range[0], x0_range[1])
            ax.set_ylim(x1_range[0], x1_range[1])
        
        # Format title to show zoom range
        input_name = self._get_input_name(0)
        title = f'Zoom: {input_name}∈[{x0_range[0]:.3g},{x0_range[1]:.3g}]'
        ax.set_title(title)
        
        ax.set_xlabel(self._get_input_name(0))
        ax.set_ylabel(self._get_input_name(1))
    
    def _plot_error_1d(self, ax, output_idx, n_points=200):
        """Plot 1D absolute error on given axes."""
        x = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points).reshape(-1, 1)
        y = self.predict(x)
        
        y_true = self.problem.solution(x, self._build_params())
        # Handle list of outputs (for multi-output problems)
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
        """Plot 2D absolute error as contour on given axes."""
        x0 = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points)
        x1 = np.linspace(self.problem.xmin[1], self.problem.xmax[1], n_points)
        X0, X1 = np.meshgrid(x0, x1)
        
        x_flat = np.column_stack([X0.ravel(), X1.ravel()])
        y = self.predict(x_flat)
        Y = y[:, output_idx].reshape(X0.shape)
        
        y_true = self.problem.solution(x_flat, self._build_params())
        # Handle list of outputs (for multi-output problems)
        if isinstance(y_true, (list, tuple)):
            y_true = np.concatenate([np.atleast_2d(yt).T if yt.ndim == 1 else yt for yt in y_true], axis=1)
        elif y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        Y_true = y_true[:, output_idx].reshape(X0.shape)
        
        Y_error = np.abs(Y - Y_true)
        extent = [x0.min(), x0.max(), x1.min(), x1.max()]
        im = ax.imshow(Y_error, extent=extent, origin='lower', aspect='auto', cmap='managua')
        cbar = plt.colorbar(im, ax=ax, label='|Error|')
        self._colorbars.append(cbar)
        output_name = self._get_output_name(output_idx)
        ax.set_title(f'Absolute Error ({output_name})')
        
        ax.set_xlabel(self._get_input_name(0))
        ax.set_ylabel(self._get_input_name(1))
    
    def _plot_residuals_1d(self, ax, output_idx, n_points=200):
        """Plot 1D PDE residuals on given axes."""
        import torch
        
        x_np = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points).reshape(-1, 1)
        x_tensor = torch.tensor(x_np, dtype=torch.float32, device=self.device, requires_grad=True)
        
        self.network.eval()
        params = self._build_params()
        y_tensor = self.network(x_tensor, params)
        
        # Compute PDE residual
        residual = self.problem.pde_fn(x_tensor, y_tensor, self._build_params())
        
        # Handle list of residuals
        if isinstance(residual, (list, tuple)):
            if output_idx < len(residual):
                res = residual[output_idx].detach().cpu().numpy().flatten()
            else:
                res = np.zeros(n_points)
        else:
            res = residual.detach().cpu().numpy().flatten()
        
        ax.plot(x_np, np.abs(res), 'm-', linewidth=2)
        
        output_name = self._get_output_name(output_idx)
        input_name = self._get_input_name(0)
        ax.set_xlabel(input_name)
        ax.set_ylabel(f'|Residual| ({output_name})')
        ax.set_title(f'PDE Residual ({output_name})')
        ax.grid(True, alpha=0.3)
        
        # Show FBPINN subdomain predictions if enabled
        show_subdomains = getattr(self, '_show_subdomains', {})
        if show_subdomains.get('residuals', False) and hasattr(self.network, 'get_subdomain_predictions'):
            self._plot_fbpinn_subdomains_1d(ax, output_idx, n_points)
        
        # Show sampling points if enabled
        show_sampling = getattr(self, '_show_sampling_points', {})
        if show_sampling.get('residuals', False):
            cmap = self._get_colormap(output_idx)
            self._plot_sampling_points_1d(ax, cmap)
    
    def _plot_residuals_2d(self, ax, output_idx, n_points=50):
        """Plot 2D PDE residuals as contour on given axes."""
        import torch
        
        x0 = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points)
        x1 = np.linspace(self.problem.xmin[1], self.problem.xmax[1], n_points)
        X0, X1 = np.meshgrid(x0, x1)
        
        x_flat = np.column_stack([X0.ravel(), X1.ravel()])
        x_tensor = torch.tensor(x_flat, dtype=torch.float32, device=self.device, requires_grad=True)
        
        self.network.eval()
        params = self._build_params()
        y_tensor = self.network(x_tensor, params)
        
        # Compute PDE residual (use structured params)
        residual = self.problem.pde_fn(x_tensor, y_tensor, self._build_params())
        
        # Handle list of residuals
        if isinstance(residual, (list, tuple)):
            if output_idx < len(residual):
                res = residual[output_idx].detach().cpu().numpy().flatten()
            else:
                res = np.zeros(x_flat.shape[0])
        else:
            res = residual.detach().cpu().numpy().flatten()
        
        Res = np.abs(res).reshape(X0.shape)
        extent = [x0.min(), x0.max(), x1.min(), x1.max()]
        im = ax.imshow(Res, extent=extent, origin='lower', aspect='auto', cmap='viridis')
        cbar = plt.colorbar(im, ax=ax, label='|Residual|')
        self._colorbars.append(cbar)
        output_name = self._get_output_name(output_idx)
        ax.set_title(f'PDE Residual ({output_name})')
        
        # Show subdomain partitions if FBPINN and enabled
        show_subdomains = getattr(self, '_show_subdomains', {})
        if show_subdomains.get('residuals', False) and hasattr(self.network, 'domain'):
            self._plot_subdomain_boundaries_2d(ax)
        
        # Show sampling points if enabled
        show_sampling = getattr(self, '_show_sampling_points', {})
        if show_sampling.get('residuals', False):
            self._plot_sampling_points_2d(ax, 'viridis')
        
        ax.set_xlabel(self._get_input_name(0))
        ax.set_ylabel(self._get_input_name(1))
    
    def _plot_residuals_nd(self, ax, output_idx, region, n_points=50):
        """Plot PDE residuals on a 1D or 2D slice of an N-dimensional problem.
        
        Args:
            ax: Matplotlib axes
            output_idx: Which PDE equation residual to plot
            region: N-element tuple specifying the region (see _parse_region_nd)
            n_points: Number of points per free dimension
        """
        free_dims, free_ranges, fixed_dims, fixed_values = self._parse_region_nd(region)
        n_free = len(free_dims)
        
        if n_free == 0:
            ax.text(0.5, 0.5, 'No free dimensions',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Residual {output_idx}')
            return
        
        elif n_free == 1:
            # 1D residual plot
            dim = free_dims[0]
            x_range = free_ranges[0]
            x_vals = np.linspace(x_range[0], x_range[1], n_points)
            
            # Build full input array
            x_full = np.zeros((n_points, self.problem.n_dims))
            x_full[:, dim] = x_vals
            for i, val in zip(fixed_dims, fixed_values):
                x_full[:, i] = val
            
            # Use batched residual computation
            residual = self.compute_residuals(x_full)
            
            if isinstance(residual, list):
                if output_idx < len(residual):
                    res = residual[output_idx]
                else:
                    res = np.zeros(n_points)
            else:
                res = residual
            
            ax.plot(x_vals, np.abs(res), 'm-', linewidth=2)
            ax.set_xlabel(self._get_input_name(dim))
            ax.set_ylabel(f'|Residual|')
            
            # Build title showing fixed dimensions
            output_name = self._get_output_name(output_idx)
            if fixed_dims:
                fixed_str = ', '.join([f'{self._get_input_name(d)}={v:.3g}' 
                                       for d, v in zip(fixed_dims, fixed_values)])
                ax.set_title(f'Residual {output_name} at {fixed_str}')
            else:
                ax.set_title(f'Residual ({output_name})')
            ax.grid(True, alpha=0.3)
            
        elif n_free == 2:
            # 2D residual plot
            dim0, dim1 = free_dims[0], free_dims[1]
            x0_range, x1_range = free_ranges[0], free_ranges[1]
            
            x0 = np.linspace(x0_range[0], x0_range[1], n_points)
            x1 = np.linspace(x1_range[0], x1_range[1], n_points)
            X0, X1 = np.meshgrid(x0, x1)
            
            # Build full input array
            n_total = X0.size
            x_full = np.zeros((n_total, self.problem.n_dims))
            x_full[:, dim0] = X0.ravel()
            x_full[:, dim1] = X1.ravel()
            for i, val in zip(fixed_dims, fixed_values):
                x_full[:, i] = val
            
            # Use batched residual computation
            residual = self.compute_residuals(x_full)
            
            if isinstance(residual, list):
                if output_idx < len(residual):
                    res = residual[output_idx]
                else:
                    res = np.zeros(n_total)
            else:
                res = residual
            
            Res = np.abs(res).reshape(X0.shape)
            
            # Plot as heatmap
            extent = [x0.min(), x0.max(), x1.min(), x1.max()]
            im = ax.imshow(Res, extent=extent, origin='lower', aspect='auto', cmap='viridis')
            cbar = plt.colorbar(im, ax=ax, label='|Residual|')
            self._colorbars.append(cbar)
            
            ax.set_xlabel(self._get_input_name(dim0))
            ax.set_ylabel(self._get_input_name(dim1))
            
            # Build title showing fixed dimensions
            output_name = self._get_output_name(output_idx)
            if fixed_dims:
                fixed_str = ', '.join([f'{self._get_input_name(d)}={v:.3g}' 
                                       for d, v in zip(fixed_dims, fixed_values)])
                ax.set_title(f'Residual {output_name} at {fixed_str}')
            else:
                ax.set_title(f'Residual ({output_name})')
        else:
            ax.text(0.5, 0.5, f'Cannot plot {n_free}D (max 2D)',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Residual {output_idx}')
    
    def _get_plottable_bcs(self):
        """Get list of Dirichlet and Neumann BCs that can be plotted."""
        plottable_bcs = []
        for i, bc in enumerate(self.problem.boundary_conditions):
            if isinstance(bc, (DirichletBC, NeumannBC)):
                plottable_bcs.append((i, bc))
        return plottable_bcs
    
    def _plot_bc_2d(self, ax_val, ax_err, bc_idx, bc, n_points=100):
        """
        Plot boundary condition values and errors for a 2D problem.
        
        Creates line plots showing predicted vs target values along the boundary,
        and absolute error.
        """
        import torch
        
        # Determine boundary location and sampling dimension
        boundary = bc.boundary
        xmin, xmax = self.problem.xmin, self.problem.xmax
        
        # Find which dimension is fixed and which is free
        fixed_dim = None
        free_dim = None
        for i, side in enumerate(boundary):
            if side is not None:
                fixed_dim = i
                fixed_val = xmin[i] if side == 0 else xmax[i]
            else:
                free_dim = i
        
        if fixed_dim is None or free_dim is None:
            ax_val.text(0.5, 0.5, 'Cannot plot BC', ha='center', va='center', 
                       transform=ax_val.transAxes)
            return
        
        # Determine sampling range for free dimension
        if bc.subdomain is not None:
            sub_min, sub_max = bc.subdomain
        else:
            sub_min, sub_max = xmin[free_dim], xmax[free_dim]
        
        # Sample along the boundary
        free_coords = np.linspace(sub_min, sub_max, n_points)
        x_np = np.zeros((n_points, 2))
        x_np[:, fixed_dim] = fixed_val
        x_np[:, free_dim] = free_coords
        
        x_tensor = torch.tensor(x_np, dtype=torch.float32, device=self.device, requires_grad=True)
        
        self.network.eval()
        params = self._build_params()
        y_tensor = self.network(x_tensor, params)
        
        # Get target values
        target = bc.get_value(x_tensor)
        target_np = target.detach().cpu().numpy()
        
        if isinstance(bc, DirichletBC):
            # For Dirichlet: compare u values
            pred = y_tensor[:, bc.component].detach().cpu().numpy()
            bc_type = 'Dirichlet'
            ylabel = f'u_{bc.component}'
        elif isinstance(bc, NeumannBC):
            # For Neumann: compare normal derivatives
            normal_dim, normal_sign = bc.get_normal_direction()
            u = y_tensor[:, bc.component:bc.component + 1]
            grads = torch.autograd.grad(
                u, x_tensor, grad_outputs=torch.ones_like(u),
                create_graph=False, retain_graph=True
            )[0]
            du_dn = (normal_sign * grads[:, normal_dim]).detach().cpu().numpy()
            pred = du_dn
            bc_type = 'Neumann'
            ylabel = f'∂u_{bc.component}/∂n'
        else:
            return
        
        # Plot values
        dim_labels = ['x', 't'] if self.problem.n_dims == 2 else ['x₀', 'x₁']
        free_label = dim_labels[free_dim]
        fixed_label = dim_labels[fixed_dim]
        side_str = 'min' if boundary[fixed_dim] == 0 else 'max'
        
        # Get BC name
        # Get BC name
        bc_names = self._get_bc_names()
        bc_name = bc_names[bc_idx] if bc_idx < len(bc_names) else f'BC {bc_idx}'
        
        ax_val.plot(free_coords, target_np, 'b-', linewidth=2, label='Target')
        ax_val.plot(free_coords, pred, 'r--', linewidth=2, label='Predicted')
        ax_val.set_xlabel(free_label)
        ax_val.set_ylabel(ylabel)
        ax_val.set_title(f'{bc_type}: {bc_name}')
        ax_val.legend(loc='best', fontsize=8)
        ax_val.grid(True, alpha=0.3)
        
        # Plot error
        error = np.abs(pred - target_np)
        ax_err.plot(free_coords, error, 'm-', linewidth=2)
        ax_err.set_xlabel(free_label)
        ax_err.set_ylabel(f'|Error|')
        ax_err.set_title(f'{bc_name} Error (max={np.max(error):.4f})')
        ax_err.grid(True, alpha=0.3)

    def _create_figure(self):
        """Create figure and axes for plotting."""
        import math
        
        n_dims = self.problem.n_dims
        n_outputs = self.problem.n_outputs
        
        # Layout: 1 row for losses spanning all columns, 
        # then for each output: [true solution], predicted solution, residuals, [error], [zoom regions]
        # For 2D: one row per BC with 2 columns (value + error)
        # For N>=3D without plot_regions: only show losses
        has_solution = self.problem.solution is not None
        n_zoom_regions = len(getattr(self, '_plot_regions', []))
        plottable_bcs = self._get_plottable_bcs() if n_dims == 2 else []
        
        # Special case: N>=3D without plot_regions - only show loss curve
        if n_dims >= 3 and n_zoom_regions == 0:
            fig = plt.figure(figsize=(10, 4))
            axes = {}
            axes['losses'] = fig.add_subplot(1, 1, 1)
            self._colorbars = []
            return fig, axes
        
        # For N>=3D with slices: each region gets 2 columns (solution + residuals)
        if n_dims >= 3:
            n_sol_cols = 2 * n_zoom_regions  # solution + residuals for each region
        elif has_solution:
            if n_dims >= 2:
                n_sol_cols = 4 + n_zoom_regions  # true, predicted, residuals, error, + zoom regions
            else:
                n_sol_cols = 3 + n_zoom_regions  # For 1D: solution, residuals, error, + zoom regions
        else:
            n_sol_cols = 2 + n_zoom_regions  # solution, residuals, + zoom regions
        
        # For BC plots: each BC gets its own row with 2 columns (value + error)
        n_bc_cols = 2
        n_bc_rows = len(plottable_bcs)
        
        # Find LCM of solution columns and BC columns for even grid
        if n_bc_rows > 0:
            total_cols = (n_sol_cols * n_bc_cols) // math.gcd(n_sol_cols, n_bc_cols)  # LCM
            sol_span = total_cols // n_sol_cols  # How many grid cols per solution plot
            bc_span = total_cols // n_bc_cols    # How many grid cols per BC plot
        else:
            total_cols = n_sol_cols
            sol_span = 1
            bc_span = 1
        
        n_rows = 1 + n_outputs + n_bc_rows  # losses + outputs + BC rows
        
        fig = plt.figure(figsize=(5 * n_sol_cols, 3.5 * n_rows))
        gs = fig.add_gridspec(n_rows, total_cols, height_ratios=[1] * n_rows)
        
        axes = {}
        # Losses span all columns
        axes['losses'] = fig.add_subplot(gs[0, :])
        # Solution, residuals, and error for each output
        for i in range(n_outputs):
            col_idx = 0
            if n_dims >= 3:
                # For N>=3D: each region gets solution + residuals columns
                for r in range(n_zoom_regions):
                    axes[f'region_sol_{i}_{r}'] = fig.add_subplot(gs[1 + i, col_idx:col_idx + sol_span])
                    col_idx += sol_span
                    axes[f'region_res_{i}_{r}'] = fig.add_subplot(gs[1 + i, col_idx:col_idx + sol_span])
                    col_idx += sol_span
            else:
                # For 1D/2D: standard layout
                if has_solution and n_dims >= 2:
                    axes[f'true_{i}'] = fig.add_subplot(gs[1 + i, col_idx:col_idx + sol_span])
                    col_idx += sol_span
                axes[f'sol_{i}'] = fig.add_subplot(gs[1 + i, col_idx:col_idx + sol_span])
                col_idx += sol_span
                axes[f'res_{i}'] = fig.add_subplot(gs[1 + i, col_idx:col_idx + sol_span])
                col_idx += sol_span
                if has_solution:
                    axes[f'err_{i}'] = fig.add_subplot(gs[1 + i, col_idx:col_idx + sol_span])
                    col_idx += sol_span
                # Add zoom region axes
                for r in range(n_zoom_regions):
                    axes[f'zoom_{i}_{r}'] = fig.add_subplot(gs[1 + i, col_idx:col_idx + sol_span])
                    col_idx += sol_span
        
        # Add BC plot axes: one row per BC, each spanning bc_span columns
        for bc_row_idx, (i, bc) in enumerate(plottable_bcs):
            row = 1 + n_outputs + bc_row_idx
            axes[f'bc_val_{i}'] = fig.add_subplot(gs[row, 0:bc_span])
            axes[f'bc_err_{i}'] = fig.add_subplot(gs[row, bc_span:2*bc_span])
        
        # Store colorbar references for cleanup
        self._colorbars = []
        
        return fig, axes
    
    def _clear_colorbars(self):
        """Remove all stored colorbars."""
        if hasattr(self, '_colorbars'):
            for cbar in self._colorbars:
                try:
                    cbar.remove()
                except:
                    pass
        self._colorbars = []
    
    def _update_figure(self, fig, axes, n_points=200):
        """Update existing figure with current data."""
        n_dims = self.problem.n_dims
        n_outputs = self.problem.n_outputs
        has_solution = self.problem.solution is not None
        plot_regions = getattr(self, '_plot_regions', [])
        
        # Clear colorbars first
        self._clear_colorbars()
        
        # Clear all axes
        for key, ax in axes.items():
            if hasattr(ax, 'clear'):
                ax.clear()
        
        # Plot losses
        self._plot_losses(axes['losses'])
        
        # Plot solutions, residuals, and errors
        self.network.eval()
        
        # Special case: N>=3 without plot_regions - only losses are shown
        if n_dims >= 3 and not plot_regions:
            fig.tight_layout()
            return
        
        for i in range(n_outputs):
            if n_dims >= 3:
                # For N>=3D: each region has its own solution + residuals columns
                for r, region in enumerate(plot_regions):
                    sol_ax = axes.get(f'region_sol_{i}_{r}', None)
                    res_ax = axes.get(f'region_res_{i}_{r}', None)
                    if sol_ax is not None:
                        self._plot_region_nd(sol_ax, i, region, n_points=n_points)
                    if res_ax is not None:
                        self._plot_residuals_nd(res_ax, i, region, n_points=n_points)
            else:
                # For 1D/2D: standard layout
                sol_ax = axes.get(f'sol_{i}', None)
                res_ax = axes.get(f'res_{i}', None)
                err_ax = axes.get(f'err_{i}', None)
                true_ax = axes.get(f'true_{i}', None)
                
                # Skip if axes don't exist
                if sol_ax is None:
                    continue
                
                if n_dims == 1:
                    self._plot_solution_1d(sol_ax, i, n_points=n_points)
                    self._plot_residuals_1d(res_ax, i, n_points=n_points)
                    if err_ax is not None:
                        self._plot_error_1d(err_ax, i, n_points=n_points)
                    # Plot zoom regions
                    for r, region in enumerate(plot_regions):
                        zoom_ax = axes.get(f'zoom_{i}_{r}', None)
                        if zoom_ax is not None:
                            self._plot_solution_1d_region(zoom_ax, i, region, n_points=n_points)
                elif n_dims == 2:
                    if true_ax is not None:
                        self._plot_true_solution_2d(true_ax, i, n_points=n_points)
                    self._plot_solution_2d(sol_ax, i, n_points=n_points)
                    self._plot_residuals_2d(res_ax, i, n_points=n_points)
                    if err_ax is not None:
                        self._plot_error_2d(err_ax, i, n_points=n_points)
                    # Plot zoom regions
                    for r, region in enumerate(plot_regions):
                        zoom_ax = axes.get(f'zoom_{i}_{r}', None)
                        if zoom_ax is not None:
                            self._plot_solution_2d_region(zoom_ax, i, region, n_points=n_points)
        
        # Plot boundary conditions for 2D problems
        if n_dims == 2:
            plottable_bcs = self._get_plottable_bcs()
            for bc_idx, bc in plottable_bcs:
                ax_val = axes.get(f'bc_val_{bc_idx}', None)
                ax_err = axes.get(f'bc_err_{bc_idx}', None)
                if ax_val is not None and ax_err is not None:
                    self._plot_bc_2d(ax_val, ax_err, bc_idx, bc, n_points=n_points)
        
        fig.tight_layout()
    
    def plot_progress(self, save_path=None, n_points=200, fig=None, axes=None, display_handle=None):
        """
        Generate a figure with loss curves and solution plots.
        
        Args:
            save_path (str): Path to save the figure. If None, displays the figure.
            n_points (int): Number of points for solution plots.
            fig: Existing figure to update (for notebook live updates).
            axes: Existing axes to update.
            display_handle: IPython display handle for in-place updates.
            
        Returns:
            tuple: (fig, axes, display_handle) for reuse in live updates.
        """
        # Create new figure if not provided
        if fig is None or axes is None:
            fig, axes = self._create_figure()
        
        # Update figure content
        self._update_figure(fig, axes, n_points)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # Handle display based on environment
        if _is_notebook() and save_path is None:
            from IPython.display import display, update_display
            if display_handle is None:
                # First time: create display with ID
                display_handle = display(fig, display_id=True)
            else:
                # Update existing display in-place
                display_handle.update(fig)
        elif save_path is None:
            plt.show()
        
        return fig, axes, display_handle
    
    def train(self):
        """
        Train the network using parameters set by compile().
        
        Training history is accumulated across multiple train() calls,
        allowing for multi-stage training (e.g., Adam then L-BFGS).
        """
        if not self._compiled:
            raise RuntimeError("Must call compile() before train()")
        
        import time
        
        epochs = self._epochs
        print_each = self._print_each
        show_plots = self._show_plots
        save_plots = self._save_plots
        show_subdomains = self._show_subdomains
        profile = self._profile
        
        self._show_subdomains = show_subdomains
        self.network.train()
        
        # Timing accumulators
        timings = {
            'zero_grad': 0.0,
            'forward': 0.0,
            'pde_loss': 0.0,
            'bc_loss': 0.0,
            'backward': 0.0,
            'optimizer_step': 0.0,
            'history': 0.0,
            'test_loss': 0.0,
            'solution_error': 0.0,
            'plotting': 0.0,
        }
        
        # Reuse persistent figure if in notebook, or create new one
        if show_plots and _is_notebook():
            # Check if figure needs recreation (e.g., zoom regions added/removed)
            n_zoom_regions = len(getattr(self, '_plot_regions', []))
            needs_recreation = self._fig is None
            if not needs_recreation and self._axes is not None and n_zoom_regions > 0:
                # Check if zoom axes exist
                if f'zoom_0_0' not in self._axes:
                    needs_recreation = True
            if needs_recreation:
                self._fig, self._axes = self._create_figure()
        
        start_epoch = self._global_epoch
        training_start_time = time.time()
        
        # Determine auto-save filename for script mode (find next available pinn_progress_X.png)
        auto_save_path = None
        if show_plots and not save_plots and not _is_notebook():
            import glob
            import os
            existing = glob.glob('./pinn_progress_*.png')
            if existing:
                # Extract numbers from filenames like pinn_progress_0.png
                nums = []
                for f in existing:
                    base = os.path.basename(f)
                    # Handle both pinn_progress_X.png and pinn_progress_X_epochYYYYY.png
                    parts = base.replace('pinn_progress_', '').replace('.png', '').split('_')
                    try:
                        nums.append(int(parts[0]))
                    except ValueError:
                        pass
                next_num = max(nums) + 1 if nums else 0
            else:
                next_num = 0
            auto_save_path = f'./pinn_progress_{next_num}.png'
        
        # For LBFGS: sample points ONCE before the loop and keep them fixed
        if self.optimizer_name == "lbfgs":
            lbfgs_x_interior = self._sample_interior(self.train_samples[0]) if self.train_samples[0] > 0 else None
            lbfgs_x_bcs = []
            for i, bc in enumerate(self.problem.boundary_conditions):
                n_samples = self.train_samples[i + 1]
                if n_samples > 0:
                    lbfgs_x_bcs.append(self._sample_boundary(bc, n_samples))
                else:
                    lbfgs_x_bcs.append(None)
        
        print(f"Starting training for {epochs} epochs...")
        
        for local_epoch in range(epochs):
            epoch = start_epoch + local_epoch
            
            # Internal training state for curriculum learning
            internal = {'global_step': epoch, 'step': local_epoch}
            
            if self.optimizer_name == "lbfgs":
                # LBFGS uses FIXED points sampled once before the training loop
                x_interior = lbfgs_x_interior
                x_bcs = lbfgs_x_bcs
                
                def closure():
                    self.optimizer.zero_grad()
                    loss, _ = self._compute_total_loss(x_interior, x_bcs, internal)
                    loss.backward()
                    return loss
                
                self.optimizer.step(closure)
                loss, losses = self._compute_total_loss(x_interior, x_bcs, internal)
            else:
                # Adam, SGD - with optional mini-batching
                batch_size = self._batch_size
                
                t0 = time.perf_counter()
                self.optimizer.zero_grad()
                timings['zero_grad'] += time.perf_counter() - t0
                
                if batch_size is None:
                    # No mini-batching: process all samples at once (original behavior)
                    t0 = time.perf_counter()
                    loss, losses = self._compute_total_loss_timed(timings, internal) if profile else self._compute_total_loss(internal=internal)
                    timings['forward'] += time.perf_counter() - t0 if not profile else 0
                    
                    t0 = time.perf_counter()
                    loss.backward()
                    timings['backward'] += time.perf_counter() - t0
                else:
                    # Mini-batch gradient accumulation
                    # Sample all points once per epoch
                    t0 = time.perf_counter()
                    x_interior_full = self._sample_interior(self.train_samples[0]) if self.train_samples[0] > 0 else None
                    x_bcs_full = []
                    for i, bc in enumerate(self.problem.boundary_conditions):
                        n_samples = self.train_samples[i + 1]
                        if n_samples > 0:
                            x_bcs_full.append(self._sample_boundary(bc, n_samples))
                        else:
                            x_bcs_full.append(None)
                    timings['forward'] += time.perf_counter() - t0
                    
                    # Process PDE samples in batches
                    # Determine number of batches based on the LARGEST sample count
                    n_pde = self.train_samples[0]
                    max_samples = n_pde
                    for i, x_bc in enumerate(x_bcs_full):
                        if x_bc is not None:
                            max_samples = max(max_samples, x_bc.shape[0])
                    
                    n_batches = max(1, (max_samples + batch_size - 1) // batch_size)
                    
                    total_loss = 0.0
                    accumulated_losses = None
                    
                    for batch_idx in range(n_batches):
                        # Get batch of interior points (proportional to n_batches)
                        if x_interior_full is not None and n_pde > 0:
                            pde_batch_size = max(1, (n_pde + n_batches - 1) // n_batches)
                            start_idx = batch_idx * pde_batch_size
                            end_idx = min(start_idx + pde_batch_size, n_pde)
                            x_interior_batch = x_interior_full[start_idx:end_idx]
                        else:
                            x_interior_batch = None
                        
                        # For BCs, distribute samples across n_batches
                        x_bcs_batch = []
                        for i, x_bc in enumerate(x_bcs_full):
                            if x_bc is not None:
                                n_bc = x_bc.shape[0]
                                bc_batch_size = max(1, (n_bc + n_batches - 1) // n_batches)
                                bc_start = batch_idx * bc_batch_size
                                bc_end = min(bc_start + bc_batch_size, n_bc)
                                if bc_start < n_bc:
                                    x_bcs_batch.append(x_bc[bc_start:bc_end])
                                else:
                                    # This batch has no samples for this BC (already exhausted)
                                    x_bcs_batch.append(x_bc[0:1])  # Use 1 sample to avoid empty tensor
                            else:
                                x_bcs_batch.append(None)
                        
                        t0 = time.perf_counter()
                        batch_loss, batch_losses = self._compute_total_loss(x_interior_batch, x_bcs_batch, internal)
                        # Scale loss by 1/n_batches for correct gradient accumulation
                        batch_weight = 1.0 / n_batches
                        scaled_loss = batch_loss * batch_weight
                        timings['forward'] += time.perf_counter() - t0
                        
                        t0 = time.perf_counter()
                        scaled_loss.backward()
                        timings['backward'] += time.perf_counter() - t0
                        
                        # Accumulate for reporting
                        total_loss += batch_loss.item() * batch_weight
                        if accumulated_losses is None:
                            accumulated_losses = {
                                'pde': batch_losses['pde'].item() * batch_weight,
                                'pde_individual': [l.item() * batch_weight for l in batch_losses.get('pde_individual', [batch_losses['pde']])],
                                'bcs': [l.item() * batch_weight for l in batch_losses['bcs']]
                            }
                        else:
                            accumulated_losses['pde'] += batch_losses['pde'].item() * batch_weight
                            for j, l in enumerate(batch_losses.get('pde_individual', [batch_losses['pde']])):
                                accumulated_losses['pde_individual'][j] += l.item() * batch_weight
                            for j, l in enumerate(batch_losses['bcs']):
                                accumulated_losses['bcs'][j] += l.item() * batch_weight
                    
                    # Convert accumulated losses to tensor format for compatibility
                    loss = torch.tensor(total_loss, device=self.device)
                    losses = {
                        'pde': torch.tensor(accumulated_losses['pde'], device=self.device),
                        'pde_individual': [torch.tensor(l, device=self.device) for l in accumulated_losses['pde_individual']],
                        'bcs': [torch.tensor(l, device=self.device) for l in accumulated_losses['bcs']]
                    }
                
                t0 = time.perf_counter()
                self.optimizer.step()
                timings['optimizer_step'] += time.perf_counter() - t0
            
            # Record history
            t0 = time.perf_counter()
            self.history['epoch'].append(epoch)
            self.history['loss'].append(loss.item())
            # Store per-equation PDE losses
            pde_individual = losses.get('pde_individual', [losses['pde']])
            self.history['loss_pde'].append([l.item() for l in pde_individual])
            self.history['loss_bcs'].append([l.item() for l in losses['bcs']])
            timings['history'] += time.perf_counter() - t0
            
            # Print progress, test loss, solution error, and plots (only at print_each intervals)
            if print_each > 0 and (local_epoch % print_each == 0 or local_epoch == epochs - 1):
                # Test loss
                t0 = time.perf_counter()
                if any(s > 0 for s in self.test_samples):
                    test_loss, _ = self._compute_test_loss()
                    self.history['test_loss'].append(test_loss)
                timings['test_loss'] += time.perf_counter() - t0
                
                # Solution error (if true solution is available)
                t0 = time.perf_counter()
                if self.problem.solution is not None:
                    sol_error = self._compute_solution_error()
                    self.history['solution_error'].append(sol_error)
                timings['solution_error'] += time.perf_counter() - t0
                
                elapsed_time = time.time() - training_start_time
                
                # Build loss breakdown string
                bc_names = self._get_bc_names()
                pde_loss_val = losses['pde'].item() if hasattr(losses['pde'], 'item') else losses['pde']
                bc_losses_str = ", ".join(
                    f"{bc_names[i]}: {l.item() if hasattr(l, 'item') else l:.2e}" 
                    for i, l in enumerate(losses['bcs'])
                )
                
                msg = (
                    f"Epoch {epoch}/{self._epochs + start_epoch} | "
                    f"Loss: {loss.item():.6f} | "
                    f"PDE: {pde_loss_val:.2e} | "
                    f"BCs: [{bc_losses_str}] | "
                    f"Time: {elapsed_time:.1f}s"
                )
                if self.problem.solution is not None:
                    msg += f" | Error: {self.history['solution_error'][-1]:.2e}"
                print(msg)
                
                # Generate plots
                t0 = time.perf_counter()
                if show_plots or save_plots:
                    if save_plots:
                        # User-specified path: create epoch-suffixed files for history
                        plot_path = f"{save_plots}_epoch{epoch:05d}.png"
                    elif auto_save_path:
                        # Script mode with show_plots: auto-save to single updating file
                        plot_path = auto_save_path
                    else:
                        # Notebook mode: display inline (no save)
                        plot_path = None
                    _, _, self._display_handle = self.plot_progress(
                        save_path=plot_path, n_points=self._plot_n_points,
                        fig=self._fig, axes=self._axes, 
                        display_handle=self._display_handle
                    )
                timings['plotting'] += time.perf_counter() - t0
        
        # Update global epoch counter
        self._global_epoch += epochs
        
        total_training_time = time.time() - training_start_time
        print(f"Training complete in {total_training_time:.1f}s")
        
        # Print timing breakdown
        if profile:
            total_time = sum(timings.values())
            print(f"\n--- Timing Breakdown ({epochs} epochs, {total_time:.2f}s total) ---")
            for key, val in sorted(timings.items(), key=lambda x: -x[1]):
                pct = 100 * val / total_time if total_time > 0 else 0
                per_epoch = 1000 * val / epochs  # ms per epoch
                print(f"  {key:20s}: {val:8.3f}s ({pct:5.1f}%) - {per_epoch:.3f} ms/epoch")
    
    def _compute_total_loss_timed(self, timings, internal=None):
        """Compute total weighted loss with timing breakdown.
        
        Args:
            timings: Dict to accumulate timing information
            internal: Internal training state dict with 'global_step' and 'step'
        """
        import time
        losses = {}
        
        # PDE loss
        if self.train_samples[0] > 0:
            t0 = time.perf_counter()
            x_interior = self._sample_interior(self.train_samples[0])
            timings['pde_loss'] += time.perf_counter() - t0
            
            t0 = time.perf_counter()
            pde_loss, pde_individual = self._compute_pde_loss(x_interior, internal)
            # Apply weight only if it's a scalar (list weights are applied inside _compute_pde_loss)
            if not isinstance(self.weights[0], (list, tuple)):
                pde_loss = self.weights[0] * pde_loss
            losses['pde'] = pde_loss
            losses['pde_individual'] = pde_individual
            timings['pde_loss'] += time.perf_counter() - t0
        else:
            losses['pde'] = torch.tensor(0.0, device=self.device)
            losses['pde_individual'] = [torch.tensor(0.0, device=self.device)]
        
        # BC losses
        losses['bcs'] = []
        for i, bc in enumerate(self.problem.boundary_conditions):
            n_samples = self.train_samples[i + 1]
            if n_samples > 0:
                t0 = time.perf_counter()
                x_bc = self._sample_boundary(bc, n_samples)
                bc_loss = self.weights[i + 1] * self._compute_bc_loss(bc, x_bc)
                losses['bcs'].append(bc_loss)
                timings['bc_loss'] += time.perf_counter() - t0
            else:
                losses['bcs'].append(torch.tensor(0.0, device=self.device))
        
        # Total loss
        total = losses['pde'] + sum(losses['bcs'])
        
        return total, losses
    
    def predict(self, x: np.ndarray, batch_size: int = None) -> np.ndarray:
        """
        Make predictions with the trained network.
        
        Args:
            x: Input points of shape (n_points, n_dims).
            batch_size: If provided, process in batches to reduce memory usage.
            
        Returns:
            Predictions of shape (n_points, n_outputs).
        """
        self.network.eval()
        
        if batch_size is None:
            batch_size = getattr(self, '_batch_size', None)
        
        with torch.no_grad():
            if batch_size is None or batch_size >= len(x):
                # Process all at once
                x_tensor = torch.tensor(x, device=self.device, dtype=self.dtype)
                params = self._build_params()
                y = self.network(x_tensor, params)
                result = y.cpu().numpy()
                del x_tensor, y
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                return result
            else:
                # Process in batches
                n_points = len(x)
                results = []
                params = self._build_params()
                for start in range(0, n_points, batch_size):
                    end = min(start + batch_size, n_points)
                    x_batch = torch.tensor(x[start:end], device=self.device, dtype=self.dtype)
                    y_batch = self.network(x_batch, params)
                    results.append(y_batch.cpu().numpy())
                    del x_batch, y_batch
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                return np.vstack(results)
    
    def compute_residuals(self, x: np.ndarray, batch_size: int = None) -> np.ndarray:
        """
        Compute PDE residuals at given points.
        
        Args:
            x: Input points of shape (n_points, n_dims).
            batch_size: If provided, process in batches to reduce memory usage.
            
        Returns:
            Residuals as numpy array. Shape depends on PDE (single or list of residuals).
        """
        import torch
        
        self.network.eval()
        
        if batch_size is None:
            batch_size = getattr(self, '_batch_size', None)
        
        n_points = len(x)
        params = self._build_params()
        
        if batch_size is None or batch_size >= n_points:
            # Process all at once
            x_tensor = torch.tensor(x, device=self.device, dtype=self.dtype, requires_grad=True)
            y_tensor = self.network(x_tensor, params)
            residual = self.problem.pde_fn(x_tensor, y_tensor, params)
            
            if isinstance(residual, (list, tuple)):
                result = [r.detach().cpu().numpy().flatten() for r in residual]
            else:
                result = residual.detach().cpu().numpy().flatten()
            
            # Clean up
            del x_tensor, y_tensor, residual
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            return result
        else:
            # Process in batches
            all_residuals = None
            for start in range(0, n_points, batch_size):
                end = min(start + batch_size, n_points)
                x_batch = torch.tensor(x[start:end], device=self.device, dtype=self.dtype, requires_grad=True)
                y_batch = self.network(x_batch, params)
                residual = self.problem.pde_fn(x_batch, y_batch, params)
                
                if isinstance(residual, (list, tuple)):
                    batch_res = [r.detach().cpu().numpy().flatten() for r in residual]
                    if all_residuals is None:
                        all_residuals = [[] for _ in batch_res]
                    for i, r in enumerate(batch_res):
                        all_residuals[i].append(r)
                else:
                    batch_res = residual.detach().cpu().numpy().flatten()
                    if all_residuals is None:
                        all_residuals = []
                    all_residuals.append(batch_res)
                
                # Clean up each batch to free GPU memory
                del x_batch, y_batch, residual
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
            
            if isinstance(all_residuals[0], list):
                # List of residuals
                return [np.concatenate(r) for r in all_residuals]
            else:
                return np.concatenate(all_residuals)
    
    def get_history(self) -> Dict:
        """Get training history."""
        return self.history
