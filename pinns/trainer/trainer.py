"""
PINN Trainer (JAX/Optax).

Provides BaseTrainer (common plotting/history utilities) and the concrete
JAX Trainer.  LR schedulers live in pinns.schedulers.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from abc import ABC, abstractmethod
from functools import partial
import time
import inspect
import threading

try:
    import jaxopt
    HAS_JAXOPT = True
except ImportError:
    HAS_JAXOPT = False

try:
    from soap_jax import soap as soap_optimizer
    HAS_SOAP = True
except ImportError:
    HAS_SOAP = False

from .schedulers import LRScheduler, ExponentialDecay, ReduceLROnPlateau, is_notebook
from ..functional import set_context, clear_context, make_derivative_fn

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
        
        self.network = network
        self.device = device
        self.dtype = None
        
        # Set normalization bounds on the network from the problem
        self._setup_network_normalization()
        
        # Training configuration defaults
        from pinns.problems.problem_strong import ProblemStrong as _ProblemStrong
        from pinns.problems.problem_weak import ProblemWeak as _ProblemWeak
        if isinstance(problem, _ProblemStrong):
            # ProblemStrong: derive defaults from registered terms
            term_names = [t.name for t in problem._terms]
            n_terms = len(term_names)
            _pde_samples = 0  # compile() will set proper values via dict
            self.train_samples = {t.name: (1000 if t.kind == 'inner' else 10)
                                  for t in problem._terms}
            self.test_samples  = {t.name: 0 for t in problem._terms}
            self.weights       = {t.name: 1.0 for t in problem._terms}
        else:
            from pinns.terms import TermPeriodicBC as _PBC2
            _periodic_types2 = (_PBC2,)
            n_bcs = sum(1 for bc in problem.boundary_conditions
                        if not isinstance(bc, _periodic_types2))
            expected_len = 1 + n_bcs
            _pde_samples = 0 if isinstance(problem, _ProblemWeak) else 100
            self.train_samples = [_pde_samples] + [10] * n_bcs
            self.test_samples = [0] + [0] * n_bcs
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
        self._plot_callback = None
        self._save_plots = None
        self._show_subdomains = {'solution': False, 'residuals': False, 'zoom': False}
        self._show_sampling_points = {'solution': False, 'residuals': False, 'zoom': False}
        self._plot_regions = []
        self._plot_n_points = 200
        self._batch_size = None
        self._compiled = False
        self._plot_kwargs = {}
        self._plot_style = {}
    
    def _setup_network_normalization(self):
        """Set up input/output normalization on the network from problem definition."""
        xmin = np.array(self.problem.xmin)
        xmax = np.array(self.problem.xmax)
        
        # Set input range from domain bounds (only if normalization is enabled)
        if hasattr(self.network, 'set_input_range'):
            if getattr(self.network, 'normalize_input', True):
                self.network.set_input_range(xmin, xmax)
        
        # Set output range on the network — output_range is owned by the network,
        # not the problem.  Only call set_output_range if the network has not
        # already been configured (i.e. output_range_min is None / absent).
        if hasattr(self.network, 'set_output_range'):
            if getattr(self.network, 'unnormalize_output', True):
                already_set = (
                    getattr(self.network, 'output_range_min', None) is not None
                    or getattr(self.network, 'output_min', None) is not None
                )
                if not already_set:
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
        """Sample initial training and test data, then notify schedulers."""
        self._sample_train_data()

        _ts = self.test_samples
        if any(v > 0 for v in (_ts.values() if isinstance(_ts, dict) else _ts)):
            self._sample_test_data()

        for s in getattr(self, '_schedulers', []):
            s.on_compile(self)
    
    def _sample_pool_data(self, pool_multiplier: int):
        """Sample a large pool for efficient resampling during training."""
        self._train_pool = {}
        samples_dict = self._list_to_dict_samples(self.train_samples)
        for name, n in samples_dict.items():
            if n > 0:
                pool_n = n * pool_multiplier
                np_data = self._sample_points_np(name, pool_n)
                self._train_pool[name] = self._to_tensor(np_data)
    
    def _select_from_pool(self, rng=None):
        """Select training data from pre-sampled pool (fast random indexing).
        
        Also updates target values for BCs with callable value functions.
        """
        if not hasattr(self, '_train_pool') or self._train_pool is None:
            # Fallback to full resampling
            self._sample_train_data()
            return
        
        if rng is None:
            rng = self.rng
        
        self._train_data = {}
        self._train_targets = {}  # Reset targets for new selection
        samples_dict = self._list_to_dict_samples(self.train_samples)
        for name, n in samples_dict.items():
            if n > 0 and name in self._train_pool:
                pool = self._train_pool[name]
                pool_size = len(pool)
                # Random indices from pool
                indices = rng.choice(pool_size, size=n, replace=False)
                self._train_data[name] = self._index_tensor(pool, indices)
                
                # Compute target values for BCs with callable value functions
                if name != 'pde':
                    bc = self._get_bc_by_name(name)
                    if bc is not None and hasattr(bc, 'value') and callable(bc.value):
                        np_data = self._to_numpy(self._train_data[name])
                        target_np = bc.value(np_data)
                        if hasattr(target_np, 'squeeze'):
                            target_np = target_np.squeeze(-1) if target_np.ndim > 1 else target_np
                        self._train_targets[name] = self._to_tensor(target_np)
    
    @abstractmethod
    def _index_tensor(self, tensor, indices):
        """Index a tensor with numpy indices. Backend-specific."""
        pass
    
    def _adaptive_resample(self, params_dict: Dict = None):
        """
        Perform adaptive resampling based on PDE residuals.
        
        Depending on adaptive_mode:
        - "replace": Replace low-residual points with new points near high-residual locations.
        - "add": Add new points near high-residual locations (growing sample count).
        - "rar": Residual-based Adaptive Resampling via importance sampling.
        
        Args:
            params_dict: Parameters dict from _build_params(). If None, will be built.
        """
        if 'pde' not in self._train_data:
            return
        
        if params_dict is None:
            params_dict = self._build_params()
        
        # Dispatch to RAR method if using importance sampling mode
        if self._adaptive_mode == "rar":
            self._adaptive_rar_resample(params_dict)
            return
        
        # Get current PDE training points as numpy
        x_pde = self._to_numpy(self._train_data['pde'])
        n_points = len(x_pde)
        
        # For "add" mode, check if we've reached max samples
        if self._adaptive_mode == "add" and self._adaptive_max_samples is not None:
            if n_points >= self._adaptive_max_samples:
                return  # Already at max, skip
        
        n_new = int(n_points * self._adaptive_ratio)
        
        # For "add" mode, cap n_new to not exceed max_samples
        if self._adaptive_mode == "add" and self._adaptive_max_samples is not None:
            n_new = min(n_new, self._adaptive_max_samples - n_points)
        
        if n_new < 1:
            return
        
        # Compute residuals at current points
        residuals = self._compute_residuals(x_pde, batch_size=self._batch_size)
        
        # Combine residuals from all equations (L2 norm)
        total_residual = np.zeros(n_points)
        for res in residuals:
            total_residual += res.flatten() ** 2
        total_residual = np.sqrt(total_residual)
        
        # Get indices of highest-residual points
        high_res_indices = np.argsort(total_residual)[-n_new:]
        high_res_points = x_pde[high_res_indices]
        
        # Compute sampling scale from domain size
        domain = self.problem.domain
        domain_scale = np.array([domain.xmax[d] - domain.xmin[d] for d in range(len(domain.xmin))])
        std = self._adaptive_std * domain_scale
        
        # Sample new points near high-residual locations
        new_points = high_res_points + self.rng.normal(0, std, size=high_res_points.shape)
        
        # Clip to domain bounds
        for d in range(new_points.shape[1]):
            new_points[:, d] = np.clip(new_points[:, d], domain.xmin[d], domain.xmax[d])
        
        if self._adaptive_mode == "add":
            # Add new points to existing dataset
            x_pde = np.concatenate([x_pde, new_points], axis=0)
        else:
            # Replace lowest-residual points with the new points
            low_res_indices = np.argsort(total_residual)[:n_new]
            x_pde[low_res_indices] = new_points
        
        # Update training data
        self._train_data['pde'] = self._to_tensor(x_pde)
    
    def _adaptive_rar_resample(self, params_dict: Dict = None):
        """
        Perform Residual-based Adaptive Resampling (RAR) via importance sampling.
        
        Algorithm:
        1. Sample factor * n_samples from the domain
        2. Compute residuals for all samples
        3. Compute weights: p = (r^k / mean(r^k)) + c
        4. Normalize: p = p / sum(p)
        5. Sample n_samples using these weights (importance sampling)
        
        Args:
            params_dict: Parameters dict from _build_params(). If None, will be built.
        """
        if params_dict is None:
            params_dict = self._build_params()
        
        # Get target number of PDE samples
        n_target = self.train_samples[0]  # PDE samples is first element
        factor = self._adaptive_factor
        k = self._adaptive_k
        c = self._adaptive_c
        
        n_candidates = factor * n_target
        x_candidates = self.problem.domain.sample_interior(
            n_candidates, rng=self.rng, params=self._build_params())
        
        # Compute residuals at candidate points
        residuals = self._compute_residuals(x_candidates, batch_size=self._batch_size)
        
        # Combine residuals from all equations (L2 norm)
        total_residual = np.zeros(n_candidates)
        for res in residuals:
            total_residual += res.flatten() ** 2
        total_residual = np.sqrt(total_residual)
        
        # Compute importance weights: p = (r^k / mean(r^k)) + c
        r_pow_k = np.power(np.abs(total_residual) + 1e-10, k)
        weights = (r_pow_k / (np.mean(r_pow_k) + 1e-10)) + c
        
        # Normalize to get probabilities
        weights = weights / np.sum(weights)
        
        # Sample n_target indices using importance weights (without replacement)
        selected_indices = self.rng.choice(
            n_candidates, 
            size=n_target, 
            replace=False, 
            p=weights
        )
        
        # Get selected points
        x_pde = x_candidates[selected_indices]
        
        # Update training data
        self._train_data['pde'] = self._to_tensor(x_pde)
    
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
        # Learning rate scheduler
        lr_scheduler: Optional[LRScheduler] = None,
        # Training schedulers (resample, adaptive, curriculum, lagrange, …)
        schedulers: Optional[List] = None,
        # L-BFGS specific parameters
        lbfgs_max_iter: int = 5,
        lbfgs_history_size: int = 50,
        lbfgs_tolerance: float = 1e-9,
        lbfgs_line_search: str = "strong_wolfe",
        # SOAP specific parameters
        soap_params: Optional[Dict[str, Any]] = None,
        # Plot keyword arguments
        plot_kwargs: Optional[Dict[str, Any]] = None,
        # Plot style
        plot_style: Optional[Dict[str, Any]] = None,
        # Periodic callback: called as plot_callback(epoch, trainer) every print_each steps
        plot_callback: Optional[callable] = None,
        # Time-step curriculum for BPTT rollout mode
        # Train epochs_by_time_step epochs with 1 step, then 2 steps, …, up to n_time_steps.
        # Each stage runs epochs_by_time_step epochs.  Set to None to disable.
        epochs_by_time_step: Optional[int] = None,
        # Snapshot time points for transient mesh solutions.
        # When set, plot_progress shows one row of panels (predicted, true?, residual, error?)
        # per time value.  E.g. plot_time_points=[0.0, 0.25, 0.5, 1.0].
        plot_time_points: Optional[List[float]] = None,
        # Gradient clipping: clip global gradient norm to this value. Useful for BPTT.
        # Set to None (default) to disable.
        grad_clip: Optional[float] = None,
        # Mini-batch over test functions (faces) for BPTT rollout.
        # Each training step uses a random subset of face_batch_size elements instead
        # of all F faces.  Reduces per-step cost; acts as stochastic regularisation.
        # Set to None (default) to use all faces (full Galerkin).
        face_batch_size: Optional[int] = None,
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
            resample_each: Resample collocation points every N epochs (0 = never).
            resample_pool_size: Pool multiplier for efficient resampling (default 10 = pool is 10x training samples).
            pool_refresh_each: Refresh entire pool with new samples every N epochs (0 = never). Use when pool_size * training_samples < epochs.
            adaptive_sampling: Enable adaptive sampling based on residuals (default: False).
            adaptive_each: Perform adaptive resampling every N epochs (default: 100).
            adaptive_ratio: Fraction of points to add/replace with high-residual neighbors (default: 0.5).
            adaptive_std: Standard deviation for sampling near high-residual points, as fraction of domain size (default: 0.1).
            adaptive_mode: "replace" (replace low-residual points), "add" (grow sample count), or "rar" (importance sampling). Default: "replace".
            adaptive_max_samples: Maximum PDE samples when mode="add" (None = no limit). Prevents OOM.
            adaptive_k: Power for residual weighting in RAR mode: p = (r^k / mean(r^k)) + c. Default: 1.0.
            adaptive_c: Offset for uniform sampling in RAR mode. Higher c = more uniform. Default: 1.0.
            adaptive_factor: Oversampling factor for RAR mode. Sample factor*n points, then select n. Default: 2.
            lr_scheduler: Learning rate scheduler (e.g., ExponentialDecay). If None, constant learning rate.
            epochs_by_time_step: Time-step curriculum for BPTT rollout mode.  When set to an
                integer N, ``train()`` runs N epochs with 1 rollout step, then N epochs
                with 2 rollout steps, …, up to ``problem.n_time_steps`` rollout steps.
                Each stage re-JITs the scan for the new length.  Set to ``None`` (default)
                to train on the full rollout from the start.
            curriculum_t_ends: Progressive domain schedule — list of end values for the chosen
                input dimension (e.g. [2.0, 5.0, 10.0]). Training starts sampling up to
                curriculum_t_ends[0], advances to curriculum_t_ends[1] after
                curriculum_t_epochs epochs, and so on. Use this to implement time-causal
                curriculum learning. If None, disabled.
            curriculum_t_epochs: Number of epochs per curriculum stage (default: 1000).
            curriculum_t_dim: Which input dimension is the "time" axis (default: 0).
            lbfgs_max_iter: Max iterations per L-BFGS step (default: 5).
            lbfgs_history_size: History size for L-BFGS (default: 50).
            lbfgs_tolerance: Tolerance for gradient convergence (default: 1e-9).
            lbfgs_line_search: Line search method - 'strong_wolfe' or None (default: 'strong_wolfe').
            soap_params: SOAP optimizer parameters (JAX only). Dict with keys:
                - b1: Adam's beta1 (default: 0.95)
                - b2: Adam's beta2 (default: 0.95)
                - shampoo_beta: Beta for preconditioner (-1 uses b2)
                - eps: Numerical stability (default: 1e-8)
                - weight_decay: Weight decay (default: 0.0)
                - precondition_frequency: How often to update preconditioner (default: 10)
                - max_precond_dim: Max preconditioner dimension (default: 10000)
                - precondition_1d: Whether to precondition 1D params (default: False)
            plot_kwargs: Optional dict of kwargs to customise individual plot panels.
                Keys select the panel; values are dicts passed to ``ax.set()``.
                Supported keys:
                  - ``"losses"``      – weighted-loss panel
                  - ``"mse_losses"``  – MSE-loss panel
                  - ``"solution"``    – all solution panels
                  - ``"residuals"``   – all residual panels
                  - ``"error"``       – all error panels
                Common matplotlib ``set()`` kwargs: ``yscale``, ``xscale``,
                ``ylim``, ``xlim``, ``title``, ``xlabel``, ``ylabel``.
                Example::

                    plot_kwargs={
                        "losses":   {"yscale": "log"},
                        "solution": {"ylim": (-1.5, 1.5)},
                    }
            plot_style: Optional dict to control the overall figure appearance.
                Keys:
                  - ``"theme"``      – ``"dark"`` or ``"light"`` (default). Applies
                    matplotlib\'s ``dark_background`` / default style.
                  - ``"bg_color"``   – axes background color (any matplotlib color,
                    e.g. ``"#1e1e2e"``, ``"white"``, ``"#0d0d0d"``).
                  - ``"fig_color"``  – figure (outer) background color.
                  - ``"text_color"`` – color for labels, titles and tick labels.
                  - ``"grid_color"`` – gridline color (default: auto from theme).
                Example::

                    plot_style={
                        "theme": "dark",
                        "bg_color": "#1e1e2e",
                        "fig_color": "#13131f",
                        "text_color": "white",
                    }
        """
        # Store L-BFGS parameters
        self._lbfgs_max_iter = lbfgs_max_iter
        self._lbfgs_history_size = lbfgs_history_size
        self._lbfgs_tolerance = lbfgs_tolerance
        self._lbfgs_line_search = lbfgs_line_search
        
        # Store SOAP parameters
        old_soap_params = getattr(self, '_soap_params', None)
        self._soap_params = soap_params if soap_params is not None else {}
        
        from pinns.problems.problem_strong import ProblemStrong as _PScompile
        from pinns.problems.problem_weak import ProblemWeak as _PWcompile
        if isinstance(self.problem, (_PScompile, _PWcompile)):
            # ProblemStrong / ProblemWeak: train_samples/weights/test_samples are plain dicts; no list conversion
            if train_samples is not None:
                self.train_samples = dict(train_samples) if isinstance(train_samples, dict) else train_samples
            if test_samples is not None:
                self.test_samples = dict(test_samples) if isinstance(test_samples, dict) else test_samples
            if weights is not None:
                self.weights = dict(weights) if isinstance(weights, dict) else weights
        else:
            n_bcs = len(self._get_bc_names())
            expected_len = 1 + n_bcs
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
                self._raw_weights_dict = weights if isinstance(weights, dict) else None
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
        
        # Set lr_scheduler BEFORE creating optimizer (affects inject_hyperparams)
        old_scheduler = getattr(self, '_lr_scheduler', None)
        self._lr_scheduler = lr_scheduler
        if (lr_scheduler is not None) != (old_scheduler is not None):
            optimizer_changed = True  # Scheduler presence changed, rebuild optimizer
        
        # Check if SOAP params changed
        if soap_params is not None and soap_params != old_soap_params:
            optimizer_changed = True

        # Check if grad_clip changed
        old_grad_clip = getattr(self, '_grad_clip', None)
        self._grad_clip = grad_clip
        if grad_clip != old_grad_clip:
            optimizer_changed = True

        old_face_batch = getattr(self, '_rollout_face_batch', None)
        self._rollout_face_batch = face_batch_size
        if face_batch_size != old_face_batch:
            optimizer_changed = True   # force recompile of loss fn
        
        if self.optimizer is None or optimizer_changed:
            self.optimizer = self._create_optimizer()
            self._init_optimizer_state()
        
        self._epochs = epochs
        self._print_each = print_each
        self._show_plots = show_plots
        self._save_plots = save_plots
        self._plot_callback = plot_callback
        self._epochs_by_time_step = epochs_by_time_step
        self._plot_kwargs = plot_kwargs if plot_kwargs is not None else {}
        self._plot_style = plot_style if plot_style is not None else {}
        
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
        self._plot_time_points = list(plot_time_points) if plot_time_points is not None else None
        self._plot_n_points = plot_n_points
        self._batch_size = batch_size
        self._schedulers = list(schedulers) if schedulers else []

        # Force creation of a new figure on next train() call
        # This ensures a fresh plot in the new cell while keeping history
        self._fig = None
        self._axes = None
        self._display_handle = None
        self._colorbars = []
        
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
        Predict output for given input points using the JAX network.apply() API.
        
        Args:
            x: Input points as numpy array of shape (n_points, n_inputs)
            batch_size: Optional batch size for large inputs
            
        Returns:
            Predictions as numpy array of shape (n_points, n_outputs)
        """
        if batch_size is None:
            batch_size = getattr(self, '_batch_size', None)
        
        params_dict = self._build_params()
        params = self.network.params

        def _infer_batch(x_batch):
            out = self.network.apply(params, jnp.asarray(x_batch, dtype=jnp.float32), params_dict)
            return np.array(out)

        if batch_size is None or batch_size >= len(x):
            return _infer_batch(x)

        results = []
        for start in range(0, len(x), batch_size):
            results.append(_infer_batch(x[start:start + batch_size]))
        return np.vstack(results)
    
    @abstractmethod
    def _to_tensor(self, np_array: np.ndarray):
        """Convert numpy array to a JAX array (jnp.array)."""
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
            Target tensor (backend-specific), 1D shape (n_points,)
        """
        from pinns.terms import TermPoints, TermCustomBC
        
        if isinstance(bc, (TermCustomBC, TermPoints)):
            # These BCs have no scalar value attribute; loss computed elsewhere
            return None
        
        if callable(bc.value):
            x_np = np.asarray(x)  # Convert to numpy for user function
            result = np.asarray(bc.value(x_np))
            # Squeeze to 1D if result is 2D (e.g., shape (n, 1))
            if result.ndim > 1:
                result = result.squeeze(-1)
            return self._to_tensor(result)
        else:
            return self._to_tensor(np.full((x.shape[0],), bc.value))
    
    # ==================== Network call helper ====================

    def _call_network(self, x, params_dict):
        """Call the JAX network via network.apply(params, x, params_dict)."""
        return self.network.apply(self.network.params, x, params_dict)

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
        y = self._call_network(x, params_dict)
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
        """Compute BC loss by delegating to ``bc.compute_loss_dict``.

        Returns the sum of all sub-losses (unweighted — the caller weights).
        """
        y = self._call_network(x, params_dict)
        ops = self._make_bc_ops(params_dict)
        return sum(bc.compute_loss_dict(x, y, ops).values())

    def _compute_custom_bc_losses_dict(self, bc, x, y, params_dict, weights_dict=None):
        """Return a {output_name: weighted_scalar_loss} dict for a TermMeshCustomBC.

        Each output in ``bc.output_names`` gets its own entry so the plotter can
        show per-component losses.
        """
        # Weak BCs (phi in signature) are handled by ProblemWeak's Galerkin
        # assembler — skip pointwise evaluation.
        if getattr(bc, 'is_weak', False):
            out_names = bc.output_names or [bc.name]
            return {oname: 0.0 for oname in out_names}
        import inspect as _inspect
        sig = _inspect.signature(bc.f)
        n_params = len(sig.parameters)
        if n_params >= 4:
            residual = bc.f(x, y, params_dict, None)
        elif n_params == 3:
            residual = bc.f(x, y, params_dict)
        else:
            residual = bc.f(x, y)
        if not isinstance(residual, (list, tuple)):
            residual = (residual,)
        out_names = (bc.output_names or ([bc.name] * len(residual)))
        default_w = (weights_dict or {}).get(bc.name, 1.0) if weights_dict else 1.0
        return {
            oname: (weights_dict or {}).get(oname, default_w) * self._mean_squared(r)
            for r, oname in zip(residual, out_names)
        }

    def _compute_custom_bc_loss(self, bc, x, y, params_dict, weights_dict=None):
        """Evaluate a TermMeshCustomBC residual and return the MSE loss.

        Override in backend subclasses to supply real autodiff derivatives.
        The base implementation calls ``f`` with ``derivative=None``.
        When ``weights_dict`` is provided and ``bc.output_names`` is set,
        each output is weighted independently.
        """
        # Weak BCs (phi in signature) are handled by ProblemWeak's Galerkin
        # assembler — skip pointwise evaluation.
        if getattr(bc, 'is_weak', False):
            return 0.0
        import inspect as _inspect
        sig = _inspect.signature(bc.f)
        n_params = len(sig.parameters)
        if n_params >= 4:
            residual = bc.f(x, y, params_dict, None)
        elif n_params == 3:
            residual = bc.f(x, y, params_dict)
        else:
            residual = bc.f(x, y)
        if isinstance(residual, (list, tuple)):
            out_names = (bc.output_names or ([bc.name] * len(residual)))
            default_w = (weights_dict or {}).get(bc.name, 1.0) if weights_dict else 1.0
            losses = [
                (weights_dict or {}).get(oname, default_w) * self._mean_squared(r)
                for r, oname in zip(residual, out_names)
            ]
            return sum(losses)
        default_w = (weights_dict or {}).get(bc.name, 1.0)
        return default_w * self._mean_squared(residual)

    def _compute_pointset_bc_losses_dict(self, bc, x, y, weights_dict=None):
        """Return a ``{sub_name: weighted_scalar_loss}`` dict for a :class:`TermPoints`.

        One sub-loss is produced per column of ``bc.outputs`` / element of
        ``bc.components``.  Sub-loss names come from ``bc.get_input_names()``
        (auto-generated as ``name_0``, ``name_1``, … or via ``output_names``).

        Args:
            bc: A :class:`~pinns.boundary.TermPoints` instance.
            x:  Input tensor (already the bc points, passed for device/dtype ref).
            y:  Network output tensor ``(N, n_outputs)``.
            weights_dict: Optional weight dict; keys may be the base ``bc.name``
                          (fallback) or individual sub-loss names.

        Returns:
            ``{sub_name: weighted_scalar_loss}``
        """
        from pinns.terms import TermPoints
        device = y.device if hasattr(y, 'device') else 'cpu'
        dtype  = y.dtype  if hasattr(y, 'dtype')  else None
        if dtype is not None:
            targets = bc.get_outputs(device=device, dtype=dtype)  # (N, K)
        else:
            targets = bc.get_outputs()
        out_names = bc.get_input_names()
        base_w = (weights_dict or {}).get(bc.name, 1.0) if (weights_dict and bc.name) else 1.0
        result = {}
        for k, (comp, oname) in enumerate(zip(bc.components, out_names)):
            diff = y[:, comp] - targets[:, k]
            w = (weights_dict or {}).get(oname, base_w)
            result[oname] = w * self._mean_squared(diff)
        return result

    def _make_bc_ops(self, params_dict):
        """Build a :class:`~pinns.boundary.TermOps` bundle for this training step.

        Wraps the backend-specific tensor helpers into an object that BC classes
        can use to compute losses without importing any backend.
        """
        from pinns.terms import TermOps
        return TermOps(
            to_tensor=self._to_tensor,
            mean_sq=self._mean_squared,
            directional_derivative=lambda x, comp, dim:
                self._compute_directional_derivative(x, comp, dim, params_dict),
            params_dict=params_dict,
        )

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
        from pinns.problems.problem_strong import ProblemStrong as _PSctl
        if isinstance(self.problem, _PSctl):
            return self._compute_strong_total_loss(data, params_dict, weights_dict)

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
        
        # BC losses — delegate sampling and loss to each BC object
        from pinns.terms import TermPeriodicBC as _PBC3
        _periodic_types3 = (_PBC3,)
        bc_names = self._get_bc_names()
        losses['bcs'] = []
        name_idx = 0
        ops = self._make_bc_ops(params_dict)
        for i, bc in enumerate(self.problem.boundary_conditions):
            if _periodic_types3 and isinstance(bc, _periodic_types3):
                continue   # handled by the JIT train step, not here
            name = bc_names[name_idx]
            name_idx += 1
            if name in data:
                x_bc = data[name]
                y_bc = self._call_network(x_bc, params_dict)
                per_output = bc.compute_loss_dict(x_bc, y_bc, ops)
                weighted_bc_loss = None
                for oname, oloss in per_output.items():
                    w = weights_dict.get(oname, weights_dict.get(name, 1.0))
                    weighted = w * oloss
                    losses[oname] = weighted
                    weighted_bc_loss = (weighted if weighted_bc_loss is None
                                        else weighted_bc_loss + weighted)
                if weighted_bc_loss is not None:
                    losses['bcs'].append(weighted_bc_loss)
                    total_loss = (weighted_bc_loss if total_loss is None
                                  else total_loss + weighted_bc_loss)

        return total_loss, losses

    def _compute_strong_total_loss(self, data: Dict, params_dict: Dict[str, Any], weights_dict: Dict):
        """
        Compute total weighted loss for ProblemStrong problems.

        Each registered term's residual is evaluated via ``term.fn(x, u, pars, derivative)``.
        Structural BC terms (add_dirichlet, add_neumann) with no callable fn use
        ``_compute_strong_bc_residual``.

        Returns:
            Tuple of (total_loss, losses_dict)
        """
        total_loss = None
        losses = {'bcs': []}
        derivative_fn = self._get_derivative_fn(params_dict)

        for term in self.problem._terms:
            if term.name not in data:
                continue
            x = data[term.name]
            # PyTorch networks expose .forward(); JAX ModelBase uses .apply(params, x).
            if hasattr(self.network, 'forward'):
                y = self.network.forward(x, params_dict)
            else:
                y = self.network.apply(self.network.params, x, params_dict)

            if term.kind == 'points':
                # Data-fitting term: minimise ||u_pred - u_data||^2
                output_col = term.output_idx if term.output_idx is not None else 0
                target = self._to_tensor(np.asarray(term.u_data).flatten())
                residual = y[:, output_col] - target

            elif term.fn is not None and callable(term.fn):
                # User-provided physics residual: fn(x, u, pars, derivative)
                residual = term.fn(x, y, params_dict, derivative_fn)
                # Multi-equation functions: pick the relevant column
                if term.eq_idx is not None:
                    if hasattr(residual, 'ndim') and residual.ndim == 2:
                        residual = residual[:, term.eq_idx]

            elif term.fn is not None:
                # fn is a constant target value (e.g., add_initial(1, ...))
                output_col = term.output_idx if term.output_idx is not None else 0
                residual = y[:, output_col:output_col + 1] - float(term.fn)

            elif term.rhs is not None:
                # Structural BC terms (add_dirichlet, add_neumann, add_robin)
                residual = self._compute_strong_bc_residual(term, x, y, params_dict, derivative_fn)

            else:
                continue  # Periodic and other unsupported kinds: skip

            loss = self._mean_squared(residual)
            w = weights_dict.get(term.name, 1.0)
            weighted = w * loss
            losses[term.name] = weighted
            losses['bcs'].append(weighted)

            if total_loss is None:
                total_loss = weighted
            else:
                total_loss = total_loss + weighted

        return total_loss, losses

    def _compute_strong_bc_residual(self, term, x, y, params_dict, derivative_fn):
        """Compute residual for structural add_dirichlet / add_neumann / add_robin terms."""
        output_col = term.output_idx if term.output_idx is not None else 0
        u = y[:, output_col:output_col + 1]

        if term.kind in ('dirichlet', 'initial'):
            rhs = term.rhs
            if callable(rhs):
                target = rhs(x, params_dict)
                if hasattr(target, 'ndim') and target.ndim == 2:
                    target = target[:, 0:1]
            else:
                target = float(rhs)
            return u - target

        elif term.kind == 'neumann':
            raise NotImplementedError(
                f"Automatic Neumann residual for term '{term.name}' is not yet supported. "
                "Use add_boundary() with a callable fn that calls derivative(u, x, ...) directly."
            )
        elif term.kind == 'robin':
            raise NotImplementedError(
                f"Automatic Robin residual for term '{term.name}' is not yet supported. "
                "Use add_boundary() with a callable fn."
            )
        else:
            raise ValueError(f"Unexpected kind '{term.kind}' in _compute_strong_bc_residual")

    def _get_derivative_fn(self, params_dict=None):
        """Return backend-specific derivative function for use in ProblemStrong term.fn calls.

        Args:
            params_dict: Optional physics params dict to pass to the model apply function
                         (e.g. for output transforms / scaling).

        Subclasses must implement this to return the correct autodiff helper.
        """
        raise NotImplementedError("_get_derivative_fn must be implemented by the backend Trainer subclass.")

    def _compute_total_loss_batched(self, data: Dict, params_dict: Dict[str, Any], 
                                     weights_dict: Dict, batch_size: int = 1000):
        """
        Compute total weighted loss from data dict using batched evaluation.
        
        This avoids OOM when computing metrics on large datasets by chunking
        the computation and averaging results.
        
        Args:
            data: Dict with 'pde' and BC name keys mapping to input tensors
            params_dict: Parameters dict from _build_params()
            weights_dict: Dict mapping loss names to weights
            batch_size: Size of each batch for evaluation
            
        Returns:
            Tuple of (total_loss, losses_dict) as Python floats
        """
        # Check if batching is needed
        max_size = max(len(v) for v in data.values()) if data else 0
        if max_size <= batch_size:
            # Small enough to compute directly
            total, losses = self._compute_total_loss(data, params_dict, weights_dict)
            return float(total), {k: float(v) if hasattr(v, '__float__') else v for k, v in losses.items()}
        
        # Batched computation: accumulate weighted sum of losses
        n_batches = (max_size + batch_size - 1) // batch_size
        accumulated_losses = {}
        total_points_per_key = {}
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, max_size)
            
            # Create batch data
            batch_data = {}
            for name, tensor in data.items():
                n = len(tensor)
                b_start = min(start_idx, n)
                b_end = min(end_idx, n)
                if b_end > b_start:
                    batch_data[name] = self._index_tensor(tensor, list(range(b_start, b_end)))
                    total_points_per_key[name] = total_points_per_key.get(name, 0) + (b_end - b_start)
            
            if not batch_data:
                continue
            
            # Compute loss for this batch
            batch_total, batch_losses = self._compute_total_loss(batch_data, params_dict, weights_dict)
            
            # Accumulate (weighted by batch size for proper averaging)
            batch_n = end_idx - start_idx
            for name, val in batch_losses.items():
                if name == 'pde_individual' or name == 'bcs':
                    continue  # Skip list/tuple values
                if hasattr(val, '__float__'):
                    if name not in accumulated_losses:
                        accumulated_losses[name] = 0.0
                    accumulated_losses[name] += float(val) * batch_n
        
        # Average by total points
        final_losses = {}
        total_loss = 0.0
        for name, acc_val in accumulated_losses.items():
            n_points = total_points_per_key.get('pde', max_size)  # Use PDE points for averaging
            final_losses[name] = acc_val / n_points if n_points > 0 else 0.0
            total_loss += final_losses[name]
        
        return total_loss, final_losses

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
            y = self._call_network(x, params_dict)
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
            y_batch = self._call_network(x_batch, params_dict)
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

    def _evaluate_observables(self, x_np: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Evaluate ``problem.obs_fn`` at the given input points.

        ``obs_fn`` has the signature::

            obs_fn(x, y, params, derivative) -> list of (n,) arrays

        The returned list is zipped with ``problem.obs_names``.
        This base implementation passes ``None`` for ``derivative``; the JAX
        subclass overrides this to supply true autodiff derivatives.

        Returns
        -------
        dict
            ``{name: np.ndarray}`` with values of shape ``(n,)``.
        """
        import inspect as _inspect
        obs_fn = getattr(self.problem, 'obs_fn', None)
        obs_names = getattr(self.problem, 'obs_names', None) or []
        if obs_fn is None or not obs_names:
            return {}
        x = self._to_tensor(x_np)
        params_dict = self._build_params()
        y = self._call_network(x, params_dict)
        try:
            sig = _inspect.signature(obs_fn)
            n_params = len(sig.parameters)
            if n_params >= 4:
                vals = obs_fn(x, y, params_dict, None)
            elif n_params == 3:
                vals = obs_fn(x, y, params_dict)
            else:
                vals = obs_fn(x, y)
        except Exception:
            return {}
        return {
            name: self._to_numpy(v).reshape(len(x_np), -1)
            for name, v in zip(obs_names, vals)
        }

    @abstractmethod
    def _to_numpy(self, tensor) -> np.ndarray:
        """Convert backend tensor to numpy array."""
        pass
    
    # ==================== Sampling Methods ====================
    
    def _get_bc_by_name(self, name: str):
        """Get boundary condition (or ProblemStrong term) by name."""
        from pinns.problems.problem_strong import ProblemStrong as _PSbc2
        if isinstance(self.problem, _PSbc2):
            return next((t for t in self.problem._terms if t.name == name), None)
        for bc in self.problem.boundary_conditions:
            if hasattr(bc, 'name') and bc.name == name:
                return bc
        return None
    
    def _sample_points_np(self, name: str, n_samples: int) -> np.ndarray:
        """Sample collocation points for any named loss term.

        Handles all three problem types (old Problem API, ProblemStrong) and
        all BC variants via duck-typing on the BC object's attributes.  This
        is the single sampling entry point — no sub-methods needed.
        """
        from pinns.problems.problem_strong import ProblemStrong as _PSsp
        domain = self.problem.domain
        params = self._build_params()
        rng = self.rng

        # ── ProblemStrong: each term carries kind + region ─────────────────
        if isinstance(self.problem, _PSsp):
            term = next((t for t in self.problem._terms if t.name == name), None)
            if term is None:
                raise ValueError(f"Unknown ProblemStrong term: '{name}'")
            if term.kind == 'points':
                return np.asarray(term.x_data)
            region = None if term.region == 'all' else term.region
            if term.kind == 'inner':
                return domain.sample_interior(n_samples, region=region, rng=rng, params=params)
            if term.kind == 'initial':
                pts = domain.sample_interior(n_samples, rng=rng, params=params)
                t_dim = getattr(domain, '_spatial_dims', 0)
                if t_dim < pts.shape[1]:
                    pts[:, t_dim] = domain.xmin[t_dim]
                return pts
            # boundary / dirichlet / neumann / robin / periodic
            return domain.sample_boundary(n_samples, region=region, rng=rng, params=params)

        # ── Old Problem API: 'pde' = interior, everything else = BC ─────────
        if name == 'pde':
            return domain.sample_interior(n_samples, rng=rng, params=params)

        bc = self._get_bc_by_name(name)
        if bc is None:
            raise ValueError(f"Unknown loss term: {name!r}")
        return self._sample_bc_np(bc, n_samples)

    def _sample_bc_np(self, bc, n_samples: int) -> np.ndarray:
        """Sample *n_samples* points for a single BC object.

        All domain knowledge stays on the domain; this method only reads
        BC attributes to decide which domain primitive to call.
        """
        domain = self.problem.domain
        rng = self.rng

        # Fixed-coordinate BCs — ignore domain / n_samples
        if hasattr(bc, 'inputs'):                      # TermPoints
            return np.asarray(bc.inputs, dtype=np.float32)
        if getattr(bc, 'bc_type', None) == 'points':  # TermMeshPointsBC
            return np.asarray(bc.node_positions, dtype=np.float32)

        # Mesh BCs — delegate to domain.sample_boundary_bc
        if hasattr(bc, 'node_positions'):
            pts, idx = domain.sample_boundary_bc(bc, n_samples, rng=rng)
            if hasattr(bc, 'edge_normals'):
                bc._sampled_normals = (
                    bc.edge_normals[idx] if bc.edge_normals is not None else None)
            return pts

        # New-style region-based BCs (TermDirichletBC, TermNeumannBC, etc.)
        if hasattr(bc, 'region'):
            return domain.sample_boundary(
                n_samples, region=bc.region, rng=rng,
                method=getattr(bc, 'sampling_method', 'uniform'),
            )

        # Cubic face BCs — boundary tuple + optional subspace
        for dim, side in enumerate(bc.boundary):
            if side is not None:
                pts = domain.sample_boundary(
                    n_samples, dim, side, rng=rng,
                    method=getattr(bc, 'sampling_method', 'uniform'),
                    transform=getattr(bc, 'sampling_transform', None),
                )
                subspace = getattr(bc, 'subspace', None)
                if subspace is not None:
                    free_dims = [d for d in range(len(bc.boundary))
                                 if bc.boundary[d] is None]
                    for i, (lo, hi) in enumerate(subspace):
                        if i >= len(free_dims):
                            break
                        d = free_dims[i]
                        orig_lo, orig_hi = domain.xmin[d], domain.xmax[d]
                        extent = orig_hi - orig_lo
                        if extent > 0:
                            pts[:, d] = (orig_lo
                                         + (pts[:, d] - orig_lo) / extent
                                         * (hi - lo)
                                         + (lo - orig_lo))
                time_subspace = getattr(bc, 'time_subspace', None)
                if time_subspace is not None:
                    t_dim = domain._spatial_dims
                    if t_dim < pts.shape[1]:
                        t_lo, t_hi = time_subspace
                        t_orig_lo = domain.xmin[t_dim]
                        t_orig_hi = domain.xmax[t_dim]
                        t_extent = t_orig_hi - t_orig_lo
                        if t_extent > 0:
                            pts[:, t_dim] = (t_orig_lo
                                             + (pts[:, t_dim] - t_orig_lo)
                                             / t_extent * (t_hi - t_lo)
                                             + (t_lo - t_orig_lo))
                return pts
        raise ValueError(f"BC {bc!r}: no fixed dimension in boundary tuple")
    
    def _list_to_dict_samples(self, samples_list) -> Dict[str, int]:
        """Convert list format samples to dict format. Dict input is returned as-is."""
        if isinstance(samples_list, dict):
            return samples_list
        bc_names = self._get_bc_names()
        result = {'pde': samples_list[0]}
        for i, name in enumerate(bc_names):
            result[name] = samples_list[i + 1]
        return result
    
    def _list_to_dict_weights(self, weights_list) -> Dict[str, float]:
        """Convert list format weights to dict format. Dict input is returned as-is."""
        if isinstance(weights_list, dict):
            return weights_list
        bc_names = self._get_bc_names()
        result = {'pde': weights_list[0]}
        for i, name in enumerate(bc_names):
            result[name] = weights_list[i + 1]
        return result
    
    def _curriculum_step(self, global_epoch: int) -> bool:
        """
        Advance the time-domain curriculum if the current epoch crosses a stage boundary.

        The domain's upper bound along `_curriculum_t_dim` is updated and training
        data is resampled. Returns True when a stage transition occurred.
        """
        if not self._curriculum_t_ends:
            return False

        stage = min(global_epoch // self._curriculum_t_epochs, len(self._curriculum_t_ends) - 1)
        if stage == self._curriculum_t_stage:
            return False

        self._curriculum_t_stage = stage
        new_end = float(self._curriculum_t_ends[stage])
        self.problem.domain.xmax[self._curriculum_t_dim] = new_end
        self._sample_train_data()
        if self._test_data:
            self._sample_test_data()
        # NOTE: optimizer state is intentionally NOT reset here.
        # Accumulated momentum from the old region provides inertia that resists
        # the large gradients from new (unseen) collocation points overwriting
        # what was already learned.
        # If using Lagrangian mode, resize λ vectors to match new sample count
        if getattr(self, '_is_lagrangian_mode', False) and hasattr(self, '_reinitialize_lagrange_if_needed'):
            self._reinitialize_lagrange_if_needed()
        print(f"  [curriculum] Stage {stage}: domain end = {new_end}")
        return True

    def _curriculum_restore(self):
        """Restore original domain upper bound after training with curriculum."""
        if self._curriculum_t_ends and hasattr(self, '_curriculum_t_original_xmax'):
            self.problem.domain.xmax[self._curriculum_t_dim] = self._curriculum_t_original_xmax

    def _sample_train_data(self):
        """Sample training data and store as backend tensors.
        
        Also precomputes target values for BCs with callable value functions.
        """
        from pinns.problems.problem_strong import ProblemStrong as _PSsd
        _is_strong = isinstance(self.problem, _PSsd)
        self._train_data = {}
        self._train_targets = {}  # Store precomputed target values
        samples_dict = self._list_to_dict_samples(self.train_samples)
        for name, n in samples_dict.items():
            if n > 0:
                np_data = self._sample_points_np(name, n)
                self._train_data[name] = self._to_tensor(np_data)
                
                # Precompute target values for BCs with callable value functions
                # (old Problem API only; ProblemStrong computes residuals via term.fn)
                if not _is_strong and name != 'pde':
                    bc = self._get_bc_by_name(name)
                    if bc is not None and hasattr(bc, 'value') and callable(bc.value):
                        target_np = bc.value(np_data)
                        if hasattr(target_np, 'squeeze'):
                            target_np = target_np.squeeze(-1) if target_np.ndim > 1 else target_np
                        self._train_targets[name] = self._to_tensor(target_np)
                    # Store matched normals (set by _sample_boundary_np for TermMeshNodeBC)
                    if bc is not None and getattr(bc, '_sampled_normals', None) is not None:
                        self._train_data[f'{name}__normals'] = self._to_tensor(
                            bc._sampled_normals.astype(np.float32))
    
    def _sample_test_data(self):
        """Sample test data and store as backend tensors."""
        from pinns.problems.problem_strong import ProblemStrong as _PSstd
        _is_strong = isinstance(self.problem, _PSstd)
        self._test_data = {}
        self._test_targets = {}  # Store precomputed target values
        samples_dict = self._list_to_dict_samples(self.test_samples)
        for name, n in samples_dict.items():
            if n > 0:
                np_data = self._sample_points_np(name, n)
                self._test_data[name] = self._to_tensor(np_data)
                
                # Precompute target values for BCs with callable value functions
                # (old Problem API only; ProblemStrong computes residuals via term.fn)
                if not _is_strong and name != 'pde':
                    bc = self._get_bc_by_name(name)
                    if bc is not None and hasattr(bc, 'value') and callable(bc.value):
                        target_np = bc.value(np_data)
                        if hasattr(target_np, 'squeeze'):
                            target_np = target_np.squeeze(-1) if target_np.ndim > 1 else target_np
                        self._test_targets[name] = self._to_tensor(target_np)
                    # Store matched normals (set by _sample_boundary_np for TermMeshNodeBC)
                    if bc is not None and getattr(bc, '_sampled_normals', None) is not None:
                        self._test_data[f'{name}__normals'] = self._to_tensor(
                            bc._sampled_normals.astype(np.float32))

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
        """Get list of boundary condition / term names."""
        from pinns.problems.problem_strong import ProblemStrong as _PSbc
        if isinstance(self.problem, _PSbc):
            return [t.name for t in self.problem._terms]
        from pinns.terms import TermPeriodicBC as _PBC
        _periodic_types = (_PBC,)
        names = []
        for i, bc in enumerate(self.problem.boundary_conditions):
            if _periodic_types and isinstance(bc, _periodic_types):
                continue   # handled separately; no sampled data needed
            if hasattr(bc, 'name') and bc.name is not None:
                names.append(bc.name)
            else:
                names.append(f'bc_{i}')
        return names

    def _get_bc_plot_names(self) -> List[str]:
        """BC names used for plot labels/legend. Override to filter entries."""
        return self._get_bc_names()

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
        from pinns.problems.problem_strong import ProblemStrong as _PSbp
        if isinstance(self.problem, _PSbp):
            fixed = self.problem.fixed_params
        else:
            fixed = self.problem.params
        return {
            "fixed": fixed,
            "infer": {},  # Reserved for future inverse problem support
            "internal": internal
        }
    
    def _convert_dict_to_list(self, data: Union[List, Dict], param_name: str) -> List:
        """Convert a dictionary of samples/weights to list format."""
        if isinstance(data, dict):
            bc_names = self._get_bc_names()
            result = []
            
            # First element is 'pde' (or the user-supplied inner-term name for ProblemWeak)
            from pinns.problems.problem_weak import ProblemWeak as _ProblemWeak
            _default_pde = 0 if isinstance(self.problem, _ProblemWeak) else None
            _volume_name = getattr(self.problem, '_volume_name', 'pde')
            if 'pde' in data:
                result.append(data['pde'])
            elif _volume_name in data:
                # User passed the inner-term name (e.g. "coco") instead of "pde"
                result.append(data[_volume_name])
            else:
                if _default_pde is None:
                    raise ValueError(f"{param_name} dict must contain 'pde' key")
                result.append(_default_pde)
            
            # Collect hard-constrained BC names (weight is irrelevant for these)
            _hard_names: set = set()
            if isinstance(self.problem, _ProblemWeak):
                _hard_names = (
                    getattr(self.problem, 'hard_bc_names', set()) |
                    getattr(self.problem, 'hard_ic_names', set())
                )

            # Then BC values in order
            for i, bc_name in enumerate(bc_names):
                if bc_name in data:
                    result.append(data[bc_name])
                elif f'bc_{i}' in data:
                    result.append(data[f'bc_{i}'])
                elif bc_name in _hard_names:
                    result.append(0.0)   # hard-constrained — weight has no effect
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
        def _first_not_none(*attrs):
            for a in attrs:
                if a is not None:
                    return a
            return None
        rmin = _first_not_none(getattr(self.network, 'output_range_min', None),
                               getattr(self.network, 'output_min', None))
        rmax = _first_not_none(getattr(self.network, 'output_range_max', None),
                               getattr(self.network, 'output_max', None))
        if rmin is None or rmax is None:
            return 'inferno'
        try:
            if rmin[output_idx] == -rmax[output_idx]:
                return 'managua_r'
        except (IndexError, TypeError):
            pass
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
    
    def _apply_plot_kwargs(self, ax, key: str):
        """Apply user-supplied plot_kwargs for *key* to *ax* via ax.set()."""
        _IMSHOW_KEYS = {'norm', 'cmap', 'vmin', 'vmax', 'alpha', 'interpolation'}
        kwargs = {k: v for k, v in getattr(self, '_plot_kwargs', {}).get(key, {}).items()
                  if k not in _IMSHOW_KEYS}
        if kwargs:
            ax.set(**kwargs)

    def _get_imshow_kwargs(self, key: str) -> dict:
        """Return imshow-specific kwargs (norm, cmap, vmin, vmax, …) from plot_kwargs[key]."""
        _IMSHOW_KEYS = {'norm', 'cmap', 'vmin', 'vmax', 'alpha', 'interpolation'}
        return {k: v for k, v in getattr(self, '_plot_kwargs', {}).get(key, {}).items()
                if k in _IMSHOW_KEYS}

    def _apply_plot_style(self, fig, axes: dict):
        """Apply plot_style settings (theme, bg_color, fig_color, text_color) to fig/axes."""
        style = getattr(self, '_plot_style', {})
        if not style:
            return

        theme = style.get('theme', 'light')
        if theme == 'dark':
            default_bg   = '#1a1a2e'
            default_fig  = '#0f0f1a'
            default_text = 'white'
            default_grid = '#444466'
        else:
            default_bg   = 'white'
            default_fig  = '#f8f8f8'
            default_text = 'black'
            default_grid = '#cccccc'

        bg_color   = style.get('bg_color',   default_bg)
        fig_color  = style.get('fig_color',  default_fig)
        text_color = style.get('text_color', default_text)
        grid_color = style.get('grid_color', default_grid)

        fig.patch.set_facecolor(fig_color)

        for ax in axes.values():
            ax.set_facecolor(bg_color)
            ax.tick_params(colors=text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.title.set_color(text_color)
            for spine in ax.spines.values():
                spine.set_edgecolor(text_color)
            # Recolor existing gridlines
            ax.grid(True, color=grid_color, alpha=0.4)
            # Recolor legend text if present
            legend = ax.get_legend()
            if legend is not None:
                legend.get_frame().set_facecolor(bg_color)
                legend.get_frame().set_edgecolor(text_color)
                for text in legend.get_texts():
                    text.set_color(text_color)

        # Recolor colorbars
        for cbar in getattr(self, '_colorbars', []):
            cbar.ax.tick_params(colors=text_color)
            cbar.ax.yaxis.label.set_color(text_color)
            cbar.ax.xaxis.label.set_color(text_color)
            cbar.outline.set_edgecolor(text_color)
            # colorbar label (set via label= kwarg in colorbar())
            if cbar.ax.get_ylabel():
                cbar.ax.yaxis.label.set_color(text_color)
            for spine in cbar.ax.spines.values():
                spine.set_edgecolor(text_color)

    def _create_figure(self):
        """Create figure and axes for plotting."""
        n_dims = self.problem.n_dims
        n_outputs = self.problem.n_outputs
        has_solution = self.problem.solution is not None
        n_regions = len(getattr(self, '_plot_regions', []))
        obs_names = list(getattr(self.problem, 'obs_names', None) or [])
        obs_spatial = list(getattr(self.problem, 'obs_spatial', None) or [])
        # Regular observables: those not listed in obs_spatial
        obs_regular = [n for n in obs_names if n not in obs_spatial]
        has_spatial = len(obs_spatial) > 0
        n_obs = len(obs_regular) + (1 if has_spatial else 0)

        if n_dims == 1:
            if has_solution:
                n_cols = 3  # solution, residuals, error
            else:
                n_cols = 2  # solution, residuals
            
            # 2 rows for losses + mse_losses
            n_rows = 2 + n_outputs + n_regions + n_obs
            fig = plt.figure(figsize=(5 * n_cols, 3.5 * n_rows))
            gs = fig.add_gridspec(n_rows, n_cols)
            
            axes = {}
            axes['losses'] = fig.add_subplot(gs[0, :])
            axes['mse_losses'] = fig.add_subplot(gs[1, :])
            
            for i in range(n_outputs):
                axes[f'sol_{i}'] = fig.add_subplot(gs[2 + i, 0])
                axes[f'res_{i}'] = fig.add_subplot(gs[2 + i, 1])
                if has_solution:
                    axes[f'err_{i}'] = fig.add_subplot(gs[2 + i, 2])
            
            # Region plots
            for r in range(n_regions):
                axes[f'region_{r}'] = fig.add_subplot(gs[2 + n_outputs + r, :])

            # Regular observable plots
            for k, name in enumerate(obs_regular):
                axes[f'obs_{name}'] = fig.add_subplot(gs[2 + n_outputs + n_regions + k, :])
            # Joint deformed-mesh plot for spatial observables (two panels: original | deformed)
            if has_spatial:
                row_s = 2 + n_outputs + n_regions + len(obs_regular)
                axes['obs__deformed_ref'] = fig.add_subplot(gs[row_s, 0])
                axes['obs__deformed_def'] = fig.add_subplot(gs[row_s, 1])
        
        elif n_dims == 2:
            if has_solution:
                n_cols = 4  # predicted, true, residuals, error
            else:
                n_cols = 2  # predicted, residuals
            
            n_rows = 2 + n_outputs + n_regions + n_obs
            fig = plt.figure(figsize=(4 * n_cols, 3.5 * n_rows))
            gs = fig.add_gridspec(n_rows, n_cols)
            
            axes = {}
            axes['losses'] = fig.add_subplot(gs[0, :])
            axes['mse_losses'] = fig.add_subplot(gs[1, :])
            
            for i in range(n_outputs):
                axes[f'sol_{i}'] = fig.add_subplot(gs[2 + i, 0])
                if has_solution:
                    axes[f'true_{i}'] = fig.add_subplot(gs[2 + i, 1])
                    axes[f'res_{i}'] = fig.add_subplot(gs[2 + i, 2])
                    axes[f'err_{i}'] = fig.add_subplot(gs[2 + i, 3])
                else:
                    axes[f'res_{i}'] = fig.add_subplot(gs[2 + i, 1])
            
            for r in range(n_regions):
                axes[f'region_{r}'] = fig.add_subplot(gs[2 + n_outputs + r, :])

            # Regular observable plots — one subplot per observable, spanning first two columns
            for k, name in enumerate(obs_regular):
                axes[f'obs_{name}'] = fig.add_subplot(gs[2 + n_outputs + n_regions + k, :2])
            # Joint deformed-mesh plot for spatial observables (two panels: original | deformed)
            if has_spatial:
                row_s = 2 + n_outputs + n_regions + len(obs_regular)
                axes['obs__deformed_ref'] = fig.add_subplot(gs[row_s, 0])
                axes['obs__deformed_def'] = fig.add_subplot(gs[row_s, 1])
        
        elif self._is_mesh_domain() and getattr(self, '_plot_time_points', None):
            # Transient mesh domain with time snapshots.
            # Columns: predicted | true (opt) | residual | error (opt)
            # Rows: 2 loss rows + one row per (time × output)
            ts = self._plot_time_points
            n_snap = len(ts) * n_outputs
            if has_solution:
                n_cols = 4  # predicted, true, residual, error
            else:
                n_cols = 2  # predicted, residual

            n_rows = 2 + n_snap
            fig = plt.figure(figsize=(4 * n_cols, 3.5 * n_rows))
            gs = fig.add_gridspec(n_rows, n_cols)

            axes = {}
            axes['losses'] = fig.add_subplot(gs[0, :])
            axes['mse_losses'] = fig.add_subplot(gs[1, :])

            row = 2
            for t_val in ts:
                for i in range(n_outputs):
                    key = f'snap_sol_{i}_t{t_val}'
                    axes[key] = fig.add_subplot(gs[row, 0])
                    if has_solution:
                        axes[f'snap_true_{i}_t{t_val}'] = fig.add_subplot(gs[row, 1])
                        axes[f'snap_res_{i}_t{t_val}'] = fig.add_subplot(gs[row, 2])
                        axes[f'snap_err_{i}_t{t_val}'] = fig.add_subplot(gs[row, 3])
                    else:
                        axes[f'snap_res_{i}_t{t_val}'] = fig.add_subplot(gs[row, 1])
                    row += 1
        else:
            # For 3D+: loss plot + region slices for all outputs with residuals
            n_cols = 2 * n_outputs  # Two columns per output (solution + residual)
            n_rows = 2 + n_regions  # 2 for losses + one row per region
            fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
            gs = fig.add_gridspec(n_rows, n_cols)
            
            axes = {}
            axes['losses'] = fig.add_subplot(gs[0, :])
            axes['mse_losses'] = fig.add_subplot(gs[1, :])
            
            for r in range(n_regions):
                for i in range(n_outputs):
                    axes[f'region_{r}_{i}'] = fig.add_subplot(gs[2 + r, 2*i])
                    axes[f'region_res_{r}_{i}'] = fig.add_subplot(gs[2 + r, 2*i + 1])
        
        self._colorbars = []
        self._apply_plot_style(fig, axes)
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
        bc_names = self._get_bc_plot_names()
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
    
    def _plot_mse_losses(self, ax):
        """Plot MSE loss components on given axes."""
        epochs = self.history['epoch']
        if not epochs:
            return
        
        # Total MSE loss
        loss_data = self.history.get('loss', self.history.get('train_loss', []))
        if loss_data:
            ax.semilogy(epochs, loss_data, 'k-', label='MSE Total', linewidth=2)
        
        # PDE MSE losses
        pde_losses = self.history.get('loss_pde', [])
        if len(pde_losses) > 0:
            if isinstance(pde_losses[0], (list, tuple)):
                pde_array = np.array(pde_losses)
                for i in range(pde_array.shape[1]):
                    ax.semilogy(epochs, pde_array[:, i], 'b--', label=f'PDE eq{i+1}')
            else:
                ax.semilogy(epochs, pde_losses, 'b--', label='PDE')
        
        # BC MSE losses with names
        bc_names = self._get_bc_plot_names()
        bc_losses = self.history.get('loss_bcs', [])
        if bc_losses and len(bc_losses) > 0:
            bc_losses_array = np.array(bc_losses)
            if bc_losses_array.ndim == 2:
                for i in range(bc_losses_array.shape[1]):
                    bc_label = bc_names[i] if i < len(bc_names) else f'BC {i+1}'
                    ax.semilogy(epochs, bc_losses_array[:, i], '--', label=bc_label)
        
        # Test loss
        test_loss = self.history.get('test_loss', [])
        if len(test_loss) > 0:
            n_test = len(test_loss)
            test_epochs = np.linspace(epochs[0], epochs[-1], n_test).astype(int) if n_test > 1 else [epochs[-1]]
            ax.semilogy(test_epochs, test_loss, 'r:', marker='o', markersize=4, label='Test', linewidth=2)
        
        # Solution error
        sol_error = self.history.get('solution_error', [])
        if len(sol_error) > 0:
            n_err = len(sol_error)
            err_epochs = np.linspace(epochs[0], epochs[-1], n_err).astype(int) if n_err > 1 else [epochs[-1]]
            ax.semilogy(err_epochs, sol_error, 'm-', marker='s', markersize=4, label='Solution Error', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title('MSE Losses (Components)')
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_solution_1d(self, ax, output_idx, n_points=200):
        """Plot 1D solution on given axes."""
        x = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points).reshape(-1, 1)
        y = self.predict(x)
        
        # Plot true solution if available
        if self.problem.solution is not None:
            y_true = self._call_solution(x)
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
    
    def _is_mesh_domain(self):
        """Return True when the problem domain is a DomainMesh."""
        try:
            from pinns.domain import DomainMesh as _DomainMesh
            return isinstance(self.problem.domain, _DomainMesh)
        except ImportError:
            return False

    def _plot_mesh_snapshot(self, ax, output_idx, t_val, kind='sol'):
        """Plot a spatial snapshot of a transient mesh solution at time t_val.

        kind : 'sol'  – predicted u(x,y,t)
               'true' – reference solution u(x,y,t)
               'res'  – absolute weak residual |R_j| (t ignored, uses stored residual)
               'err'  – absolute pointwise error |pred - true|
        """
        import matplotlib.tri as _mtri

        dom      = self.problem.domain
        verts_xy = dom._vertices          # (N, 2)
        faces    = dom._faces             # (F, 3)
        n_verts  = len(verts_xy)
        t_dim    = getattr(dom, '_t_dim', self.problem.n_dims - 1)

        # Build space-time input: (N, n_inputs) with t injected at t_dim
        n_inputs = self.problem.n_dims
        x_st = np.zeros((n_verts, n_inputs), dtype=np.float32)
        spatial_dims = [d for d in range(n_inputs) if d != t_dim]
        for k, sd in enumerate(spatial_dims):
            x_st[:, sd] = verts_xy[:, k]
        x_st[:, t_dim] = float(t_val)

        tri_obj = _mtri.Triangulation(verts_xy[:, 0], verts_xy[:, 1], faces)

        if kind == 'res':
            # Use the true weak-form per-node residual R_j = Σ_k ∫_T_k φ_j·volume_fn dΩ
            # evaluated at the requested time t_val.  Falls back to a placeholder if
            # the machinery is not available (e.g. non-weak problem).
            try:
                from pinns.problems.problem_weak import ProblemWeak as _PW
                _u_and_grad = getattr(self, '_u_and_grad_fn', None)
                if not isinstance(self.problem, _PW) or _u_and_grad is None:
                    raise ValueError('weak residual not available')
                import jax as _jax
                _res_fn = _jax.jit(
                    self.problem.make_residual_vector_fn_at_t(_u_and_grad, float(t_val))
                )
                R_full = np.array(_res_fn(self.network.params))   # (n_dofs * n_comp,)
                _n_dofs = self.problem.n_dofs
                # Take the component for output_idx (row block)
                R_comp = R_full[output_idx * _n_dofs:(output_idx + 1) * _n_dofs]
                # Place values on nodes; non-free nodes get NaN (Dirichlet BCs)
                vals = np.full(n_verts, np.nan, dtype=float)
                for _nj in self.problem.free_nodes:
                    if int(_nj) < n_verts:
                        vals[int(_nj)] = float(np.abs(R_comp[int(_nj)]))
                # Normalise by node support area so the colour scale is O(residual)
                _node_norm = getattr(self.problem, 'node_norm', None)
                if _node_norm is not None:
                    for _nj in self.problem.free_nodes:
                        if int(_nj) < n_verts and _node_norm[int(_nj)] > 0:
                            vals[int(_nj)] /= float(_node_norm[int(_nj)])
                # Mask triangles where ALL three vertices are non-free (Dirichlet)
                tri_mask = np.array([
                    np.isnan(vals[f[0]]) and np.isnan(vals[f[1]]) and np.isnan(vals[f[2]])
                    for f in faces
                ])
                tri_obj.set_mask(tri_mask)
                # Fill NaN boundary nodes with 0 for contour plotting
                vals_plot = np.where(np.isnan(vals), 0.0, vals)
            except Exception:
                ax.text(0.5, 0.5, 'Residual\nnot available',
                        ha='center', va='center', transform=ax.transAxes, fontsize=9, color='gray')
                ax.set_title(f'|R_j| (t={t_val:.3g})')
                return
            cmap = 'inferno'
            label = f'|R_j| (t={t_val:.3g})'
            im = ax.tricontourf(tri_obj, vals_plot, levels=50, cmap=cmap)
        else:
            y_pred = self.predict(x_st)[:, output_idx]
            if kind == 'true':
                if self.problem.solution is None:
                    return
                y_true_raw = self._call_solution(x_st)
                if isinstance(y_true_raw, (list, tuple)):
                    y_true_raw = np.concatenate(
                        [np.atleast_2d(yt).T if yt.ndim == 1 else yt for yt in y_true_raw], axis=1)
                elif y_true_raw.ndim == 1:
                    y_true_raw = y_true_raw.reshape(-1, 1)
                vals = y_true_raw[:, output_idx].astype(float)
                label = f'True (t={t_val:.3g})'
                cmap = self._get_colormap(output_idx)
            elif kind == 'err':
                if self.problem.solution is None:
                    return
                y_true_raw = self._call_solution(x_st)
                if isinstance(y_true_raw, (list, tuple)):
                    y_true_raw = np.concatenate(
                        [np.atleast_2d(yt).T if yt.ndim == 1 else yt for yt in y_true_raw], axis=1)
                elif y_true_raw.ndim == 1:
                    y_true_raw = y_true_raw.reshape(-1, 1)
                vals = np.abs(y_pred - y_true_raw[:, output_idx]).astype(float)
                label = f'|Error| (t={t_val:.3g})'
                cmap = 'Reds'
            else:  # 'sol'
                vals = y_pred.astype(float)
                label = f'Predicted (t={t_val:.3g})'
                cmap = self._get_colormap(output_idx)
            # Mask triangles where any vertex has a non-finite value (e.g. NaN
            # returned by the reference interpolator near irregular boundaries).
            nan_mask = np.array([
                not (np.isfinite(vals[f[0]]) and np.isfinite(vals[f[1]]) and np.isfinite(vals[f[2]]))
                for f in faces
            ])
            tri_obj.set_mask(nan_mask)
            vals_plot = np.where(np.isfinite(vals), vals, 0.0)
            im = ax.tricontourf(tri_obj, vals_plot, levels=50, cmap=cmap)

        ax.triplot(tri_obj, color='gray', lw=0.3, alpha=0.3)
        cbar = self._fig.colorbar(im, ax=ax)
        self._colorbars.append(cbar)
        out_name = self._get_output_name(output_idx)
        ax.set_title(f'{label} ({out_name})')
        ax.set_xlabel(self._get_input_name(spatial_dims[0]))
        ax.set_ylabel(self._get_input_name(spatial_dims[1]) if len(spatial_dims) > 1 else '')
        ax.set_aspect('equal')

    def _plot_solution_2d(self, ax, output_idx, n_points=50, plot_key='solution'):
        """Plot 2D solution as heatmap on given axes."""
        cmap = self._get_colormap(output_idx)
        ikw = {'cmap': cmap}
        ikw.update(self._get_imshow_kwargs(plot_key))

        if self._is_mesh_domain():
            dom = self.problem.domain
            tri = mtri.Triangulation(dom._vertices[:, 0], dom._vertices[:, 1], dom._faces)
            y = self.predict(dom._vertices)
            vals = y[:, output_idx]
            im = ax.tricontourf(tri, vals, levels=50, **ikw)
            ax.triplot(tri, color='gray', lw=0.3, alpha=0.3)
        else:
            x0 = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points)
            x1 = np.linspace(self.problem.xmin[1], self.problem.xmax[1], n_points)
            X0, X1 = np.meshgrid(x0, x1)
            x_flat = np.column_stack([X0.ravel(), X1.ravel()])
            y = self.predict(x_flat)
            Y = y[:, output_idx].reshape(X0.shape)
            extent = [x0.min(), x0.max(), x1.min(), x1.max()]
            im = ax.imshow(Y, extent=extent, origin='lower', aspect='equal', **ikw)

        cbar = self._fig.colorbar(im, ax=ax)
        self._colorbars.append(cbar)

        output_name = self._get_output_name(output_idx)
        ax.set_title(f'Predicted ({output_name})')
        ax.set_xlabel(self._get_input_name(0))
        ax.set_ylabel(self._get_input_name(1))
    
    def _plot_true_solution_2d(self, ax, output_idx, n_points=50, plot_key='solution'):
        """Plot 2D true solution as heatmap on given axes."""
        if self.problem.solution is None:
            return

        cmap = self._get_colormap(output_idx)
        ikw = {'cmap': cmap}
        ikw.update(self._get_imshow_kwargs(plot_key))

        def _normalise(y_true):
            if isinstance(y_true, (list, tuple)):
                y_true = np.concatenate([np.atleast_2d(yt).T if yt.ndim == 1 else yt for yt in y_true], axis=1)
            elif y_true.ndim == 1:
                y_true = y_true.reshape(-1, 1)
            return y_true

        if self._is_mesh_domain():
            dom = self.problem.domain
            tri = mtri.Triangulation(dom._vertices[:, 0], dom._vertices[:, 1], dom._faces)
            y_true = _normalise(self._call_solution(dom._vertices))
            vals = y_true[:, output_idx]
            im = ax.tricontourf(tri, vals, levels=50, **ikw)
            ax.triplot(tri, color='gray', lw=0.3, alpha=0.3)
        else:
            x0 = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points)
            x1 = np.linspace(self.problem.xmin[1], self.problem.xmax[1], n_points)
            X0, X1 = np.meshgrid(x0, x1)
            x_flat = np.column_stack([X0.ravel(), X1.ravel()])
            y_true = _normalise(self._call_solution(x_flat))
            Y_true = y_true[:, output_idx].reshape(X0.shape)
            extent = [x0.min(), x0.max(), x1.min(), x1.max()]
            im = ax.imshow(Y_true, extent=extent, origin='lower', aspect='equal', **ikw)

        cbar = self._fig.colorbar(im, ax=ax)
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
        
        y_true = self._call_solution(x)
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
    
    def _plot_error_2d(self, ax, output_idx, n_points=50, plot_key='error'):
        """Plot 2D absolute error as heatmap."""
        if self.problem.solution is None:
            return

        ikw = {'cmap': 'Reds'}
        ikw.update(self._get_imshow_kwargs(plot_key))

        def _normalise(y_true):
            if isinstance(y_true, (list, tuple)):
                y_true = np.concatenate([np.atleast_2d(yt).T if yt.ndim == 1 else yt for yt in y_true], axis=1)
            elif y_true.ndim == 1:
                y_true = y_true.reshape(-1, 1)
            return y_true

        if self._is_mesh_domain():
            dom = self.problem.domain
            tri = mtri.Triangulation(dom._vertices[:, 0], dom._vertices[:, 1], dom._faces)
            y = self.predict(dom._vertices)
            y_true = _normalise(self._call_solution(dom._vertices))
            error = np.abs(y[:, output_idx] - y_true[:, output_idx])
            im = ax.tricontourf(tri, error, levels=50, **ikw)
            ax.triplot(tri, color='gray', lw=0.3, alpha=0.3)
        else:
            x0 = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points)
            x1 = np.linspace(self.problem.xmin[1], self.problem.xmax[1], n_points)
            X0, X1 = np.meshgrid(x0, x1)
            x_flat = np.column_stack([X0.ravel(), X1.ravel()])
            y = self.predict(x_flat)
            y_true = _normalise(self._call_solution(x_flat))
            error = np.abs(y[:, output_idx] - y_true[:, output_idx]).reshape(X0.shape)
            extent = [x0.min(), x0.max(), x1.min(), x1.max()]
            im = ax.imshow(error, extent=extent, origin='lower', aspect='equal', **ikw)

        cbar = self._fig.colorbar(im, ax=ax)
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
            cbar = self._fig.colorbar(im, ax=ax)
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
            cbar = self._fig.colorbar(im, ax=ax, label='|Residual|')
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
        self._apply_plot_kwargs(axes['losses'], 'losses')
        
        # Plot MSE losses if axis exists
        if 'mse_losses' in axes:
            self._plot_mse_losses(axes['mse_losses'])
            self._apply_plot_kwargs(axes['mse_losses'], 'mse_losses')
        
        if n_dims == 1:
            for i in range(n_outputs):
                if f'sol_{i}' in axes:
                    self._plot_solution_1d(axes[f'sol_{i}'], i, n_points)
                    self._apply_plot_kwargs(axes[f'sol_{i}'], 'solution')
                if f'res_{i}' in axes:
                    self._plot_residuals_1d(axes[f'res_{i}'], i, n_points)
                    self._apply_plot_kwargs(axes[f'res_{i}'], 'residuals')
                if f'err_{i}' in axes and has_solution:
                    self._plot_error_1d(axes[f'err_{i}'], i, n_points)
                    self._apply_plot_kwargs(axes[f'err_{i}'], 'error')
        
        elif n_dims == 2:
            for i in range(n_outputs):
                if f'sol_{i}' in axes:
                    self._plot_solution_2d(axes[f'sol_{i}'], i, n_points, plot_key='solution')
                    self._apply_plot_kwargs(axes[f'sol_{i}'], 'solution')
                if f'true_{i}' in axes and has_solution:
                    self._plot_true_solution_2d(axes[f'true_{i}'], i, n_points, plot_key='solution')
                    self._apply_plot_kwargs(axes[f'true_{i}'], 'solution')
                if f'res_{i}' in axes:
                    self._plot_residuals_2d(axes[f'res_{i}'], i, n_points, plot_key='residuals')
                    self._apply_plot_kwargs(axes[f'res_{i}'], 'residuals')
                if f'err_{i}' in axes and has_solution:
                    self._plot_error_2d(axes[f'err_{i}'], i, n_points, plot_key='error')
                    self._apply_plot_kwargs(axes[f'err_{i}'], 'error')

        # ── Transient mesh snapshots (works regardless of n_dims) ──────────────
        _pts = getattr(self, '_plot_time_points', None)
        if _pts and self._is_mesh_domain():
            for _t_val in _pts:
                for _i in range(n_outputs):
                    _sol_key  = f'snap_sol_{_i}_t{_t_val}'
                    _true_key = f'snap_true_{_i}_t{_t_val}'
                    _res_key  = f'snap_res_{_i}_t{_t_val}'
                    _err_key  = f'snap_err_{_i}_t{_t_val}'
                    if _sol_key in axes:
                        self._plot_mesh_snapshot(axes[_sol_key], _i, _t_val, 'sol')
                    if _true_key in axes and has_solution:
                        self._plot_mesh_snapshot(axes[_true_key], _i, _t_val, 'true')
                    if _res_key in axes:
                        self._plot_mesh_snapshot(axes[_res_key], _i, _t_val, 'res')
                    if _err_key in axes and has_solution:
                        self._plot_mesh_snapshot(axes[_err_key], _i, _t_val, 'err')

        # Plot observables (1D or 2D)
        obs_spatial = list(getattr(self.problem, 'obs_spatial', None) or [])
        for obs_name in (getattr(self.problem, 'obs_names', None) or []):
            if obs_name in obs_spatial:
                continue  # handled below as joint deformed mesh
            ax_key = f'obs_{obs_name}'
            if ax_key not in axes:
                continue
            if n_dims == 1:
                self._plot_observable_1d(axes[ax_key], obs_name, n_points)
            else:
                self._plot_observable_2d(axes[ax_key], obs_name, n_points)
        # Joint deformed-mesh plot
        if obs_spatial and 'obs__deformed_ref' in axes:
            self._plot_deformed_mesh_spatial(axes['obs__deformed_ref'], axes['obs__deformed_def'], obs_spatial, n_points)

        # Plot regions (for any dimension)
        regions = getattr(self, '_plot_regions', [])
        n_outputs = self.problem.n_outputs
        for r, region in enumerate(regions):
            # Check if we have per-output region axes (3D+ case)
            if f'region_{r}_0' in axes:
                for i in range(n_outputs):
                    if f'region_{r}_{i}' in axes:
                        self._plot_region_nd(axes[f'region_{r}_{i}'], i, region, n_points)
                        self._apply_plot_kwargs(axes[f'region_{r}_{i}'], 'region')
                        self._apply_plot_kwargs(axes[f'region_{r}_{i}'], f'region_{r}')
                    if f'region_res_{r}_{i}' in axes:
                        self._plot_region_residuals_nd(axes[f'region_res_{r}_{i}'], i, region, n_points)
                        self._apply_plot_kwargs(axes[f'region_res_{r}_{i}'], 'region')
                        self._apply_plot_kwargs(axes[f'region_res_{r}_{i}'], f'region_{r}')
            elif f'region_{r}' in axes:
                # 1D/2D case: single axis per region (plots first output)
                self._plot_region_nd(axes[f'region_{r}'], 0, region, n_points)
                self._apply_plot_kwargs(axes[f'region_{r}'], 'region')
                self._apply_plot_kwargs(axes[f'region_{r}'], f'region_{r}')
        
        self._apply_plot_style(fig, axes)
        fig.tight_layout()
    
    def plot_progress(self, save_path=None, n_points=200, fig=None, axes=None, display_handle=None, **kwargs):
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
        from pinns.problems.problem_weak import ProblemWeak as _ProblemWeak
        if isinstance(self.problem, _ProblemWeak):
            self._plot_weak_residuals_on_mesh(ax, output_idx)
            return
        x_np = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points).reshape(-1, 1)
        
        residuals = self._compute_residuals(x_np)
        
        if output_idx < len(residuals):
            res = np.abs(residuals[output_idx]).flatten()
        else:
            res = np.zeros(n_points)
        
        ax.plot(x_np.flatten(), res, 'm-', linewidth=2)
        
        output_name = self._get_output_name(output_idx)
        input_name = self._get_input_name(0)
        ax.set_xlabel(input_name)
        ax.set_ylabel(f'|Residual| ({output_name})')
        ax.set_title(f'PDE Residual ({output_name})')
        ax.grid(True, alpha=0.3)

    def _plot_weak_residuals_on_mesh(self, ax, output_idx=0):
        """Plot the nodal weak-form residual |R_j| on the mesh triangulation.

        Only free (interior) nodes are shown; Dirichlet boundary nodes are
        masked (NaN / white) because their residual is never minimized and
        has no meaningful interpretation in the loss.
        """
        import matplotlib.tri as _mtri
        weak_res_fn = getattr(self, '_weak_residual_fn', None)
        if weak_res_fn is None:
            ax.text(0.5, 0.5, 'Weak residual\nnot available',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='gray')
            ax.set_title('Weak Residual')
            return

        R_raw = weak_res_fn(self.network.params)
        # make_residual_vectors_fn returns a dict {name: array}; flatten to 1-D
        if isinstance(R_raw, dict):
            import jax.numpy as _jnp
            R_raw = _jnp.concatenate(list(R_raw.values()))
        R = np.array(R_raw)   # (n_dofs,) or (n_dofs * n_terms,)

        dom        = self.problem.domain
        verts_xy   = dom._vertices                        # (n_verts, 2)
        faces      = dom._faces                           # (n_faces, 3)
        n_verts    = len(verts_xy)

        # R at the original vertex nodes (first n_verts DOFs for any Lagrange order)
        R_verts = np.abs(R[:n_verts]).astype(float)

        # Mask Dirichlet boundary nodes — set to NaN so they are white in the plot.
        free_nodes = getattr(self.problem, 'free_nodes', None)
        if free_nodes is not None:
            free_set = set(int(i) for i in free_nodes if int(i) < n_verts)
            mask = np.array([i not in free_set for i in range(n_verts)])
            R_verts[mask] = np.nan

        # Build a triangulation from the original mesh connectivity.
        # Mask any triangle that has at least one fully-masked (boundary) vertex
        # on ALL three corners — tricontourf cannot handle interior NaN nodes.
        tri_obj = _mtri.Triangulation(verts_xy[:, 0], verts_xy[:, 1], faces)

        # Mask triangles whose nodes are ALL outside free_set (pure boundary triangles)
        if free_nodes is not None:
            tri_mask = np.array([
                np.isnan(R_verts[f[0]]) and np.isnan(R_verts[f[1]]) and np.isnan(R_verts[f[2]])
                for f in faces
            ])
            tri_obj.set_mask(tri_mask)

        # For any remaining triangle that still has a NaN corner, replace NaN
        # with the mean of the valid corners (edge-boundary triangles touching
        # both a free interior node and a Dirichlet corner).
        # This avoids the "masked points within triangulation" ValueError from
        # tricontourf while keeping the colormap driven by free-node values.
        R_plot = R_verts.copy()
        nan_mask = np.isnan(R_plot)
        if nan_mask.any():
            valid_mean = float(np.nanmean(R_plot)) if not np.all(nan_mask) else 0.0
            R_plot[nan_mask] = valid_mean

        ikw = {'cmap': 'inferno'}
        ikw.update(self._get_imshow_kwargs('residuals'))
        try:
            im = ax.tricontourf(tri_obj, R_plot, levels=50, **ikw)
        except Exception:
            # Fallback: scatter plot coloured by free-node residuals
            sc = ax.scatter(verts_xy[:, 0], verts_xy[:, 1],
                            c=R_plot, cmap='inferno', s=8)
            im = sc

        # Overlay the original mesh edges
        ax.triplot(tri_obj, color='white', lw=0.3, alpha=0.4)
        cbar = self._fig.colorbar(im, ax=ax, label='|R_j| (free nodes only)')
        self._colorbars.append(cbar)
        ax.set_title('Weak Residual |R_j| (free nodes)')
        ax.set_xlabel(self._get_input_name(0))
        ax.set_ylabel(self._get_input_name(1))

    def _plot_residuals_2d(self, ax, output_idx, n_points=50, plot_key='residuals'):
        """Plot 2D PDE residuals as heatmap."""
        from pinns.problems.problem_weak import ProblemWeak as _ProblemWeak
        if isinstance(self.problem, _ProblemWeak):
            self._plot_weak_residuals_on_mesh(ax, output_idx)
            return
        ikw = {'cmap': 'viridis'}
        ikw.update(self._get_imshow_kwargs(plot_key))

        if self._is_mesh_domain():
            dom = self.problem.domain
            tri = mtri.Triangulation(dom._vertices[:, 0], dom._vertices[:, 1], dom._faces)
            residuals = self._compute_residuals(dom._vertices)
            if output_idx < len(residuals):
                res = np.abs(residuals[output_idx]).flatten()
            else:
                res = np.zeros(dom._vertices.shape[0])
            im = ax.tricontourf(tri, res, levels=50, **ikw)
            ax.triplot(tri, color='gray', lw=0.3, alpha=0.3)
        else:
            x0 = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points)
            x1 = np.linspace(self.problem.xmin[1], self.problem.xmax[1], n_points)
            X0, X1 = np.meshgrid(x0, x1)
            x_np = np.column_stack([X0.ravel(), X1.ravel()])
            try:
                residuals = self._compute_residuals(x_np)
            except Exception:
                residuals = []
            expected = X0.size
            if output_idx < len(residuals):
                res = np.abs(residuals[output_idx]).flatten()
                if res.size != expected:
                    res = np.zeros(expected)
            else:
                res = np.zeros(expected)
            Res = res.reshape(X0.shape)
            extent = [x0.min(), x0.max(), x1.min(), x1.max()]
            im = ax.imshow(Res, extent=extent, origin='lower', aspect='equal', **ikw)

        cbar = self._fig.colorbar(im, ax=ax, label='|Residual|')
        self._colorbars.append(cbar)

        output_name = self._get_output_name(output_idx)
        ax.set_title(f'PDE Residual ({output_name})')
        ax.set_xlabel(self._get_input_name(0))
        ax.set_ylabel(self._get_input_name(1))

    # ==================== Observable Plotting ====================

    def _plot_observable_1d(self, ax, obs_name: str, n_points: int = 200):
        """Plot a 1D observable field on the given axes."""
        x_np = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points).reshape(-1, 1)
        obs = self._evaluate_observables(x_np)
        if obs_name not in obs:
            ax.text(0.5, 0.5, f'Observable\n{obs_name!r}\nnot available',
                    ha='center', va='center', transform=ax.transAxes)
            return
        vals = obs[obs_name]   # (n, 1) or (n,)
        ax.plot(x_np.flatten(), vals.flatten(), linewidth=2)
        ax.set_xlabel(self._get_input_name(0))
        ax.set_ylabel(obs_name)
        ax.set_title(f'Observable: {obs_name}')
        ax.grid(True, alpha=0.3)

    def _plot_observable_2d(self, ax, obs_name: str, n_points: int = 50):
        """Plot a scalar 2D observable as a filled contour / heatmap."""
        ikw = {'cmap': 'viridis'}
        if self._is_mesh_domain():
            dom = self.problem.domain
            x_np = dom._vertices
            obs = self._evaluate_observables(x_np)
        else:
            x0 = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points)
            x1 = np.linspace(self.problem.xmin[1], self.problem.xmax[1], n_points)
            X0, X1 = np.meshgrid(x0, x1)
            x_np = np.column_stack([X0.ravel(), X1.ravel()])
            obs = self._evaluate_observables(x_np)

        if obs_name not in obs:
            ax.text(0.5, 0.5, f'Observable\n{obs_name!r}\nnot available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Observable: {obs_name}')
            return

        vals = obs[obs_name]   # (n, k)
        scalar = vals[:, 0] if vals.ndim == 2 else vals.flatten()

        if self._is_mesh_domain():
            tri = mtri.Triangulation(x_np[:, 0], x_np[:, 1], dom._faces)
            im = ax.tricontourf(tri, scalar, levels=50, **ikw)
            ax.triplot(tri, color='gray', lw=0.3, alpha=0.3)
        else:
            Y = scalar.reshape(n_points, n_points)
            extent = [x_np[:, 0].min(), x_np[:, 0].max(),
                      x_np[:, 1].min(), x_np[:, 1].max()]
            im = ax.imshow(Y, extent=extent, origin='lower', aspect='equal', **ikw)

        cbar = self._fig.colorbar(im, ax=ax)
        self._colorbars.append(cbar)
        ax.set_title(f'Observable: {obs_name}')
        ax.set_xlabel(self._get_input_name(0))
        ax.set_ylabel(self._get_input_name(1))

    def _plot_deformed_mesh_spatial(self, ax_ref, ax_def, obs_spatial: list, n_points: int = 50):
        """Side-by-side deformed-mesh plot.

        ``obs_spatial`` is an ordered list of observable names whose scalar
        values are the **absolute new positions** of each node (e.g. ``x + u1``,
        ``y + u2``).

        * Left panel (``ax_ref``): original (undeformed) mesh
        * Right panel (``ax_def``): deformed mesh coloured by ``‖displacement‖``
        """
        if self._is_mesh_domain():
            dom = self.problem.domain
            x_np = dom._vertices   # (n, 2)
        else:
            x0 = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points)
            x1 = np.linspace(self.problem.xmin[1], self.problem.xmax[1], n_points)
            X0, X1 = np.meshgrid(x0, x1)
            x_np = np.column_stack([X0.ravel(), X1.ravel()])

        obs = self._evaluate_observables(x_np)

        # Stack displacement components: each spatial name contributes one column
        disp_cols = []
        for name in obs_spatial:
            if name not in obs:
                for ax in (ax_ref, ax_def):
                    ax.text(0.5, 0.5, f'Spatial observable\n{name!r}\nnot available',
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Deformed mesh')
                return
            v = obs[name]   # (n,) or (n, k)
            disp_cols.append(v[:, 0] if v.ndim == 2 else v.flatten())

        x_def = np.column_stack(disp_cols)              # (n, n_spatial_dims)
        mag   = np.linalg.norm(x_def - x_np[:, :x_def.shape[1]], axis=1)  # |displacement|

        xlabel = self._get_input_name(0)
        ylabel = self._get_input_name(1)

        # Shared colour scale across both panels
        vmin, vmax = mag.min(), mag.max()
        levels = np.linspace(vmin, vmax, 51)

        # Shared axis limits: union of original and deformed extents
        all_x = np.concatenate([x_np[:, 0], x_def[:, 0]])
        all_y = np.concatenate([x_np[:, 1], x_def[:, 1]])
        pad_x = (all_x.max() - all_x.min()) * 0.05 or 0.05
        pad_y = (all_y.max() - all_y.min()) * 0.05 or 0.05
        xlim = (all_x.min() - pad_x, all_x.max() + pad_x)
        ylim = (all_y.min() - pad_y, all_y.max() + pad_y)

        if self._is_mesh_domain():
            faces = dom._faces
            ref_tri = mtri.Triangulation(x_np[:, 0], x_np[:, 1], faces)
            def_tri = mtri.Triangulation(x_def[:, 0], x_def[:, 1], faces)

            # Left: original mesh coloured by displacement magnitude
            im = ax_ref.tricontourf(ref_tri, mag, levels=levels, cmap='inferno', vmin=vmin, vmax=vmax)
            ax_ref.triplot(ref_tri, color='white', lw=0.3, alpha=0.4)

            # Right: deformed mesh coloured by displacement magnitude
            ax_def.tricontourf(def_tri, mag, levels=levels, cmap='inferno', vmin=vmin, vmax=vmax)
            ax_def.triplot(def_tri, color='white', lw=0.3, alpha=0.4)
        else:
            scatter_kw = dict(c=mag, cmap='inferno', vmin=vmin, vmax=vmax, s=4)
            im = ax_ref.scatter(x_np[:, 0], x_np[:, 1], **scatter_kw)
            ax_def.scatter(x_def[:, 0], x_def[:, 1], **scatter_kw)

        for ax, title in ((ax_ref, 'Original (undeformed)'), (ax_def, 'Deformed')):
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect('equal')
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        # Place the colorbar to the right of ax_def, and add a matching invisible
        # axes to the left of ax_ref so both plots remain the same width.
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        div_ref = make_axes_locatable(ax_ref)
        cax_dummy = div_ref.append_axes("left", size="5%", pad=0.08)
        cax_dummy.set_visible(False)
        divider = make_axes_locatable(ax_def)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        cbar = self._fig.colorbar(im, cax=cax, label='‖displacement‖')
        self._colorbars.append(cbar)

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
                        y_min = np.asarray(y_min)
                        y_max = np.asarray(y_max)
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
                # Convert to numpy
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
            x_train = self.problem.domain.sample_interior(
                self.train_samples[0], rng=self.rng)
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
            x_train = self.problem.domain.sample_interior(
                self.train_samples[0], rng=self.rng)
            n_train = len(x_train)
            train_size = max(1, min(20, 500 / n_train))
            ax.scatter(x_train[:, 0], x_train[:, 1], s=train_size, c=train_color,
                      alpha=1, marker='.', label=f'Train ({n_train})', zorder=5)
        
        for i, bc in enumerate(self.problem.boundary_conditions):
            if self.train_samples[i + 1] > 0:
                x_bc = self._sample_bc_np(bc, self.train_samples[i + 1])
                n_bc = len(x_bc)
                bc_size = max(2, min(30, 300 / n_bc))
                ax.scatter(x_bc[:, 0], x_bc[:, 1], s=bc_size, c=bc_color,
                          alpha=1, marker='x', zorder=6)
    
    # ==================== Solution Error Computation ====================

    def _call_solution(self, x: np.ndarray) -> np.ndarray:
        """Call problem.solution with either 1-arg or 2-arg signature."""
        try:
            return self.problem.solution(x, self._build_params())
        except TypeError:
            return self.problem.solution(x)

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

        # For transient mesh domains, evaluate only at the plot_time_points
        # snapshots (cheap — no large random interpolation over the full
        # space-time cloud).  Fall back to a single random time level if no
        # plot_time_points are configured.
        if self._is_mesh_domain():
            verts = self.problem.domain._vertices
            t_min = getattr(self.problem.domain, '_t_min', None)
            t_max = getattr(self.problem.domain, '_t_max', None)
            if t_min is not None and t_max is not None:
                t_vals = getattr(self, '_plot_time_points', None) or \
                         [float(np.random.uniform(t_min, t_max))]
                xs = [np.hstack([verts, np.full((len(verts), 1), tv)]) for tv in t_vals]
                x = np.vstack(xs)
            else:
                x = verts
        else:
            x = self.problem.domain.sample_interior(n_points)

        y_pred = self.predict(x)
        y_true = self._call_solution(x)
        
        if isinstance(y_true, (list, tuple)):
            y_true = np.concatenate([np.atleast_2d(y).T if y.ndim == 1 else y for y in y_true], axis=1)
        elif y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        
        # Mask out rows where the reference returns NaN (e.g. points outside
        # an irregular domain like a U-shape where the interpolator returns NaN).
        valid = np.isfinite(y_true).all(axis=1) & np.isfinite(y_pred).all(axis=1)
        if not valid.any():
            return None
        y_pred = y_pred[valid]
        y_true = y_true[valid]

        # MSE between predicted and reference solution
        return float(np.mean((y_pred - y_true) ** 2))


class Trainer(BaseTrainer):
    """
    JAX-based trainer for Physics-Informed Neural Networks.
    
    Inherits plotting, history management, and utilities from BaseTrainer.
    Implements JAX-specific training loop and autodiff.
    """
    
    def __init__(
        self,
        problem,
        network,
        device=None,
    ):
        """
        Initialize trainer.
        
        Args:
            problem: The problem to solve (domain, PDE, BCs, params).
            network: The neural network to train (FBPINN or FNN Flax module).
            device: Device to use ('cpu', 'gpu', 'tpu'). Default: auto-detect.
        """
        # Reset network parameters so to() always reinitializes fresh weights
        if hasattr(network, 'params'):
            network.params = None

        # Initialize base class (handles network.to(), normalization, defaults)
        super().__init__(problem, network, device)

        # Rollout AL mode (BPTT) — minimal state needed
        self.lagrange_lr = 1.0
        self._lagrange_lr_ratio = 1.0

    def _problem_uses_lagrange(self) -> bool:
        lagrange = getattr(self.problem, 'lagrange_multipliers', None)
        return bool(lagrange)

    def _resolve_problem_lagrange_constraints(self) -> Optional[List[str]]:
        from pinns.problems.problem_weak import ProblemWeak as _ProblemWeak
        lagrange = getattr(self.problem, 'lagrange_multipliers', None)
        if not lagrange:
            return None
        # For ProblemWeak the list already contains the correct term / BC names.
        # No remapping to 'pde' is needed — each add_inner() term keeps its own name.
        if isinstance(self.problem, _ProblemWeak):
            return list(lagrange)
        if not lagrange:
            return None

        requested = set(lagrange)
        resolved = []

        output_names = list(getattr(self.problem, 'output_names', []) or [])
        pde_tokens = {'pde'}
        for name in output_names:
            pde_tokens.add(name)
            pde_tokens.add(f"DE_{name}")
            pde_tokens.add(f"R_{name}")
        if any(token in requested for token in pde_tokens):
            resolved.append('pde')

        for bc_name in self._get_bc_names():
            if bc_name in requested:
                resolved.append(bc_name)
        return resolved

    def _get_soft_bc_names(self) -> list:
        """BC names to show in plots/history — hard-constrained BCs excluded for ProblemWeak."""
        names = super()._get_bc_names()
        from pinns.problems.problem_weak import ProblemWeak as _PW
        if isinstance(self.problem, _PW):
            _hard = self.problem._rollout_ic_bc_names
            names = [n for n in names if n not in _hard]
        return names

    def _get_bc_plot_names(self) -> list:
        """Use soft (filtered) names for plot labels."""
        return self._get_soft_bc_names()

    def compile(
        self,
        *args,
        train_samples: dict = None,
        test_samples=None,
        step_weight_exp: float = 0.0,
        schedulers: Optional[List] = None,
        **kwargs,
    ):
        """
        Compile trainer.

        Parameters
        ----------
        train_samples : dict, optional
            Per-component sample counts.  Currently supports:
            ``{'time': N}`` — override the number of random time points sampled
            per epoch when ``problem.random_time_sampling=True``.
            ``{'pde': N}`` — mini-batch N free nodes per epoch (unbiased estimator).
        test_samples : dict or list, optional
            Same format as ``train_samples``.  For ``ProblemWeak``, the ``pde``
            key is accepted and silently ignored — test metrics always evaluate
            the full-domain weak loss for stability.  Other BC keys are forwarded
            to the base class as usual.
        schedulers : list, optional
            List of Scheduler instances to use during training.
        """
        # For ProblemWeak: strip 'pde' from test_samples before base class sees it.
        # The base class would try to create collocation test points, which makes no
        # sense for ProblemWeak — node batching is handled separately below.
        from pinns.problems.problem_weak import ProblemWeak as _PW
        _is_weak = isinstance(self.problem, _PW)
        _user_wants_test = test_samples is not None
        # Capture test node count BEFORE stripping (used to build _weak_loss_fn_test).
        _rts_n_nodes_test = None
        if _is_weak and isinstance(test_samples, dict):
            _rts_n_nodes_test = test_samples.get('pde', None)
            if _rts_n_nodes_test is not None:
                _rts_n_nodes_test = int(_rts_n_nodes_test)
            test_samples = {k: v for k, v in test_samples.items() if k != 'pde'} or None
        self._rts_n_nodes_test = _rts_n_nodes_test
        self._step_weight_exp = float(step_weight_exp)

        super().compile(*args, test_samples=test_samples, schedulers=schedulers, **kwargs)
        # Weak-form test PDE loss: evaluates the weak residual on a DIFFERENT random
        # batch of nodes than the training batch, giving a true held-out PDE metric.
        # Only active when the user passes test_samples={'pde': N}.
        self._weak_test_loss = bool(_is_weak and _rts_n_nodes_test is not None)

        # For ProblemWeak: rollout IC BCs are handled separately — no soft loss removal needed here.
        if _is_weak:
            _rollout = self.problem._rollout_ic_bc_names
            for _k in _rollout:
                self._train_data.pop(_k, None)
                self._train_targets.pop(_k, None)
                if self._test_data:
                    self._test_data.pop(_k, None)
                    self._test_targets.pop(_k, None)

        self._train_samples = train_samples or {}

    def _constraint_uses_quadratic(self, constraint_name: str) -> bool:
        """Return True if a constraint keeps its quadratic penalty term."""
        no_quadratic = getattr(self.problem, 'no_quadratic', None)
        if not no_quadratic:
            return True

        if constraint_name == 'pde':
            aliases = {'pde'}
            for name in list(getattr(self.problem, 'output_names', []) or []):
                aliases.add(name)
                aliases.add(f"DE_{name}")
                aliases.add(f"R_{name}")
            return not any(alias in no_quadratic for alias in aliases)

        return constraint_name not in no_quadratic

    # ==================== Device Detection ====================
    
    def _auto_detect_device(self) -> str:
        """Auto-detect the best available device using JAX."""
        return jax.devices()[0].platform  # 'cpu', 'gpu', 'tpu'
    
    # ==================== Optimizer ====================
    
    def _create_optimizer(self):
        """Create the optimizer with injectable hyperparameters for LR scheduling."""
        lr_scheduler = getattr(self, '_lr_scheduler', None)
        grad_clip    = getattr(self, '_grad_clip', None)

        def _wrap(opt):
            """Optionally prepend gradient clipping."""
            if grad_clip is not None:
                return optax.chain(optax.clip_by_global_norm(grad_clip), opt)
            return opt

        if self.optimizer_name == "adam":
            if lr_scheduler is not None:
                return _wrap(optax.inject_hyperparams(optax.adam)(learning_rate=self.learning_rate))
            return _wrap(optax.adam(self.learning_rate))
        elif self.optimizer_name == "sgd":
            if lr_scheduler is not None:
                return _wrap(optax.inject_hyperparams(optax.sgd)(learning_rate=self.learning_rate))
            return _wrap(optax.sgd(self.learning_rate))
        elif self.optimizer_name == "rmsprop":
            if lr_scheduler is not None:
                return _wrap(optax.inject_hyperparams(optax.rmsprop)(learning_rate=self.learning_rate))
            return _wrap(optax.rmsprop(self.learning_rate))
        elif self.optimizer_name == "lbfgs":
            if not HAS_JAXOPT:
                raise ImportError(
                    "L-BFGS requires jaxopt. Install with: pip install jaxopt"
                )
            # Return None - L-BFGS solver is created per training run
            return None
        elif self.optimizer_name == "soap":
            if not HAS_SOAP:
                raise ImportError(
                    "SOAP requires soap_jax. Install with: pip install git+https://github.com/haydn-jones/SOAP_JAX"
                )
            # Get SOAP-specific parameters with defaults
            soap_params = getattr(self, '_soap_params', {})
            return soap_optimizer(
                learning_rate=self.learning_rate,
                b1=soap_params.get('b1', 0.95),
                b2=soap_params.get('b2', 0.95),
                shampoo_beta=soap_params.get('shampoo_beta', -1),
                eps=soap_params.get('eps', 1e-8),
                weight_decay=soap_params.get('weight_decay', 0.0),
                precondition_frequency=soap_params.get('precondition_frequency', 10),
                max_precond_dim=soap_params.get('max_precond_dim', 10000),
                precondition_1d=soap_params.get('precondition_1d', False),
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
    
    def _init_optimizer_state(self):
        """Initialize optax optimizer state."""
        # ModelBase and ModelPartitioned (JAX) use a lazy-init pattern: params
        # are returned by network.init(rng) and not stored automatically.
        # Initialise here so the optimizer can inspect parameter shapes.
        from ..models.model_base import ModelBase as _ModelBase
        from ..models.model_partitioned import ModelPartitioned as _ModelPartitioned
        _jax_model_types = (_ModelBase, _ModelPartitioned)
        if isinstance(self.network, _jax_model_types):
            if not hasattr(self.network, 'params') or self.network.params is None:
                rng = jax.random.PRNGKey(0)
                self.network.params = self.network.init(rng)

        if self.optimizer_name == "lbfgs":
            # L-BFGS state is managed by jaxopt solver
            self.opt_state = None
        else:
            self.opt_state = self.optimizer.init(self.network.params)

    def _after_compile_hook(self):
        """Sample data and precompute FBPINN sparse data if applicable."""
        # Call parent to sample train/test data (or pool)
        super()._after_compile_hook()
        
        # Precompute sparse FBPINN data if network is FBPINN
        # Skip when batching or resampling is enabled (indices change)
        use_batching = self._batch_size is not None and self._batch_size > 0
        resample_each = getattr(self, '_resample_each', 0)
        use_resampling = resample_each > 0
        
        from ..models.model_partitioned import ModelPartitioned as _MP
        if self._use_sparse_fbpinn and isinstance(self.network, _MP) and not use_batching and not use_resampling:
            self._precompute_sparse_data()
        elif use_batching or use_resampling:
            # Clear any precomputed sparse data when batching or resampling
            self._precomputed_pde = None
            self._precomputed_bcs = {}
    
    def _precompute_sparse_data(self):
        """Precompute sparse training data for FBPINN (old PyTorch FBPINN only).

        New JAX ``ModelPartitioned`` models do not expose the precomputation API;
        this method is a no-op for those models.
        """
        if not hasattr(self.network, 'precompute_sparse_indices_jit'):
            return

        params_dict = self._build_params()
        
        # Precompute for PDE data - use sparse indices (supports derivatives)
        if 'pde' in self._train_data:
            x_pde = self._train_data['pde']
            # Store sparse indices for differentiable PDE computation
            self._precomputed_pde = self.network.precompute_sparse_indices_jit(
                x_pde, threshold=self._sparse_threshold, params_dict=params_dict
            )
        
        # Precompute for each BC - use full precomputation (no derivatives needed)
        self._precomputed_bcs = {}
        for name, x_bc in self._train_data.items():
            if name != 'pde':
                self._precomputed_bcs[name] = self.network.precompute_training_data_jit(
                    x_bc, threshold=self._sparse_threshold, params_dict=params_dict
                )

    # ==================== Tensor Conversion ====================
    
    def _to_tensor(self, np_array: np.ndarray):
        """Convert numpy array to JAX array."""
        return jnp.array(np_array)

    def _to_numpy(self, tensor) -> np.ndarray:
        """Convert JAX array to numpy."""
        return np.array(tensor)
    
    def _index_tensor(self, tensor, indices):
        """Index a JAX tensor with numpy indices."""
        return tensor[jnp.array(indices)]

    # ==================== Residual (Abstract Implementation) ====================
    
    def _get_pde_residual_tensor(self, x, y, params_dict):
        """Compute PDE residual using JAX autodiff - supports both 3-arg and 4-arg PDEs,
        and ProblemStrong (which uses _terms instead of pde_fn)."""
        from pinns.problems.problem_strong import ProblemStrong as _PSResidual
        model_apply = lambda p, xin: self.network.apply(p, xin, params_dict)

        if isinstance(self.problem, _PSResidual):
            # ProblemStrong: evaluate the first 'inner' term as the PDE residual
            inner_terms = [t for t in self.problem._terms if t.kind == 'inner']
            if not inner_terms:
                return jnp.zeros((x.shape[0], 1))
            deriv_fn = make_derivative_fn(model_apply, self.network.params)
            residuals = []
            for term in inner_terms:
                if term.fn is not None and callable(term.fn):
                    r = term.fn(x, y, params_dict, deriv_fn)
                    n = x.shape[0]
                    r = r.reshape(n, -1)[:, :1]   # normalise to (N, 1)
                    residuals.append(r)
            if not residuals:
                return jnp.zeros((x.shape[0], 1))
            return residuals[0] if len(residuals) == 1 else jnp.concatenate(residuals, axis=-1)

        # Generic path: problem.pde_fn
        sig = inspect.signature(self.problem.pde_fn)
        if len(sig.parameters) >= 4:
            # 4-arg PDE: pass derivative function directly (JIT-compatible)
            deriv_fn = make_derivative_fn(model_apply, self.network.params)
            return self.problem.pde_fn(x, y, params_dict, deriv_fn)
        else:
            # 3-arg PDE: use legacy context-based approach
            set_context(model_apply, self.network.params)
            try:
                return self.problem.pde_fn(x, y, params_dict)
            finally:
                clear_context()

    def _compute_custom_bc_losses_dict(self, bc, x, y, params_dict, weights_dict=None):
        """Return {output_name: weighted_loss} with full JAX autodiff."""
        if getattr(bc, 'is_weak', False):
            out_names = bc.output_names or [bc.name]
            return {oname: 0.0 for oname in out_names}
        model_apply = lambda p, xin: self.network.apply(p, xin, params_dict)
        deriv_fn = make_derivative_fn(model_apply, self.network.params)
        import inspect as _inspect
        sig = _inspect.signature(bc.f)
        n_params = len(sig.parameters)
        if n_params >= 4:
            residual = bc.f(x, y, params_dict, deriv_fn)
        elif n_params == 3:
            residual = bc.f(x, y, params_dict)
        else:
            residual = bc.f(x, y)
        if not isinstance(residual, (list, tuple)):
            residual = (residual,)
        out_names = (bc.output_names or ([bc.name] * len(residual)))
        default_w = (weights_dict or {}).get(bc.name, 1.0) if weights_dict else 1.0
        return {
            oname: (weights_dict or {}).get(oname, default_w) * self._mean_squared(r)
            for r, oname in zip(residual, out_names)
        }

    def _compute_custom_bc_loss(self, bc, x, y, params_dict, weights_dict=None):
        """Evaluate a TermMeshCustomBC residual with full JAX autodiff."""
        if getattr(bc, 'is_weak', False):
            return 0.0
        model_apply = lambda p, xin: self.network.apply(p, xin, params_dict)
        deriv_fn = make_derivative_fn(model_apply, self.network.params)
        import inspect as _inspect
        sig = _inspect.signature(bc.f)
        n_params = len(sig.parameters)
        if n_params >= 4:
            residual = bc.f(x, y, params_dict, deriv_fn)
        elif n_params == 3:
            residual = bc.f(x, y, params_dict)
        else:
            residual = bc.f(x, y)
        if isinstance(residual, (list, tuple)):
            out_names = (bc.output_names or ([bc.name] * len(residual)))
            default_w = (weights_dict or {}).get(bc.name, 1.0) if weights_dict else 1.0
            losses = [
                (weights_dict or {}).get(oname, default_w) * self._mean_squared(r)
                for r, oname in zip(residual, out_names)
            ]
            return sum(losses)
        default_w = (weights_dict or {}).get(bc.name, 1.0)
        return default_w * self._mean_squared(residual)

    def _evaluate_observables(self, x_np):
        """Evaluate obs_fn with full JAX autodiff derivative support."""
        import inspect as _inspect
        obs_fn   = getattr(self.problem, 'obs_fn',    None)
        obs_names = getattr(self.problem, 'obs_names', None) or []
        if obs_fn is None or not obs_names:
            return {}
        x = jnp.array(x_np)
        params_dict = self._build_params()
        model_apply = lambda p, xin: self.network.apply(p, xin, params_dict)
        y = self.network.apply(self.network.params, x, params_dict)
        deriv_fn = make_derivative_fn(model_apply, self.network.params)
        try:
            sig = _inspect.signature(obs_fn)
            n_params = len(sig.parameters)
            if n_params >= 4:
                vals = obs_fn(x, y, params_dict, deriv_fn)
            elif n_params == 3:
                vals = obs_fn(x, y, params_dict)
            else:
                vals = obs_fn(x, y)
        except Exception:
            return {}
        return {
            name: np.array(v).reshape(len(x_np), -1)
            for name, v in zip(obs_names, vals)
        }
    
    def _mean_squared(self, tensor):
        """Compute mean squared value of a JAX tensor."""
        return jnp.mean(tensor ** 2)
    
    def _compute_directional_derivative(self, x, component: int, dim: int, params_dict):
        """Compute derivative of y[component] w.r.t. x[dim] using JAX JVP."""
        params = self.network.params
        model_apply = lambda p, xin: self.network.apply(p, xin, params_dict)
        
        n_dims = x.shape[1]
        eye = jnp.eye(n_dims)
        
        def single_grad_fwd(xi):
            def scalar_output(xin):
                return model_apply(params, xin.reshape(1, -1))[0, component]
            _, du_dd = jax.jvp(scalar_output, (xi,), (eye[dim],))
            return du_dd
        
        return jax.vmap(single_grad_fwd)(x)
    
    # ==================== Backend-specific helpers ====================

    def _get_derivative_fn(self, params_dict=None):
        """Return a derivative function for ProblemStrong metric computation.

        Uses the current network parameters (non-JIT path for test loss etc.).
        For JIT training, make_derivative_fn is called inline in compute_loss_strong.
        """
        from ..functional import make_derivative_fn
        network = self.network
        params = network.params
        _pd = params_dict

        def _model_apply_metric(p, x):
            return network.apply(p, x, _pd)

        return make_derivative_fn(_model_apply_metric, params)

    # ==================== JIT Training Step ====================
    
    def _make_jit_train_step(self, weights, params_dict):
        """Create a JIT-compiled training step function."""
        from pinns.terms import TermNeumannBC, TermCustomBC, TermPeriodicBC, TermPoints as _TermPoints_jax

        # Pre-extract BC info as static data
        bc_info = []
        dirichlet_bcs = []
        neumann_bcs = []
        mesh_neumann_bcs = []   # TermMeshNodeBC with bc_type="neumann"
        custom_bc_list = []     # TermMeshCustomBC entries
        periodic_bcs  = []      # TermPeriodicBC entries (fixed precomputed points)
        
        # Get precomputed targets for callable BCs
        train_targets = getattr(self, '_train_targets', {})

        # ── Precompute TermPeriodicBC point arrays as JAX constants ──────────────
        _domain_bcs = getattr(getattr(self.problem, 'domain', None), 'boundary_conditions', [])
        _n_out = len(self.problem.output_names) if (hasattr(self.problem, 'output_names') and self.problem.output_names) else getattr(self.problem, 'n_outputs', 1)
        for bc in _domain_bcs:
            if isinstance(bc, TermPeriodicBC):
                _rng_p = np.random.default_rng()
                _n_p = bc.n_pairs or 200
                if hasattr(bc, 'node_positions_a'):
                    # Legacy mesh-style: pre-computed node arrays
                    _pts_a = np.asarray(bc.node_positions_a)
                    _pts_b = np.asarray(bc.node_positions_b)
                    _dim_p = 1
                else:
                    # New-style: region strings → sample from domain
                    _pts_a = self.problem.domain.sample_boundary(_n_p, region=bc.region_a, rng=_rng_p)
                    _pts_b = self.problem.domain.sample_boundary(_n_p, region=bc.region_b, rng=_rng_p)
                    _dim_p = 1
                _comps = [bc.component] if bc.component is not None else list(range(_n_out))
                for _i in _comps:
                    _sub_name = bc.name if bc.component is not None else f'{bc.name}_{_i}'
                    periodic_bcs.append({
                        'name':          _sub_name,
                        'x_a':           jnp.asarray(_pts_a, dtype=jnp.float32),
                        'x_b':           jnp.asarray(_pts_b, dtype=jnp.float32),
                        'component':     _i,
                        'weight':        weights.get(_sub_name, weights.get(bc.name, 1.0)),
                        'match_x_deriv': getattr(bc, 'match_x_derivative', False),
                        'x_deriv_dim':   _dim_p,
                    })

        from pinns.problems.problem_weak import ProblemWeak as _ProblemWeak
        _hard_bc_names = set()

        # ProblemStrong terms are not legacy BCs — skip that loop entirely.
        from pinns.problems.problem_strong import ProblemStrong as _PScheck
        _problem_is_strong = isinstance(self.problem, _PScheck)

        for name in self._train_data.keys():
            if _problem_is_strong:
                break
            if name == 'pde':
                continue
            # Hard-constrained BCs (ProblemWeak + output_transform) → skip soft loss
            if name in _hard_bc_names:
                continue
            bc = self._get_bc_by_name(name)
            if bc is not None:
                # TermCustomBC: captured for the custom-BC loss block below
                if isinstance(bc, TermCustomBC):
                    import inspect as _insp
                    _n = len(_insp.signature(bc.f).parameters)
                    custom_bc_list.append((bc, name, weights.get(name, 1.0), _n))
                    continue

                is_neumann = isinstance(bc, TermNeumannBC)
                if is_neumann:
                    normal_info = self.problem.domain.get_face_normal_direction(
                        getattr(bc, 'region', '')) or (0, 1)
                else:
                    normal_info = (0, 1)
                bc_data = {
                    'name': name,
                    'component': bc.component,
                    'is_neumann': is_neumann,
                    'normal_dim': normal_info[0],
                    'normal_sign': normal_info[1],
                    'const_value': bc.value if not callable(bc.value) else None,
                    'has_callable_value': callable(bc.value),
                    'weight': weights.get(name, 1.0),
                }
                bc_info.append(bc_data)
                if is_neumann:
                    neumann_bcs.append(bc_data)
                else:
                    dirichlet_bcs.append(bc_data)
        
        from pinns.problems.problem_weak import ProblemWeak as _ProblemWeak
        _is_weak = isinstance(self.problem, _ProblemWeak)

        # ── ProblemStrong: build inline JIT train step ────────────────────
        from pinns.problems.problem_strong import ProblemStrong as _ProblemStrong
        if isinstance(self.problem, _ProblemStrong):
            _terms_strong = list(self.problem._terms)
            _params_dict_s = params_dict
            _weights_s = weights
            _optim_s = self.optimizer
            # Capture any network-level extra losses (e.g. X-PINN interface terms).
            _net_losses = list(getattr(self.network, 'network_losses', []))

            def _model_apply_s(params, x):
                return self.network.apply(params, x, _params_dict_s)

            def compute_loss_strong(params, train_data, targets_dict):
                total_loss = jnp.array(0.0)
                deriv_fn = make_derivative_fn(_model_apply_s, params)
                for term in _terms_strong:
                    if term.name not in train_data:
                        continue
                    x = train_data[term.name]
                    u = _model_apply_s(params, x)
                    if term.kind == 'points':
                        output_col = term.output_idx if term.output_idx is not None else 0
                        target = jnp.array(term.u_data, dtype=jnp.float32).flatten()
                        residual = u[:, output_col] - target
                    elif term.fn is not None and callable(term.fn):
                        residual = term.fn(x, u, _params_dict_s, deriv_fn)
                        if (term.eq_idx is not None
                                and hasattr(residual, 'ndim') and residual.ndim == 2):
                            residual = residual[:, term.eq_idx]
                    elif term.fn is not None:
                        output_col = term.output_idx if term.output_idx is not None else 0
                        residual = u[:, output_col:output_col + 1] - float(term.fn)
                    elif term.kind in ('dirichlet', 'neumann', 'robin', 'initial') and hasattr(term, 'rhs'):
                        # Structural BC: add_dirichlet / add_neumann / add_robin store
                        # the prescribed value in term.rhs (not term.fn).
                        output_col = term.output_idx if term.output_idx is not None else 0
                        rhs = term.rhs
                        if callable(rhs):
                            target = rhs(x, _params_dict_s)
                        else:
                            target = float(rhs)
                        residual = u[:, output_col:output_col + 1] - target
                    else:
                        continue
                    loss = jnp.mean(residual ** 2)
                    w = _weights_s.get(term.name, 1.0)
                    total_loss = total_loss + w * loss
                # ── Network-level extra losses (architecture-driven) ──────
                for nloss in _net_losses:
                    x_nl = nloss.x if nloss.x is not None else train_data.get('pde')
                    if x_nl is not None:
                        nl_val = nloss.fn(params, x_nl)
                        w_nl = _weights_s.get(nloss.name, nloss.weight)
                        total_loss = total_loss + w_nl * nl_val
                return total_loss

            @jax.jit
            def train_step_strong(params, opt_state, train_data, targets_dict):
                loss, grads = jax.value_and_grad(compute_loss_strong)(
                    params, train_data, targets_dict)
                updates, new_opt_state = _optim_s.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return new_params, new_opt_state, loss

            return train_step_strong, True, False

        model_apply = self.network.apply
        pde_fn = None if _is_weak else self.problem.pde_fn
        pde_weight = weights.get('pde', 1.0)

        # ── Weak-form: pre-build and JIT the FEM assembler loss ──────────
        if _is_weak:
            # Per-term weights are already embedded inside make_loss_fn (bc_weights=weights),
            # so the outer pde_weight factor must be 1.0 to avoid double-weighting.
            pde_weight = 1.0
            _network = self.network
            _n_out = self.problem.n_outputs

            if getattr(self.problem, 'n_time_steps', None) is not None:
                # BPTT rollout mode: one scan over all time steps; no Lagrange needed.
                # Override lagrangian mode — BCs are hard-constrained in the network.
                self._is_lagrangian_mode = False
                # Warn if the user also requested soft (Lagrange) BCs — these
                # are silently ignored in rollout mode.  Use hard_constraints=True
                # on AlphaPINN instead (the default).
                _soft_bcs = [
                    n for n in (getattr(self.problem, 'lagrange_multipliers', None) or [])
                    if n != 'pde'
                ]
                if _soft_bcs:
                    import warnings
                    warnings.warn(
                        f"Rollout (BPTT) mode detected: soft Lagrange BCs {_soft_bcs} are "
                        "not enforced during rollout training.  Set hard_constraints=True "
                        "on the AlphaPINN network to enforce BCs exactly at every time step.",
                        UserWarning, stacklevel=3,
                    )
                # Check whether per-step AL is requested via lagrange_multipliers=["pde"]
                _uses_rollout_al = "pde" in (getattr(self.problem, 'lagrange_multipliers', None) or [])
                self._rollout_al_mode = _uses_rollout_al
                # In rollout mode IC BCs are the initial state u0, not loss terms.
                # Remove them from train/test data so no soft loss is computed.
                _rollout_ic = self.problem._rollout_ic_bc_names
                for _ic_name in _rollout_ic:
                    self._train_data.pop(_ic_name, None)
                    self._train_targets.pop(_ic_name, None)
                    if self._test_data:
                        self._test_data.pop(_ic_name, None)
                        self._test_targets.pop(_ic_name, None)
                _face_batch = getattr(self, '_rollout_face_batch', None)
                _n_faces    = self.problem.cubature_data['phi'].shape[0]  # F
                self._rollout_n_faces = _n_faces
                # Compute per-step exponential weights if requested
                _swe = getattr(self, '_step_weight_exp', 0.0)
                _n_cur_steps = self.problem.n_time_steps
                if _swe != 0.0:
                    import numpy as _np_sw
                    _sw = _np_sw.exp(_swe * _np_sw.arange(_n_cur_steps) / max(_n_cur_steps - 1, 1))
                    _step_weights = _sw.astype(_np_sw.float32)
                else:
                    _step_weights = None
                # Full-batch version (used for metrics)
                _weak_loss_fn = jax.jit(
                    self.problem.make_rollout_loss_fn(_network, step_weights=_step_weights)
                )
                self._weak_loss_fn = _weak_loss_fn
                # Training version (may be mini-batch)
                if _face_batch is not None:
                    _weak_loss_fn_train = jax.jit(
                        self.problem.make_rollout_loss_fn(_network, face_batch_size=_face_batch, step_weights=_step_weights)
                    )
                    self._weak_loss_fn_train = _weak_loss_fn_train
                else:
                    self._weak_loss_fn_train = _weak_loss_fn
                    _weak_loss_fn_train = _weak_loss_fn
                self._weak_residual_fn = None
                # ── Per-step AL (dual ascent over rollout residuals) ─────
                if _uses_rollout_al and _face_batch is None:
                    _n_cur  = self.problem.n_time_steps
                    _n_free = len(self.problem.free_nodes)
                    _al_fn  = jax.jit(self.problem.make_rollout_al_loss_fn(_network))
                    self._rollout_al_fn = _al_fn
                    # Always reset lambdas at each curriculum stage.
                    # Carrying λ from the k-step problem into the (k+1)-step problem
                    # causes the optimizer to focus on keeping R_1..R_k near-zero
                    # (dominated by large accumulated λ) at the expense of R_{k+1},
                    # which stalls learning after the first stage.
                    self._rollout_lambdas = jnp.zeros((_n_cur, _n_free), dtype=jnp.float32)
                    _optim = self.optimizer

                    @jax.jit
                    def _rollout_al_train_step(params, opt_state, lambdas):
                        def _fl(p):
                            loss, res = _al_fn(p, lambdas)
                            return loss, res
                        (loss, res), grads = jax.value_and_grad(_fl, has_aux=True)(params)
                        updates, new_opt_state = _optim.update(grads, opt_state, params)
                        new_params = optax.apply_updates(params, updates)
                        return new_params, new_opt_state, loss, res

                    self._rollout_al_train_step = _rollout_al_train_step

            else:
                # Standard single-step weak-form path
                _weak_params_dict = self.problem._build_params()
                if _n_out == 1:
                    def _u_and_grad(params, xy):
                        def u_single(z):
                            return _network.apply(params, z[None], _weak_params_dict)[0, 0]
                        return jax.value_and_grad(u_single)(xy)
                else:
                    # Multi-output: return full Jacobian (n_out, n_dims)
                    def _u_and_grad(params, xy):
                        def u_vec(z):
                            return _network.apply(params, z[None], _weak_params_dict)[0]  # (n_out,)
                        u = u_vec(xy)
                        jac = jax.jacobian(u_vec)(xy)  # (n_out, n_dims)
                        return u, jac
                _weak_loss_fn = jax.jit(self.problem.make_loss_fn(_u_and_grad, bc_weights=weights))
                self._weak_loss_fn = _weak_loss_fn
                self._u_and_grad_fn = _u_and_grad
                _is_random_time = getattr(self.problem, '_random_time_sampling', False)
                if _is_random_time:
                    _rts_t_min = float(getattr(self.problem, '_t_min', None) or 0.0)
                    _rts_t_max = float(getattr(self.problem, '_t_max', None) or 1.0)
                    _rts_n_t   = int(getattr(self.problem, '_n_t', 10))
                    _rts_n_t   = int(getattr(self, '_train_samples', {}).get('time', _rts_n_t))
                    _rts_method = getattr(self.problem, '_t_sampling_method', 'uniform')
                    self._rts_t_min = _rts_t_min
                    self._rts_t_max = _rts_t_max
                    self._rts_n_t   = _rts_n_t
                    self._rts_sampling_method = _rts_method
                    # Node mini-batching: triggered by train_samples={'pde': N}
                    _rts_n_nodes = getattr(self, '_train_samples', {}).get('pde', None)
                    if _rts_n_nodes is not None:
                        _rts_n_nodes = int(_rts_n_nodes)
                        _n_free = len(self.problem.free_nodes)
                        self._rts_n_nodes = _rts_n_nodes
                        self._rts_n_free_nodes = _n_free
                        self._weak_loss_fn_train = jax.jit(
                            self.problem.make_loss_fn(
                                _u_and_grad, bc_weights=weights,
                                node_batch_size=_rts_n_nodes,
                            )
                        )
                    # Test node batch: build a separate loss fn for test_samples={'pde': N}.
                    # Uses a different random node draw each evaluation → true held-out metric.
                    _rts_n_nodes_test = getattr(self, '_rts_n_nodes_test', None)
                    if _rts_n_nodes_test is not None:
                        _n_free = _n_free if _rts_n_nodes is not None else len(self.problem.free_nodes)
                        self._rts_n_free_nodes = getattr(self, '_rts_n_free_nodes', _n_free)
                        self._weak_loss_fn_test = jax.jit(
                            self.problem.make_loss_fn(
                                _u_and_grad, bc_weights=weights,
                                node_batch_size=_rts_n_nodes_test,
                            )
                        )
                self._weak_residual_fn = jax.jit(
                    self.problem.make_residual_vector_fn(_u_and_grad))
        else:
            _weak_loss_fn = None

        def model_apply_with_params(params, x):
            return self.network.apply(params, x, params_dict)
        
        # Check if we should use sparse FBPINN
        try:
            from pinns.models import FBPINN as _FBPINN
        except ImportError:
            _FBPINN = type(None)
        use_sparse = (self._use_sparse_fbpinn and 
                      isinstance(self.network, _FBPINN) and 
                      self._precomputed_bcs)
        
        # Check if we have sparse PDE indices for differentiable sparse forward
        use_sparse_pde = (self._use_sparse_fbpinn and 
                          isinstance(self.network, _FBPINN) and 
                          self._precomputed_pde is not None)
        
        # Store precomputed data references for closure
        precomputed_bcs = self._precomputed_bcs if use_sparse else {}
        precomputed_pde = self._precomputed_pde if use_sparse_pde else None
        network = self.network
        
        def model_apply_sparse(params, precomputed):
            """Apply network using precomputed sparse data (no derivatives)."""
            return network.apply_precomputed_jit(params, precomputed, params_dict)
        
        def model_apply_sparse_diff(params, x, sparse_indices):
            """Apply network using sparse indices with derivative support."""
            return network.apply_sparse_differentiable(params, x, sparse_indices, params_dict)
        
        pde_accepts_derivative = False
        if pde_fn is not None:
            sig = inspect.signature(pde_fn)
            pde_accepts_derivative = len(sig.parameters) >= 4
        # Weak-form always uses the full-JIT path (derivative not needed externally)
        if _is_weak:
            pde_accepts_derivative = True

        # Precompute n_dims from first BC if available
        n_dims = self.problem.n_dims

        _is_rollout_batch = (_is_weak and
                              getattr(self, '_rollout_face_batch', None) is not None and
                              getattr(self.problem, 'n_time_steps', None) is not None)
        _is_random_time  = getattr(self, '_rts_t_min', None) is not None
        _is_node_batch   = getattr(self, '_rts_n_nodes', None) is not None
        _rts_sampling_method = getattr(self, '_rts_sampling_method', 'uniform')
        _weak_loss_fn_train = getattr(self, '_weak_loss_fn_train', _weak_loss_fn)

        def compute_loss(params, train_data, targets_dict, lm_params=None):
            total_loss = 0.0

            # ===== PDE / Weak-form Loss =====
            if _is_weak:
                # Weak-form: cubature assembly, no collocation points needed
                if _is_rollout_batch:
                    face_idx = train_data['_rollout_face_idx']
                    pde_loss = _weak_loss_fn_train(params, face_idx)
                elif _is_random_time:
                    if _is_node_batch:
                        pde_loss = _weak_loss_fn_train(params, train_data['_node_idx'], train_data['_t_vals'])
                    else:
                        pde_loss = _weak_loss_fn(params, train_data['_t_vals'])
                else:
                    pde_loss = _weak_loss_fn(params)
                total_loss = total_loss + pde_weight * pde_loss
            elif 'pde' in train_data:
                x_pde = train_data['pde']

                # Use sparse differentiable forward if available
                if precomputed_pde is not None:
                    # Sparse path with derivative support
                    def sparse_apply(p, x):
                        return model_apply_sparse_diff(p, x, precomputed_pde)
                    y_pde = sparse_apply(params, x_pde)
                    deriv_fn = make_derivative_fn(sparse_apply, params)
                else:
                    # Standard path
                    y_pde = model_apply_with_params(params, x_pde)
                    deriv_fn = make_derivative_fn(model_apply_with_params, params)

                if pde_accepts_derivative:
                    residual = pde_fn(x_pde, y_pde, params_dict, deriv_fn)
                else:
                    set_context(model_apply, params)
                    try:
                        residual = pde_fn(x_pde, y_pde, params_dict)
                    finally:
                        clear_context()

                if isinstance(residual, (list, tuple)):
                    pde_loss = sum(jnp.mean(r**2) for r in residual) / len(residual)
                else:
                    pde_loss = jnp.mean(residual**2)
                total_loss = total_loss + pde_weight * pde_loss
            
            # ===== Dirichlet BC Loss (batched forward pass) =====
            for bc_data in dirichlet_bcs:
                bc_name = bc_data['name']
                
                # Use sparse precomputed data if available
                if bc_name in precomputed_bcs:
                    y_bc = model_apply_sparse(params, precomputed_bcs[bc_name])
                else:
                    x_bc = train_data[bc_name]
                    y_bc = model_apply_with_params(params, x_bc)
                
                # Get target
                if bc_data.get('is_points'):
                    target = jnp.array(bc_data['u_data'], dtype=jnp.float32)
                elif bc_data['const_value'] is not None:
                    target = bc_data['const_value']
                elif bc_name in targets_dict:
                    target = targets_dict[bc_name]
                else:
                    target = 0.0
                
                bc_loss = jnp.mean((y_bc[:, bc_data['component']] - target) ** 2)
                total_loss = total_loss + bc_data['weight'] * bc_loss
            
            # ===== Neumann BC Loss (efficient batched JVP) =====
            for bc_data in neumann_bcs:
                x_bc = train_data[bc_data['name']]
                comp = bc_data['component']
                normal_dim = bc_data['normal_dim']
                normal_sign = bc_data['normal_sign']
                
                # Get target: const_value for scalar BCs, precomputed targets for callable BCs
                bc_name = bc_data['name']
                if bc_data['const_value'] is not None:
                    target = bc_data['const_value']
                elif bc_name in targets_dict:
                    target = targets_dict[bc_name]
                else:
                    target = 0.0
                
                # Efficient batched derivative: tangent vectors point along normal direction
                # Shape: (batch_size, n_dims) with 1.0 in normal_dim column
                tangent = jnp.zeros_like(x_bc)
                tangent = tangent.at[:, normal_dim].set(1.0)
                
                # Single batched JVP call instead of vmap over points
                def forward_component(x):
                    return model_apply_with_params(params, x)[:, comp]
                
                _, du_dn = jax.jvp(forward_component, (x_bc,), (tangent,))
                
                bc_loss = jnp.mean((normal_sign * du_dn - target) ** 2)
                total_loss = total_loss + bc_data['weight'] * bc_loss
            
            # ===== Mesh Neumann BC Loss (per-sample normals read from train_data) =====
            for bc_data in mesh_neumann_bcs:
                bc_name = bc_data['name']
                x_bc = train_data[bc_name]
                comp = bc_data['component']
                # Normals are stored alongside the BC points in train_data
                normals_rt = train_data.get(f'{bc_name}__normals', None)

                if bc_data['const_value'] is not None:
                    mn_target = bc_data['const_value']
                elif bc_name in targets_dict:
                    mn_target = targets_dict[bc_name]
                else:
                    mn_target = 0.0

                if normals_rt is not None:
                    # du/dn = Σ_i (∂u/∂x_i) * n_i  via JVP with per-sample tangents
                    def forward_mesh_comp(x):
                        return model_apply_with_params(params, x)[:, comp]

                    _, du_dn_mesh = jax.jvp(forward_mesh_comp, (x_bc,), (normals_rt,))
                    bc_loss = jnp.mean((du_dn_mesh - mn_target) ** 2)
                else:
                    y_bc = model_apply_with_params(params, x_bc)
                    bc_loss = jnp.mean((y_bc[:, comp] - mn_target) ** 2)

                total_loss = total_loss + bc_data['weight'] * bc_loss

            # ===== Custom Residual BC Loss (TermMeshCustomBC) =====
            if custom_bc_list:
                _deriv_fn = make_derivative_fn(model_apply_with_params, params)
                for _bc, _bc_name, _bc_weight, _n in custom_bc_list:
                    # Weak BCs (phi in signature) are handled by the Galerkin
                    # assembler — skip pointwise evaluation entirely.
                    if getattr(_bc, 'is_weak', False):
                        continue
                    _x_bc = train_data[_bc_name]
                    _y_bc = model_apply_with_params(params, _x_bc)
                    if _n >= 4:
                        _residual = _bc.f(_x_bc, _y_bc, params_dict, _deriv_fn)
                    elif _n == 3:
                        _residual = _bc.f(_x_bc, _y_bc, params_dict)
                    else:
                        _residual = _bc.f(_x_bc, _y_bc)
                    if isinstance(_residual, (list, tuple)):
                        _out_names = _bc.output_names or ([_bc_name] * len(_residual))
                        for _r, _oname in zip(_residual, _out_names):
                            _w = weights.get(_oname, _bc_weight)
                            total_loss = total_loss + _w * jnp.mean(_r ** 2)
                    else:
                        _bc_loss = jnp.mean(_residual ** 2)
                        total_loss = total_loss + _bc_weight * _bc_loss

            # ===== Periodic BC Loss =====
            for _pbc in periodic_bcs:
                _x_a  = _pbc['x_a']
                _x_b  = _pbc['x_b']
                _y_a  = model_apply_with_params(params, _x_a)
                _y_b  = model_apply_with_params(params, _x_b)
                _comp = _pbc['component']
                if _comp is not None:
                    _pbc_loss = jnp.mean((_y_a[:, _comp] - _y_b[:, _comp]) ** 2)
                else:
                    _pbc_loss = jnp.mean((_y_a - _y_b) ** 2)

                # Optional: also penalise u_x(a) - u_x(b)  (neumann-style periodicity)
                if _pbc['match_x_deriv']:
                    _cidx   = _comp if _comp is not None else 0
                    _dim    = _pbc.get('x_deriv_dim', 1)
                    _tang_a = jnp.zeros_like(_x_a).at[:, _dim].set(1.0)
                    _tang_b = jnp.zeros_like(_x_b).at[:, _dim].set(1.0)
                    def _fa(x): return model_apply_with_params(params, x)[:, _cidx]
                    def _fb(x): return model_apply_with_params(params, x)[:, _cidx]
                    _, _ux_a = jax.jvp(_fa, (_x_a,), (_tang_a,))
                    _, _ux_b = jax.jvp(_fb, (_x_b,), (_tang_b,))
                    _pbc_loss = _pbc_loss + jnp.mean((_ux_a - _ux_b) ** 2)

                total_loss = total_loss + _pbc['weight'] * _pbc_loss

            return total_loss

        if pde_accepts_derivative:
            @jax.jit
            def train_step(params, opt_state, train_data, targets_dict):
                loss, grads = jax.value_and_grad(compute_loss)(params, train_data, targets_dict)
                updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return new_params, new_opt_state, loss
            
            return train_step, True, False
        else:
            grad_fn = jax.value_and_grad(compute_loss)
            
            @jax.jit
            def apply_updates(params, grads, opt_state):
                updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return new_params, new_opt_state
            
            return (grad_fn, apply_updates), False, False
    
    # ==================== Solution error (rollout override) ====================

    def _compute_solution_error(self, n_points: int = 1000):
        """Override: for BPTT rollout mode unroll the full trajectory and compare."""
        from pinns.problems.problem_weak import ProblemWeak as _PW
        _is_rollout = (
            isinstance(self.problem, _PW)
            and getattr(self.problem.domain, '_time_mode', None) == 'discrete'
            and getattr(self.problem, 'n_time_steps', None) is not None
            and self.problem.solution is not None
            and hasattr(self.network, 'predict_rollout')
        )
        if not _is_rollout:
            return super()._compute_solution_error(n_points)

        import numpy as _np
        domain = self.problem.domain
        # Use the current curriculum stage, not the full domain horizon
        n_steps = self.problem.n_time_steps
        dt      = float(domain.dt)
        u_all = self.network.predict_rollout(n_steps=n_steps, dt=dt)
        # u_all: (n_steps+1, n_nodes)
        verts = _np.array(domain._vertices)  # (n_nodes, 2)
        t_vals = _np.array(domain._time_points[:n_steps + 1])  # only trained steps
        preds, trues = [], []
        for step_i, tv in enumerate(t_vals):
            x_ref = _np.hstack([verts, _np.full((len(verts), 1), tv)])
            y_true = self._call_solution(x_ref)  # (n_nodes, 1)
            if y_true is None:
                continue
            y_true_np = _np.atleast_2d(y_true).reshape(-1, 1) if y_true.ndim == 1 else y_true
            y_pred_np = u_all[step_i].reshape(-1, 1)
            valid = _np.isfinite(y_true_np).all(axis=1)
            if valid.any():
                preds.append(y_pred_np[valid])
                trues.append(y_true_np[valid])
        if not preds:
            return None
        y_pred_all = _np.concatenate(preds)
        y_true_all = _np.concatenate(trues)
        return float(_np.mean((y_pred_all - y_true_all) ** 2))

    # ==================== Plot snapshot (rollout override) ====================

    def _plot_mesh_snapshot(self, ax, output_idx, t_val, kind='sol'):
        """Override: for BPTT rollout, fetch the correct time-step from predict_rollout."""
        from pinns.problems.problem_weak import ProblemWeak as _PW
        import numpy as _np

        _is_rollout = (
            isinstance(self.problem, _PW)
            and getattr(self.problem.domain, '_time_mode', None) == 'discrete'
            and getattr(self.problem, 'n_time_steps', None) is not None
            and hasattr(self.network, 'predict_rollout')
        )
        if not _is_rollout or kind not in ('sol', 'err', 'true', 'res'):
            return super()._plot_mesh_snapshot(ax, output_idx, t_val, kind)

        domain = self.problem.domain
        # Always roll out the full domain horizon for plotting
        n_steps = domain.n_steps
        dt = float(domain.dt)
        t_points = _np.array(domain._time_points)  # (n_steps+1,)

        # run rollout over the full domain
        try:
            u_all = self.network.predict_rollout(n_steps=n_steps, dt=dt)  # (n_steps+1, n_nodes)
        except Exception:
            ax.text(0.5, 0.5, 'Rollout not ready', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_title(f't={t_val:.3g}')
            return

        # linearly interpolate between the two bracketing time steps
        t_val_f = float(t_val)
        idx_hi = int(_np.searchsorted(t_points, t_val_f, side='left'))
        idx_hi = int(_np.clip(idx_hi, 1, len(t_points) - 1))
        idx_lo = idx_hi - 1
        t_lo, t_hi = float(t_points[idx_lo]), float(t_points[idx_hi])
        alpha = (t_val_f - t_lo) / (t_hi - t_lo) if t_hi > t_lo else 0.0
        u_snap = (1.0 - alpha) * u_all[idx_lo] + alpha * u_all[idx_hi]
        t_actual = t_val_f

        import matplotlib.tri as _mtri
        verts_xy = _np.array(domain._vertices)
        faces = _np.array(domain._faces)
        tri_obj = _mtri.Triangulation(verts_xy[:, 0], verts_xy[:, 1], faces)

        if kind == 'sol':
            vals = u_snap.astype(float)
            cmap = 'viridis'
            label = f'u pred (t={t_actual:.3g})'
            vmin, vmax = 0.0, 1.0
            im = ax.tricontourf(tri_obj, vals, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
            cbar = self._fig.colorbar(im, ax=ax)
            self._colorbars.append(cbar)
            ax.set_aspect('equal')
            ax.set_title(label)
            ax.set_xlabel('x'); ax.set_ylabel('y')

        elif kind == 'err' and self.problem.solution is not None:
            x_ref = _np.hstack([verts_xy, _np.full((len(verts_xy), 1), t_actual, dtype=_np.float32)])
            y_true = self._call_solution(x_ref)
            if y_true is None:
                return
            y_true_np = _np.atleast_2d(y_true).reshape(-1) if y_true.ndim > 1 else y_true
            err = _np.abs(u_snap - y_true_np)
            err[~_np.isfinite(err)] = _np.nan
            vals_plot = _np.where(_np.isnan(err), 0.0, err)
            cmap = 'viridis'
            label = f'|error| (t={t_actual:.3g})'
            im = ax.tricontourf(tri_obj, vals_plot, levels=50, cmap=cmap)
            cbar = self._fig.colorbar(im, ax=ax)
            self._colorbars.append(cbar)
            ax.set_aspect('equal')
            ax.set_title(label)
            ax.set_xlabel('x'); ax.set_ylabel('y')

        elif kind == 'true' and self.problem.solution is not None:
            x_ref = _np.hstack([verts_xy, _np.full((len(verts_xy), 1), t_actual, dtype=_np.float32)])
            y_true = self._call_solution(x_ref)
            if y_true is None:
                return
            y_true_np = _np.atleast_2d(y_true).reshape(-1) if y_true.ndim > 1 else y_true
            vals = y_true_np.astype(float)
            vals_plot = _np.where(_np.isfinite(vals), vals, 0.0)
            nan_mask = _np.array([
                not (_np.isfinite(vals[f[0]]) and _np.isfinite(vals[f[1]]) and _np.isfinite(vals[f[2]]))
                for f in faces
            ])
            tri_obj.set_mask(nan_mask)
            im = ax.tricontourf(tri_obj, vals_plot, levels=50, cmap='viridis', vmin=0.0, vmax=1.0)
            cbar = self._fig.colorbar(im, ax=ax)
            self._colorbars.append(cbar)
            ax.set_aspect('equal')
            ax.set_title(f'True (t={t_actual:.3g})')
            ax.set_xlabel('x'); ax.set_ylabel('y')

        elif kind == 'res':
            # Accumulated |residual| over steps 1..k, where k corresponds to t_val.
            acc_res = self._compute_rollout_accumulated_residual(u_all)  # (n_steps+1, n_nodes)
            # Interpolate between bracketing steps (same logic as u_snap above)
            res_snap = (1.0 - alpha) * acc_res[idx_lo] + alpha * acc_res[idx_hi]
            vals_plot = res_snap.astype(float)
            label = f'Accum |R| (t={t_actual:.3g})'
            im = ax.tricontourf(tri_obj, vals_plot, levels=50, cmap='inferno')
            cbar = self._fig.colorbar(im, ax=ax)
            self._colorbars.append(cbar)
            ax.set_aspect('equal')
            ax.set_title(label)
            ax.set_xlabel('x'); ax.set_ylabel('y')

    def _compute_rollout_accumulated_residual(self, u_all):
        """Compute accumulated absolute per-node weak residual across rollout steps.

        Returns array of shape (n_steps+1, n_nodes) where entry [k] is the
        sum of |R_1| + |R_2| + ... + |R_k| (entry [0] is all zeros).
        """
        import numpy as _np
        cd   = self.problem.cubature_data
        phi  = _np.array(cd['phi'],      dtype=float)   # (F, Q, L)
        gph  = _np.array(cd['grad_phi'], dtype=float)   # (F, Q, L, 2)
        wts  = _np.array(cd['weights'],  dtype=float)   # (F, Q)
        nid  = _np.array(cd['node_ids'], dtype=int)     # (F, L)
        dt   = float(self.problem.domain.dt)
        kappa = float(self.problem.params.get('kappa', 1.0))

        F, Q, L = phi.shape
        n_nodes = u_all.shape[1]

        phi_f = phi.reshape(F * Q, L)           # (FQ, L)
        gph_f = gph.reshape(F * Q, L, 2)        # (FQ, L, 2)
        wts_f = wts.reshape(F * Q)              # (FQ,)
        # node id for each (quad-pt, basis-fn) pair
        nid_f = _np.tile(nid[:, None, :], (1, Q, 1)).reshape(F * Q, L)  # (FQ, L)

        acc = _np.zeros(n_nodes, dtype=float)
        result = [acc.copy()]  # step 0: zero residual (no step taken)

        for n in range(1, u_all.shape[0]):
            u_prev = u_all[n - 1].astype(float)  # (n_nodes,)
            u_next = u_all[n].astype(float)       # (n_nodes,)

            u_prev_cub = _np.sum(phi_f * u_prev[nid_f], axis=-1)   # (FQ,)
            u_next_cub = _np.sum(phi_f * u_next[nid_f], axis=-1)   # (FQ,)
            grad_u     = _np.einsum('kl,kli->ki', u_next[nid_f], gph_f)  # (FQ, 2)

            mass  = phi_f * ((u_next_cub - u_prev_cub) / dt)[:, None]  # (FQ, L)
            diff  = kappa * _np.einsum('ki,kli->kl', grad_u, gph_f)    # (FQ, L)
            contrib = (mass + diff) * wts_f[:, None]                    # (FQ, L)

            R = _np.zeros(n_nodes, dtype=float)
            _np.add.at(R, nid_f.reshape(-1), contrib.reshape(-1))

            acc = acc + _np.abs(R)
            result.append(acc.copy())

        return _np.stack(result, axis=0)  # (n_steps+1, n_nodes)

    # ==================== Loss Computation (weak-form override) ====================

    def _compute_total_loss(self, data, params_dict, weights_dict):
        """Override for ProblemWeak: strip 'pde' key and delegate to base class.

        The actual weak-form PDE loss is added by ``_compute_total_loss_batched``.
        This override only ensures the 'pde' collocation key (absent for weak
        problems) never reaches the base-class logic that would crash on it.
        Do NOT add PDE here — that would double-count it when called indirectly
        from the parent ``_compute_total_loss_batched`` → ``self._compute_total_loss``.
        """
        from pinns.problems.problem_weak import ProblemWeak as _ProblemWeak
        if isinstance(self.problem, _ProblemWeak) and hasattr(self, '_weak_loss_fn'):
            _inner_names_ct = {t['name'] for t in getattr(self.problem, '_inner_terms', []) or []} | {'pde'}
            bc_data = {k: v for k, v in data.items() if k not in _inner_names_ct}
            return super()._compute_total_loss(bc_data, params_dict, weights_dict)
        return super()._compute_total_loss(data, params_dict, weights_dict)

    def _compute_total_loss_batched(self, data, params_dict, weights_dict, batch_size=1000):
        """Override so the weak-form PDE loss is computed once, not per-batch."""
        import numpy as _np
        from pinns.problems.problem_weak import ProblemWeak as _ProblemWeak
        if isinstance(self.problem, _ProblemWeak) and hasattr(self, '_weak_loss_fn'):
            # Strip inner-term keys (handled by weak assembler) and internal '_'-prefixed keys.
            _inner_names_ctb = ({t['name'] for t in getattr(self.problem, '_inner_terms', []) or []}
                                | {'pde'})
            bc_data = {k: v for k, v in data.items()
                       if k not in _inner_names_ctb and not k.startswith('_')}
            if bc_data:
                total_loss, losses = super()._compute_total_loss_batched(
                    bc_data, params_dict, weights_dict, batch_size)
            else:
                total_loss, losses = 0.0, {}
            _is_train_data = data is getattr(self, '_train_data', None)
            pde_weight = weights_dict.get('pde', 1.0)
            # Per-term weights are already embedded inside make_loss_fn
            # (bc_weights=weights_dict), so the outer pde_weight factor must be 1.0
            # to avoid double-weighting.
            from pinns.problems.problem_weak import ProblemWeak as _PW_wl
            if isinstance(self.problem, _PW_wl):
                pde_weight = 1.0
            if _is_train_data:
                # Training: use precomputed _t_vals / _node_idx from train_data
                if getattr(self, '_rts_t_min', None) is not None:
                    if getattr(self, '_rts_n_nodes', None) is not None:
                        # Node-batched: one independent random time per node, shape (B,)
                        _node_idx = jnp.array(_np.random.choice(
                            self._rts_n_free_nodes, size=self._rts_n_nodes,
                            replace=self._rts_n_nodes > self._rts_n_free_nodes
                        ).astype(_np.int32))
                        _t_per_node = jnp.array(
                            (self._rts_t_min + _np.random.uniform(0, 1, self._rts_n_nodes)
                             * (self._rts_t_max - self._rts_t_min)).astype(_np.float32)
                        )
                        weak_pde_loss = float(self._weak_loss_fn_train(self.network.params, _node_idx, _t_per_node))
                    else:
                        _t_vals = jnp.array(
                            _np.linspace(self._rts_t_min, self._rts_t_max, self._rts_n_t)
                        )
                        weak_pde_loss = float(self._weak_loss_fn(self.network.params, _t_vals))
                else:
                    weak_pde_loss = float(self._weak_loss_fn(self.network.params))
                losses['pde'] = pde_weight * weak_pde_loss
                total_loss = (total_loss or 0.0) + pde_weight * weak_pde_loss
            elif getattr(self, '_weak_test_loss', False):
                # Test: sample a DIFFERENT random batch of nodes → held-out PDE metric
                _weak_fn_test = getattr(self, '_weak_loss_fn_test',
                                        getattr(self, '_weak_loss_fn_train',
                                                self._weak_loss_fn))
                if getattr(self, '_rts_t_min', None) is not None:
                    _n_free = getattr(self, '_rts_n_free_nodes', len(self.problem.free_nodes))
                    _n_test_nodes = getattr(self, '_rts_n_nodes_test', None) or getattr(self, '_rts_n_nodes', None)
                    if _n_test_nodes is not None:
                        # Node-batched test: one random time per test node, shape (B_test,)
                        _node_idx_test = jnp.array(_np.random.choice(
                            _n_free, size=_n_test_nodes,
                            replace=_n_test_nodes > _n_free
                        ).astype(_np.int32))
                        _t_per_node_test = jnp.array(
                            (self._rts_t_min + _np.random.uniform(0, 1, _n_test_nodes)
                             * (self._rts_t_max - self._rts_t_min)).astype(_np.float32)
                        )
                        weak_pde_loss_test = float(_weak_fn_test(self.network.params, _node_idx_test, _t_per_node_test))
                    else:
                        _t_vals_test = jnp.array(
                            _np.linspace(self._rts_t_min, self._rts_t_max, self._rts_n_t)
                        )
                        weak_pde_loss_test = float(_weak_fn_test(self.network.params, _t_vals_test))
                else:
                    weak_pde_loss_test = float(_weak_fn_test(self.network.params))
                losses['pde'] = pde_weight * weak_pde_loss_test
                total_loss = (total_loss or 0.0) + pde_weight * weak_pde_loss_test
            return total_loss, losses
        return super()._compute_total_loss_batched(data, params_dict, weights_dict, batch_size)

    # ==================== Training ====================

    def train(self):
        """
        Train the model.
        
        For full JIT compilation with JAX, define your PDE function with 4 arguments:
            def my_pde(X, U, params, derivative):
                u_x = derivative(U, X, 0, (0,))
                ...
        """
        from .schedulers.scheduler_lagrange import SchedulerLagrange as _SLag
        _lag = next((s for s in getattr(self, '_schedulers', []) if isinstance(s, _SLag)), None)
        if _lag is not None:
            _lag.run_training(self)
            return

        # ── Time-step curriculum (BPTT rollout mode only) ────────────────
        # If epochs_by_time_step is set, run progressive stages:
        #   stage 1 → unroll 1 step  for epochs_by_time_step epochs
        #   stage 2 → unroll 2 steps for epochs_by_time_step epochs
        #   …
        #   stage N → unroll N_TIME steps for epochs_by_time_step epochs
        # Each stage re-JITs the rollout loss fn for the new scan length.
        _ebt = getattr(self, '_epochs_by_time_step', None)
        _n_time_max = getattr(self.problem, 'n_time_steps', None)
        _in_curriculum = getattr(self, '_curriculum_running', False)
        if _ebt is not None and _n_time_max is not None and not _in_curriculum:
            self._curriculum_running = True
            self._curriculum_total_epochs = _n_time_max * _ebt   # for display
            saved_epochs = self._epochs
            saved_n_time = self.problem.n_time_steps
            try:
                for _stage_steps in range(1, _n_time_max + 1):
                    # Temporarily shrink n_time_steps so _make_jit_train_step
                    # compiles the scan for exactly _stage_steps steps.
                    self.problem.n_time_steps = _stage_steps
                    self._epochs = _ebt
                    # Expose stage info so print messages can show "Stage X/Y"
                    self._curriculum_stage   = _stage_steps
                    self._curriculum_n_stages = _n_time_max
                    # Reset optimizer state at each stage: the new rollout length
                    # changes the loss landscape significantly, so stale SOAP/Adam
                    # second-order statistics from the previous (shorter) stage
                    # would cause overshooting on the first few updates.
                    self.opt_state = self.optimizer.init(self.network.params)
                    self.train()   # recursive call; _curriculum_running guards re-entry
            finally:
                self._curriculum_running = False
                self._epochs = saved_epochs
                self.problem.n_time_steps = saved_n_time
                # Rebuild loss fn for the full rollout
                _swe_final = getattr(self, '_step_weight_exp', 0.0)
                if _swe_final != 0.0:
                    import numpy as _np_sw2
                    _sw_f = _np_sw2.exp(_swe_final * _np_sw2.arange(_n_time_max) / max(_n_time_max - 1, 1))
                    _step_weights_final = _sw_f.astype(_np_sw2.float32)
                else:
                    _step_weights_final = None
                self._weak_loss_fn = jax.jit(
                    self.problem.make_rollout_loss_fn(self.network, n_steps=_n_time_max, step_weights=_step_weights_final)
                )
                # Rebuild AL fn for the full rollout if AL mode is active
                if getattr(self, '_rollout_al_mode', False):
                    _n_free = len(self.problem.free_nodes)
                    _al_fn  = jax.jit(self.problem.make_rollout_al_loss_fn(self.network, n_steps=_n_time_max))
                    self._rollout_al_fn = _al_fn
                    # Reset lambdas for full-rollout phase (same rationale as per-stage reset)
                    self._rollout_lambdas = jnp.zeros((_n_time_max, _n_free), dtype=jnp.float32)
                    _optim = self.optimizer
                    @jax.jit
                    def _rollout_al_train_step_full(params, opt_state, lambdas):
                        def _fl(p):
                            loss, res = _al_fn(p, lambdas)
                            return loss, res
                        (loss, res), grads = jax.value_and_grad(_fl, has_aux=True)(params)
                        updates, new_opt_state = _optim.update(grads, opt_state, params)
                        new_params = optax.apply_updates(params, updates)
                        return new_params, new_opt_state, loss, res
                    self._rollout_al_train_step = _rollout_al_train_step_full
            return

        epochs = self._epochs
        print_each = self._print_each
        show_plots = self._show_plots
        save_plots = self._save_plots
        
        params_dict = self._build_params()
        weights = self._list_to_dict_weights(self.weights)
        
        # L-BFGS uses different training loop
        if self.optimizer_name == "lbfgs":
            self._train_lbfgs(epochs, print_each, show_plots, save_plots, 
                             params_dict, weights)
            return
        
        result, is_full_jit, _ = self._make_jit_train_step(weights, params_dict)
        
        # Calculate number of batches
        n_batches = self._get_n_batches()
        use_batching = n_batches > 1
        
        _in_cur = getattr(self, '_curriculum_running', False)
        if is_full_jit:
            train_step = result
            if not _in_cur:
                if use_batching:
                    print(f"Starting training for {epochs} epochs, {n_batches} batches/epoch (JIT-compiled)...")
                else:
                    print(f"Starting training for {epochs} epochs (JIT-compiled)...")
        else:
            grad_fn, apply_updates = result
            if not _in_cur:
                if use_batching:
                    print(f"Starting training for {epochs} epochs, {n_batches} batches/epoch...")
                else:
                    print(f"Starting training for {epochs} epochs...")
                    print("Note: For faster training, define PDE with 4th 'derivative' argument")
        
        start_time = time.time()
        start_epoch = self._global_epoch
        
        # Auto-save path for script mode (non-interactive file saving)
        auto_save_path = None
        if show_plots and not save_plots and not is_notebook():
            import glob
            import os
            existing = glob.glob('./pinn_progress_*.png')
            if existing:
                nums = []
                for f in existing:
                    base = os.path.basename(f)
                    parts = base.replace('pinn_progress_', '').replace('.png', '').split('_')
                    try:
                        nums.append(int(parts[0]))
                    except ValueError:
                        pass
                next_num = max(nums) + 1 if nums else 0
            else:
                next_num = 0
            auto_save_path = f'./pinn_progress_{next_num}.png'
        
        if show_plots:
            n_zoom_regions = len(getattr(self, '_plot_regions', []))
            needs_recreation = self._fig is None
            if not needs_recreation and self._axes is not None and n_zoom_regions > 0:
                if f'zoom_0_0' not in self._axes:
                    needs_recreation = True
            if needs_recreation:
                self._fig, self._axes = self._create_figure()
            
            # Display initial plot before training starts
            _, _, self._display_handle = self.plot_progress(
                save_path=None, n_points=self._plot_n_points,
                fig=self._fig, axes=self._axes, 
                display_handle=self._display_handle
            )
        
        # Initialize RNG key for shuffling
        shuffle_key = jax.random.PRNGKey(self.rng.integers(0, 2**31))

        # Learning rate scheduler
        lr_scheduler = getattr(self, '_lr_scheduler', None)
        
        # Print epoch 0 (before any training)
        if print_each > 0:
            metrics_batch_size = self._batch_size if self._batch_size and self._batch_size > 0 else 1000
            params_dict = self._build_params({'global_step': start_epoch, 'step': 0})
            full_train_loss, individual_losses = self._compute_total_loss_batched(
                self._train_data, params_dict, weights, batch_size=metrics_batch_size
            )
            pde_loss = float(individual_losses.get('pde', 0.0))
            bc_names = self._get_soft_bc_names()
            bc_losses_str = ", ".join(
                f"{name}: {individual_losses.get(name, 0.0):.2e}" 
                for name in bc_names
            )
            
            self.history['epoch'].append(start_epoch)
            self.history['train_loss'].append(float(full_train_loss))
            self.history['loss'].append(float(full_train_loss))
            self.history['loss_pde'].append(pde_loss)
            bc_losses = [float(individual_losses.get(name, 0.0)) for name in self._train_data.keys() if name != 'pde' and not name.startswith('_')]
            self.history['loss_bcs'].append(bc_losses)
            
            if (any(v > 0 for v in (self.test_samples.values() if isinstance(self.test_samples, dict) else self.test_samples)) and self._test_data) or getattr(self, '_weak_test_loss', False):
                _tw_data = self._test_data if self._test_data else {}
                test_weights = {k: 1.0 for k in _tw_data.keys()}
                test_total, _ = self._compute_total_loss_batched(
                    _tw_data, params_dict, test_weights, batch_size=metrics_batch_size
                )
                self.history['test_loss'].append(float(test_total))
            
            if self.problem.solution is not None:
                sol_error = self._compute_solution_error()
                self.history['solution_error'].append(sol_error)
            
            _cur_stage  = getattr(self, '_curriculum_stage', None)
            _n_stages   = getattr(self, '_curriculum_n_stages', None)
            _tot_epochs = getattr(self, '_curriculum_total_epochs', None)
            _stage_pfx  = f"Stage {_cur_stage}/{_n_stages} | " if _cur_stage is not None else ""
            _ep_total   = _tot_epochs if _tot_epochs is not None else epochs
            msg = (_stage_pfx +
                   f"Epoch {start_epoch}/{_ep_total} | "
                   f"Loss: {full_train_loss:.2e} | "
                   f"MSE Loss: {full_train_loss:.2e} | "
                   f"PDE: {pde_loss:.2e} | "
                   f"BCs: [{bc_losses_str}] | "
                   f"Time: 0.0s")
            if self.history['test_loss']:
                msg += f" | Test Loss: {self.history['test_loss'][-1]:.2e}"
            if self.problem.solution is not None:
                msg += f" | Error: {self.history['solution_error'][-1]:.2e}"
            print(msg)
        
        for epoch in range(epochs):
            epoch_start = time.time()
            global_epoch = start_epoch + epoch

            # Scheduler on_epoch_start hooks (resample, adaptive, curriculum, etc.)
            for _s in getattr(self, '_schedulers', []):
                _s.on_epoch_start(self, epoch)

            # Update learning rate if scheduler is provided
            # Skip for SOAP (has its own LR handling) and L-BFGS
            if lr_scheduler is not None and self.optimizer_name not in ("lbfgs", "soap"):
                new_lr = lr_scheduler.lr(self.learning_rate, global_epoch)
                if hasattr(self.opt_state, 'hyperparams'):
                    new_hyperparams = dict(self.opt_state.hyperparams)
                    new_hyperparams['learning_rate'] = new_lr
                    self.opt_state = self.opt_state._replace(hyperparams=new_hyperparams)
            
            # ── Rollout face mini-batching: sample fresh indices each epoch ──────────
            _rfb = getattr(self, '_rollout_face_batch', None)
            _rnf = getattr(self, '_rollout_n_faces', None)
            if _rfb is not None and _rnf is not None:
                _face_idx = np.random.choice(_rnf, size=_rfb, replace=False).astype(np.int32)
                self._train_data['_rollout_face_idx'] = jnp.array(_face_idx)

            # ── Node mini-batching (train_samples={'pde': N}) ──────────────────────
            _rts_n_nodes = getattr(self, '_rts_n_nodes', None)
            if _rts_n_nodes is not None:
                _n_free = self._rts_n_free_nodes
                _node_idx = np.random.choice(_n_free, size=_rts_n_nodes, replace=_rts_n_nodes > _n_free).astype(np.int32)
                self._train_data['_node_idx'] = jnp.array(_node_idx)
                # One independent random time per sampled node
                _t_per_node = (self._rts_t_min + np.random.uniform(0, 1, _rts_n_nodes) * (self._rts_t_max - self._rts_t_min)).astype(np.float32)
                self._train_data['_t_vals'] = jnp.array(_t_per_node)

            # ── Random time sampling: draw fresh t_vals each epoch ────────────────
            _rts_t_min = getattr(self, '_rts_t_min', None)
            if _rts_t_min is not None and _rts_n_nodes is None:  # skip when node-batching (handled above)
                _rts_t_max    = self._rts_t_max
                _rts_n_t      = self._rts_n_t
                _rts_method   = getattr(self, '_rts_sampling_method', 'uniform')
                if callable(_rts_method):
                    _u = _rts_method(_rts_n_t, np.random.default_rng())
                elif _rts_method in ('latin_hypercube', 'lhs'):
                    _u = (np.arange(_rts_n_t) + np.random.uniform(0, 1, _rts_n_t)) / _rts_n_t
                elif _rts_method == 'sobol':
                    try:
                        from scipy.stats.qmc import Sobol as _Sobol
                        _u = _Sobol(d=1, scramble=True).random(_rts_n_t).ravel()
                    except Exception:
                        _u = np.random.uniform(0, 1, _rts_n_t)
                elif _rts_method == 'halton':
                    try:
                        from scipy.stats.qmc import Halton as _Halton
                        _u = _Halton(d=1, scramble=True).random(_rts_n_t).ravel()
                    except Exception:
                        _u = np.random.uniform(0, 1, _rts_n_t)
                else:  # "uniform" or any unknown string
                    _u = np.random.uniform(0, 1, _rts_n_t)
                _t_vals = (_rts_t_min + _u * (_rts_t_max - _rts_t_min)).astype(np.float32)
                self._train_data['_t_vals'] = jnp.array(_t_vals)

            if use_batching:
                # Shuffle data and targets at the start of each epoch
                shuffle_key, subkey = jax.random.split(shuffle_key)
                shuffled_train_data = {}
                shuffled_train_targets = {}
                train_targets = getattr(self, '_train_targets', {})
                # Keys that should not be shuffled/batched (non-point arrays)
                _no_shuffle_keys = {'_rollout_face_idx', '_t_vals', '_node_idx'}
                for name, data in self._train_data.items():
                    if name in _no_shuffle_keys:
                        shuffled_train_data[name] = data  # pass through as-is
                        continue
                    n_points = len(data)
                    perm = jax.random.permutation(subkey, n_points)
                    shuffled_train_data[name] = data[perm]
                    # Also shuffle targets with the same permutation
                    if name in train_targets:
                        shuffled_train_targets[name] = train_targets[name][perm]
                    subkey, _ = jax.random.split(subkey)  # New key for next data array
                
                # Mini-batch training
                epoch_loss = 0.0
                for batch_idx in range(n_batches):
                    # Create batch data
                    batch_data = {}
                    batch_targets = {}
                    for name, data in shuffled_train_data.items():
                        if name in _no_shuffle_keys:
                            batch_data[name] = data  # same face indices for every mini-batch
                            continue
                        n_points = len(data)
                        start_idx, end_idx = self._get_batch_indices(n_points, batch_idx, n_batches)
                        batch_data[name] = data[start_idx:end_idx]
                        if name in shuffled_train_targets:
                            batch_targets[name] = shuffled_train_targets[name][start_idx:end_idx]
                    
                    if is_full_jit:
                        self.network.params, self.opt_state, loss = train_step(
                                self.network.params, self.opt_state, batch_data, batch_targets
                            )
                    else:
                        loss, grads = grad_fn(self.network.params, batch_data, batch_targets)
                        self.network.params, self.opt_state = apply_updates(self.network.params, grads, self.opt_state)
                    
                    epoch_loss += float(loss)
                
                loss = epoch_loss / n_batches
            else:
                # Full-batch training
                train_targets = getattr(self, '_train_targets', {})
                if is_full_jit:
                    # ── Per-step Lagrangian rollout (AL dual-ascent) ──────────
                    _al_step_fn = getattr(self, '_rollout_al_train_step', None)
                    if _al_step_fn is not None:
                        self.network.params, self.opt_state, loss, _step_res = _al_step_fn(
                            self.network.params, self.opt_state, self._rollout_lambdas
                        )
                        # Dual ascent: λ ← λ + lr * R̂  (plain numpy, outside JIT)
                        # step_res are already normalised by node_norm inside the
                        # loss fn, so the update is scale-free w.r.t. mesh size.
                        self._rollout_lambdas = self._rollout_lambdas + self.lagrange_lr * _step_res
                    else:
                        self.network.params, self.opt_state, loss = train_step(
                                self.network.params, self.opt_state, self._train_data, train_targets
                            )
                else:
                    loss, grads = grad_fn(self.network.params, self._train_data, train_targets)
                    self.network.params, self.opt_state = apply_updates(self.network.params, grads, self.opt_state)
            
            epoch_time = time.time() - epoch_start
            self.history['epoch_times'].append(epoch_time)
            
            # Update ReduceLROnPlateau scheduler with current loss
            if lr_scheduler is not None and hasattr(lr_scheduler, 'step'):
                lr_scheduler.step(float(loss), global_epoch)
            
            if print_each > 0 and ((global_epoch + 1) % print_each == 0 or epoch == epochs - 1):
                elapsed = time.time() - start_time
                
                # Compute losses on FULL training data for fair metrics (batched to avoid OOM)
                metrics_batch_size = self._batch_size if self._batch_size and self._batch_size > 0 else 1000
                full_train_loss, individual_losses = self._compute_total_loss_batched(
                    self._train_data, params_dict, weights, batch_size=metrics_batch_size
                )
                pde_loss = float(individual_losses.get('pde', 0.0))
                bc_losses = [float(individual_losses.get(name, 0.0)) for name in self._train_data.keys() if name != 'pde' and not name.startswith('_')]
                
                self.history['epoch'].append(global_epoch)
                self.history['train_loss'].append(float(full_train_loss))
                self.history['loss'].append(float(full_train_loss))
                self.history['loss_pde'].append(pde_loss)
                self.history['loss_bcs'].append(bc_losses)
                
                # Test loss (if test data available)
                if (any(v > 0 for v in (self.test_samples.values() if isinstance(self.test_samples, dict) else self.test_samples)) and self._test_data) or getattr(self, '_weak_test_loss', False):
                    _tw_data = self._test_data if self._test_data else {}
                    test_weights = {k: 1.0 for k in _tw_data.keys()}
                    test_total, _ = self._compute_total_loss_batched(
                        _tw_data, params_dict, test_weights, batch_size=metrics_batch_size
                    )
                    self.history['test_loss'].append(float(test_total))
                
                if self.problem.solution is not None:
                    sol_error = self._compute_solution_error()
                    self.history['solution_error'].append(sol_error)
                
                bc_names = self._get_soft_bc_names()
                bc_losses_str = ", ".join(
                    f"{name}: {individual_losses.get(name, 0.0):.2e}" 
                    for name in bc_names
                )
                
                _cur_stage  = getattr(self, '_curriculum_stage', None)
                _n_stages   = getattr(self, '_curriculum_n_stages', None)
                _tot_epochs = getattr(self, '_curriculum_total_epochs', None)
                _stage_pfx  = f"Stage {_cur_stage}/{_n_stages} | " if _cur_stage is not None else ""
                _ep_total   = _tot_epochs if _tot_epochs is not None else epochs + start_epoch
                msg = (_stage_pfx +
                       f"Epoch {global_epoch + 1}/{_ep_total} | "
                       f"Loss: {full_train_loss:.2e} | "
                       f"MSE Loss: {full_train_loss:.2e} | "
                       f"PDE: {pde_loss:.2e} | "
                       f"BCs: [{bc_losses_str}] | "
                       f"Time: {elapsed:.1f}s")
                if self.history['test_loss']:
                    msg += f" | Test Loss: {self.history['test_loss'][-1]:.2e}"
                if self.problem.solution is not None:
                    msg += f" | Error: {self.history['solution_error'][-1]:.2e}"
                print(msg)

                # User-supplied periodic callback: plot_callback(epoch, trainer)
                _cb = getattr(self, '_plot_callback', None)
                if _cb is not None:
                    _cb(global_epoch + 1, self)

                if show_plots or save_plots:
                    if save_plots:
                        plot_path = f"{save_plots}_epoch{global_epoch:05d}.png"
                    elif auto_save_path:
                        plot_path = auto_save_path
                    else:
                        plot_path = None
                    _, _, self._display_handle = self.plot_progress(
                        save_path=plot_path, n_points=self._plot_n_points,
                        fig=self._fig, axes=self._axes, 
                        display_handle=self._display_handle
                    )
        
        self._global_epoch += epochs
        if not getattr(self, '_curriculum_running', False):
            print(f"Training complete in {time.time() - start_time:.1f}s")
            for _s in getattr(self, '_schedulers', []):
                _s.on_training_end(self)
        self._curriculum_restore()
        
        # Close figure to prevent duplicate display in notebooks
        if is_notebook() and show_plots and self._fig is not None:
            plt.close(self._fig)

    def _curriculum_restore(self):
        """Clear per-stage display attributes after a stage's train() call ends."""
        # Only clear when not inside an active curriculum loop.
        # The outer curriculum dispatcher will set _curriculum_stage for the next stage.
        if not getattr(self, '_curriculum_running', False):
            self._curriculum_stage         = None
            self._curriculum_n_stages      = None
            self._curriculum_total_epochs  = None

    def _train_lbfgs(self, epochs, print_each, show_plots, save_plots, 
                     params_dict, weights):
        """
        Train using L-BFGS optimizer via jaxopt.
        
        L-BFGS typically converges faster than gradient descent for PINN problems,
        especially in the later stages of training.
        """
        # Build the loss function for L-BFGS
        compute_loss = self._make_lbfgs_loss_fn(weights, params_dict)
        
        # Get L-BFGS parameters from compile()
        max_iter = getattr(self, '_lbfgs_max_iter', 5)
        history_size = getattr(self, '_lbfgs_history_size', 50)
        tolerance = getattr(self, '_lbfgs_tolerance', 1e-9)
        
        # Create L-BFGS solver
        solver = jaxopt.LBFGS(
            fun=compute_loss,
            maxiter=max_iter,
            history_size=history_size,
            tol=tolerance,
        )
        
        print(f"Starting L-BFGS training for {epochs} epochs "
              f"({max_iter} iterations per epoch, history={history_size})...")
        
        start_time = time.time()
        start_epoch = self._global_epoch
        
        # Auto-save path for script mode (non-interactive file saving)
        auto_save_path = None
        if show_plots and not save_plots and not is_notebook():
            import glob
            import os
            existing = glob.glob('./pinn_progress_*.png')
            if existing:
                nums = []
                for f in existing:
                    base = os.path.basename(f)
                    parts = base.replace('pinn_progress_', '').replace('.png', '').split('_')
                    try:
                        nums.append(int(parts[0]))
                    except ValueError:
                        pass
                next_num = max(nums) + 1 if nums else 0
            else:
                next_num = 0
            auto_save_path = f'./pinn_progress_{next_num}.png'
        
        if show_plots:
            n_zoom_regions = len(getattr(self, '_plot_regions', []))
            needs_recreation = self._fig is None
            if not needs_recreation and self._axes is not None and n_zoom_regions > 0:
                if f'zoom_0_0' not in self._axes:
                    needs_recreation = True
            if needs_recreation:
                self._fig, self._axes = self._create_figure()
            
            # Display initial plot before training starts
            _, _, self._display_handle = self.plot_progress(
                save_path=None, n_points=self._plot_n_points,
                fig=self._fig, axes=self._axes, 
                display_handle=self._display_handle
            )
        
        # Initialize solver state
        state = solver.init_state(self.network.params, self._train_data)
        
        # Print epoch 0 (before any training)
        if print_each > 0:
            metrics_batch_size = 1000
            full_train_loss, individual_losses = self._compute_total_loss_batched(
                self._train_data, params_dict, weights, batch_size=metrics_batch_size
            )
            pde_loss = float(individual_losses.get('pde', 0.0))
            bc_names = self._get_soft_bc_names()
            bc_losses_str = ", ".join(
                f"{name}: {individual_losses.get(name, 0.0):.2e}" 
                for name in bc_names
            )
            
            self.history['epoch'].append(start_epoch)
            self.history['train_loss'].append(float(full_train_loss))
            self.history['loss'].append(float(full_train_loss))
            self.history['loss_pde'].append(pde_loss)
            bc_losses = [float(individual_losses.get(name, 0.0)) for name in self._train_data.keys() if name != 'pde' and not name.startswith('_')]
            self.history['loss_bcs'].append(bc_losses)
            
            if (any(v > 0 for v in (self.test_samples.values() if isinstance(self.test_samples, dict) else self.test_samples)) and self._test_data) or getattr(self, '_weak_test_loss', False):
                _tw_data = self._test_data if self._test_data else {}
                test_weights = {k: 1.0 for k in _tw_data.keys()}
                test_total, _ = self._compute_total_loss_batched(
                    _tw_data, params_dict, test_weights, batch_size=metrics_batch_size
                )
                self.history['test_loss'].append(float(test_total))
            
            if self.problem.solution is not None:
                sol_error = self._compute_solution_error()
                self.history['solution_error'].append(sol_error)
            
            msg = (f"Epoch 0/{epochs + start_epoch} | "
                   f"Loss: {full_train_loss:.2e} | "
                   f"MSE Loss: {full_train_loss:.2e} | "
                   f"PDE: {pde_loss:.2e} | "
                   f"BCs: [{bc_losses_str}] | "
                   f"Time: 0.0s")
            if self.history['test_loss']:
                msg += f" | Test Loss: {self.history['test_loss'][-1]:.2e}"
            if self.problem.solution is not None:
                msg += f" | Error: {self.history['solution_error'][-1]:.2e}"
            print(msg)
        
        for epoch in range(epochs):
            epoch_start = time.time()
            global_epoch = start_epoch + epoch
            
            # Run L-BFGS step
            self.network.params, state = solver.update(
                self.network.params, state, self._train_data
            )
            loss = state.value
            
            epoch_time = time.time() - epoch_start
            self.history['epoch_times'].append(epoch_time)
            
            if print_each > 0 and ((global_epoch + 1) % print_each == 0 or epoch == epochs - 1):
                elapsed = time.time() - start_time
                
                # Use batched loss computation for individual losses (avoid OOM)
                metrics_batch_size = 1000  # L-BFGS uses full batch but metrics can be batched
                _, individual_losses = self._compute_total_loss_batched(
                    self._train_data, params_dict, weights, batch_size=metrics_batch_size
                )
                pde_loss = float(individual_losses.get('pde', 0.0))
                bc_losses = [float(individual_losses.get(name, 0.0)) for name in self._train_data.keys() if name != 'pde' and not name.startswith('_')]
                
                self.history['epoch'].append(global_epoch)
                self.history['train_loss'].append(float(loss))
                self.history['loss'].append(float(loss))
                self.history['loss_pde'].append(pde_loss)
                self.history['loss_bcs'].append(bc_losses)
                
                # Test loss
                if (any(v > 0 for v in (self.test_samples.values() if isinstance(self.test_samples, dict) else self.test_samples)) and self._test_data) or getattr(self, '_weak_test_loss', False):
                    _tw_data = self._test_data if self._test_data else {}
                    test_weights = {k: 1.0 for k in _tw_data.keys()}
                    test_total, _ = self._compute_total_loss_batched(
                        _tw_data, params_dict, test_weights, batch_size=metrics_batch_size
                    )
                    self.history['test_loss'].append(float(test_total))
                
                if self.problem.solution is not None:
                    sol_error = self._compute_solution_error()
                    self.history['solution_error'].append(sol_error)
                
                bc_names = self._get_soft_bc_names()
                bc_losses_str = ", ".join(
                    f"{name}: {individual_losses.get(name, 0.0):.2e}" 
                    for name in bc_names
                )
                
                msg = (f"Epoch {global_epoch + 1}/{epochs + start_epoch} | "
                       f"Loss: {loss:.2e} | "
                       f"MSE Loss: {loss:.2e} | "
                       f"PDE: {pde_loss:.2e} | "
                       f"BCs: [{bc_losses_str}] | "
                       f"Time: {elapsed:.1f}s")
                if self.history['test_loss']:
                    msg += f" | Test Loss: {self.history['test_loss'][-1]:.2e}"
                if self.problem.solution is not None:
                    msg += f" | Error: {self.history['solution_error'][-1]:.2e}"
                print(msg)
                
                if show_plots or save_plots:
                    if save_plots:
                        plot_path = f"{save_plots}_epoch{global_epoch:05d}.png"
                    elif auto_save_path:
                        plot_path = auto_save_path
                    else:
                        plot_path = None
                    _, _, self._display_handle = self.plot_progress(
                        save_path=plot_path, n_points=self._plot_n_points,
                        fig=self._fig, axes=self._axes, 
                        display_handle=self._display_handle
                    )
        
        self._global_epoch += epochs
        print(f"L-BFGS training complete in {time.time() - start_time:.1f}s")
        
        # Close figure to prevent duplicate display in notebooks
        if is_notebook() and show_plots and self._fig is not None:
            plt.close(self._fig)
    
    def _make_lbfgs_loss_fn(self, weights, params_dict):
        """Create a loss function suitable for L-BFGS optimization."""
        # Similar to _make_jit_train_step but returns only loss function
        # FBPINN removed - use PartitionFB instead
        
        # Pre-extract BC info
        dirichlet_bcs = []
        neumann_bcs = []
        mesh_neumann_bcs = []   # TermMeshNodeBC with bc_type="neumann"
        bc_names = self._get_soft_bc_names()

        from pinns.problems.problem_weak import ProblemWeak as _ProblemWeak
        _hard_bc_names_lbfgs = set()
        
        # Get precomputed targets for callable BCs
        train_targets = getattr(self, '_train_targets', {})
        
        from pinns.terms import TermPeriodicBC as _PBC5
        _periodic_types5 = (_PBC5,)
        _bc_name_idx5 = 0
        for i, bc in enumerate(getattr(self.problem, 'boundary_conditions', [])):
            from pinns.terms import TermDirichletBC, TermNeumannBC, TermRobinBC
            if _periodic_types5 and isinstance(bc, _periodic_types5):
                continue
            name = bc_names[_bc_name_idx5]
            _bc_name_idx5 += 1

            # Hard-constrained BCs (ProblemWeak + output_transform) → skip soft loss
            if name in _hard_bc_names_lbfgs:
                continue

            is_neumann = isinstance(bc, (TermNeumannBC, TermRobinBC))
            if is_neumann:
                _ninfo = self.problem.domain.get_face_normal_direction(
                    getattr(bc, 'region', '')) or (0, 1)
            else:
                _ninfo = (0, 1)
            bc_data = {
                'name': name,
                'component': bc.component,
                'weight': weights.get(name, 1.0),
                'const_value': bc.value if not callable(bc.value) else None,
                'has_callable_value': callable(bc.value),
                'dim': _ninfo[0],
            }
            if is_neumann:
                bc_data['normal_sign'] = float(_ninfo[1])
                if isinstance(bc, TermNeumannBC):
                    neumann_bcs.append(bc_data)
            else:
                dirichlet_bcs.append(bc_data)

        # ── ProblemStrong: build L-BFGS loss fn inline ────────────────────
        from pinns.problems.problem_strong import ProblemStrong as _ProblemStrongLBFGS
        if isinstance(self.problem, _ProblemStrongLBFGS):
            _terms_lbfgs = list(self.problem._terms)
            _params_dict_l = params_dict
            _weights_l = weights

            def _model_apply_l(params, x):
                return self.network.apply(params, x, _params_dict_l)

            def compute_loss_strong_lbfgs(params, train_data):
                total_loss = jnp.array(0.0)
                deriv_fn = make_derivative_fn(_model_apply_l, params)
                for term in _terms_lbfgs:
                    if term.name not in train_data:
                        continue
                    x = train_data[term.name]
                    u = _model_apply_l(params, x)
                    if term.kind == 'points':
                        output_col = term.output_idx if term.output_idx is not None else 0
                        target = jnp.array(term.u_data, dtype=jnp.float32).flatten()
                        residual = u[:, output_col] - target
                    elif term.fn is not None and callable(term.fn):
                        residual = term.fn(x, u, _params_dict_l, deriv_fn)
                        if (term.eq_idx is not None
                                and hasattr(residual, 'ndim') and residual.ndim == 2):
                            residual = residual[:, term.eq_idx]
                    elif term.fn is not None:
                        output_col = term.output_idx if term.output_idx is not None else 0
                        residual = u[:, output_col:output_col + 1] - float(term.fn)
                    else:
                        continue
                    loss = jnp.mean(residual ** 2)
                    w = _weights_l.get(term.name, 1.0)
                    total_loss = total_loss + w * loss
                return total_loss

            return compute_loss_strong_lbfgs

        pde_fn = self.problem.pde_fn
        pde_weight = weights.get('pde', 1.0)
        
        use_sparse = False
        use_sparse_pde = False
        precomputed_bcs = {}
        precomputed_pde = None
        network = self.network

        sig = inspect.signature(pde_fn)
        pde_accepts_derivative = len(sig.parameters) >= 4
        
        n_dims = self.problem.n_dims
        
        def model_apply_with_params(params, x):
            return network.apply(params, x, params_dict)
        
        def model_apply_sparse(params, precomputed):
            return network.apply_precomputed_jit(params, precomputed, params_dict)
        
        def model_apply_sparse_diff(params, x, sparse_indices):
            return network.apply_sparse_differentiable(params, x, sparse_indices, params_dict)
        
        def compute_loss(params, train_data):
            total_loss = 0.0
            
            # PDE Loss
            if 'pde' in train_data:
                x_pde = train_data['pde']
                
                if precomputed_pde is not None:
                    def sparse_apply(p, x):
                        return model_apply_sparse_diff(p, x, precomputed_pde)
                    y_pde = sparse_apply(params, x_pde)
                    deriv_fn = make_derivative_fn(sparse_apply, params)
                else:
                    y_pde = model_apply_with_params(params, x_pde)
                    deriv_fn = make_derivative_fn(model_apply_with_params, params)
                
                if pde_accepts_derivative:
                    residual = pde_fn(x_pde, y_pde, params_dict, deriv_fn)
                else:
                    set_context(network.apply, params)
                    try:
                        residual = pde_fn(x_pde, y_pde, params_dict)
                    finally:
                        clear_context()
                
                if isinstance(residual, (list, tuple)):
                    pde_loss = sum(jnp.mean(r**2) for r in residual) / len(residual)
                else:
                    pde_loss = jnp.mean(residual**2)
                total_loss = total_loss + pde_weight * pde_loss
            
            # Dirichlet BC Loss
            for bc_data in dirichlet_bcs:
                bc_name = bc_data['name']
                
                if bc_name in precomputed_bcs:
                    y_bc = model_apply_sparse(params, precomputed_bcs[bc_name])
                else:
                    x_bc = train_data[bc_name]
                    y_bc = model_apply_with_params(params, x_bc)
                
                # Get target
                if bc_data.get('is_points'):
                    target = jnp.array(bc_data['u_data'], dtype=jnp.float32)
                elif bc_data['const_value'] is not None:
                    target = bc_data['const_value']
                elif bc_name in train_targets:
                    target = train_targets[bc_name]
                else:
                    target = 0.0
                
                bc_loss = jnp.mean((y_bc[:, bc_data['component']] - target) ** 2)
                total_loss = total_loss + bc_data['weight'] * bc_loss
            
            # Neumann BC Loss
            for bc_data in neumann_bcs:
                bc_name = bc_data['name']
                x_bc = train_data[bc_name]
                dim = bc_data['dim']
                component = bc_data['component']
                normal_sign = bc_data['normal_sign']
                
                # Get target: const_value for scalar BCs, precomputed targets for callable BCs
                if bc_data['const_value'] is not None:
                    target = bc_data['const_value']
                elif bc_name in train_targets:
                    target = train_targets[bc_name]
                else:
                    target = 0.0
                
                def forward_component(x):
                    y = model_apply_with_params(params, x)
                    return y[:, component]
                
                tangent = jnp.zeros_like(x_bc)
                tangent = tangent.at[:, dim].set(1.0)
                
                _, du_dn = jax.jvp(forward_component, (x_bc,), (tangent,))
                
                bc_loss = jnp.mean((normal_sign * du_dn - target) ** 2)
                total_loss = total_loss + bc_data['weight'] * bc_loss
            
            # ===== Mesh Neumann BC Loss =====
            for bc_data in mesh_neumann_bcs:
                bc_name = bc_data['name']
                x_bc = train_data[bc_name]
                comp = bc_data['component']
                normals_rt = train_data.get(f'{bc_name}__normals', None)

                if bc_data['const_value'] is not None:
                    mn_target = bc_data['const_value']
                elif bc_name in train_targets:
                    mn_target = train_targets[bc_name]
                else:
                    mn_target = 0.0

                if normals_rt is not None:
                    def forward_mesh_comp_lbfgs(x):
                        return model_apply_with_params(params, x)[:, comp]
                    _, du_dn_mesh = jax.jvp(forward_mesh_comp_lbfgs, (x_bc,), (normals_rt,))
                    bc_loss = jnp.mean((du_dn_mesh - mn_target) ** 2)
                else:
                    y_bc = model_apply_with_params(params, x_bc)
                    bc_loss = jnp.mean((y_bc[:, comp] - mn_target) ** 2)
                total_loss = total_loss + bc_data['weight'] * bc_loss

            return total_loss
        
        return compute_loss

    def get_history(self) -> Dict:
        """Get training history."""
        return self.history


def _initialize_lagrange_multipliers_impl(self):
    self.lagrange_multipliers = {}
    self._lagrange_opt_states = {}
    if self._lagrange_optimizer_name == 'adam':
        self._lagrange_optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=self.lagrange_lr)
    elif self._lagrange_optimizer_name == 'sgd':
        self._lagrange_optimizer = optax.inject_hyperparams(optax.sgd)(learning_rate=self.lagrange_lr)
    else:
        self._lagrange_optimizer = None

    if 'pde' in self._train_data and ((self._lagrange_constraints is None) or ('pde' in self._lagrange_constraints)):
        n = len(self._train_data['pde'])
        self.lagrange_multipliers['pde'] = jnp.zeros(n)
        if self._lagrange_optimizer is not None:
            self._lagrange_opt_states['pde'] = self._lagrange_optimizer.init(self.lagrange_multipliers['pde'])

    # For ProblemWeak the PDE residual has one vector per inner term,
    # each of size n_free_nodes * n_outputs.
    from pinns.problems.problem_weak import ProblemWeak as _ProblemWeak
    if isinstance(self.problem, _ProblemWeak):
        _n_free_comp = self.problem.n_free_nodes * self.problem.n_outputs
        _weak_terms = (getattr(self.problem, '_inner_terms', None)
                       or [{'fn': None, 'name': 'pde'}])
        for _wt in _weak_terms:
            _wname = _wt['name']
            if ((self._lagrange_constraints is None)
                    or (_wname in self._lagrange_constraints)):
                self.lagrange_multipliers[_wname] = jnp.zeros(_n_free_comp)
                if self._lagrange_optimizer is not None:
                    self._lagrange_opt_states[_wname] = self._lagrange_optimizer.init(
                        self.lagrange_multipliers[_wname])

    for name in self._get_bc_names():
        if name in self._train_data and ((self._lagrange_constraints is None) or (name in self._lagrange_constraints)):
            n = len(self._train_data[name])
            self.lagrange_multipliers[name] = jnp.zeros(n)
            if self._lagrange_optimizer is not None:
                self._lagrange_opt_states[name] = self._lagrange_optimizer.init(self.lagrange_multipliers[name])

    # Initialize Lagrange multipliers for TermPeriodicBC entries
    from pinns.terms import TermPeriodicBC as _PBC_L
    _periodic_types_L = (_PBC_L,)
    _n_out_L = len(self.problem.output_names) if (hasattr(self.problem, 'output_names') and self.problem.output_names) else getattr(self.problem, 'n_outputs', 1)
    for bc in getattr(getattr(self.problem, 'domain', None), 'boundary_conditions', []):
        if not (_periodic_types_L and isinstance(bc, _periodic_types_L)):
            continue
        _comps_L = [bc.component] if bc.component is not None else list(range(_n_out_L))
        for _i_L in _comps_L:
            _sub_name_L = bc.name if bc.component is not None else f'{bc.name}_{_i_L}'
            if (self._lagrange_constraints is None) or (_sub_name_L in self._lagrange_constraints):
                _n_pairs_L = bc.n_pairs if hasattr(bc, 'n_pairs') else len(bc.node_positions_a)
                _n_res_L = _n_pairs_L * 2 if getattr(bc, 'match_x_derivative', False) else _n_pairs_L
                self.lagrange_multipliers[_sub_name_L] = jnp.zeros(_n_res_L)
                if self._lagrange_optimizer is not None:
                    self._lagrange_opt_states[_sub_name_L] = self._lagrange_optimizer.init(self.lagrange_multipliers[_sub_name_L])


def _reinitialize_lagrange_if_needed_impl(self):
    for name, data in self._train_data.items():
        if name in self.lagrange_multipliers and len(self.lagrange_multipliers[name]) != len(data):
            self.lagrange_multipliers[name] = jnp.zeros(len(data))
            if self._lagrange_optimizer is not None:
                self._lagrange_opt_states[name] = self._lagrange_optimizer.init(self.lagrange_multipliers[name])


def _make_al_loss_fn_impl(self, params_dict):
    from pinns.terms import TermNeumannBC, TermRobinBC
    from pinns.problems.problem_weak import ProblemWeak as _ProblemWeak
    _is_weak = isinstance(self.problem, _ProblemWeak)

    # ── For ProblemWeak: replace the strong-form PDE residual (collocation)
    # with the FEM weak-form residual vector R[free_nodes].  BC residuals
    # remain identical to the strong form (point evaluation u − g).
    if _is_weak:
        network = self.network
        _n_out = self.problem.n_outputs

        if _n_out == 1:
            def _u_and_grad(p, xy):
                def _u(z): return network.apply(p, z[None])[0, 0]
                return jax.value_and_grad(_u)(xy)
        else:
            def _u_and_grad(p, xy):
                def _u_vec(z): return network.apply(p, z[None])[0]  # (n_out,)
                u = _u_vec(xy)
                jac = jax.jacobian(_u_vec)(xy)  # (n_out, n_dims)
                return u, jac

        _weak_res_fn = jax.jit(self.problem.make_residual_vectors_fn(_u_and_grad))
        # Store for the residual plot
        self._weak_residual_fn = _weak_res_fn
        self._u_and_grad_fn = _u_and_grad
        _n_dofs  = self.problem.n_dofs
        _n_comp  = self.problem.n_outputs
        # For multi-component: residual vector is [R1; R2; ...] of length n_dofs*n_comp
        # Free indices span all components
        # Collect inner term names for distinguishing PDE vs BC residuals
        _inner_term_names = set(t['name'] for t in getattr(self.problem, '_inner_terms', []) or [{'name': 'pde'}])
        _free_base = jnp.array(self.problem.free_nodes, dtype=jnp.int32)
        _free_nodes_jax = jnp.concatenate(
            [_free_base + k * _n_dofs for k in range(_n_comp)]
        ) if _n_comp > 1 else _free_base

        def _model_apply(p, x):
            return network.apply(p, x)

        train_targets = getattr(self, '_train_targets', {})
        _volume_name_weak = getattr(self.problem, '_volume_name', 'pde')
        bc_names = self._get_bc_names()

        # Collect Dirichlet BC info (same as strong form)
        _bc_point_info = []
        from pinns.terms import TermPeriodicBC as _PBC6
        _periodic_types6 = (_PBC6,)
        _bc_name_idx6 = 0
        for i, bc in enumerate(self.problem.boundary_conditions):
            if _periodic_types6 and isinstance(bc, _periodic_types6):
                continue
            name = bc_names[_bc_name_idx6]
            _bc_name_idx6 += 1
            _bc_point_info.append({
                    'name': name,
                    'component': bc.component,
                    'const_value': bc.value if not callable(bc.value) else None,
                })

        def compute_residuals_weak(params, train_data, targets_dict=None):
            targets_dict = {} if targets_dict is None else targets_dict
            residuals = {}
            # PDE: weak-form R_free per inner term (each shape n_free_nodes * n_comp)
            R_dict = _weak_res_fn(params)
            for _tname, R_full in R_dict.items():
                residuals[_tname] = R_full[_free_nodes_jax]
            # BCs: point evaluation  u(x_k) − g
            for info in _bc_point_info:
                bname = info['name']
                if bname not in train_data:
                    continue
                x_bc = train_data[bname]
                y_bc = _model_apply(params, x_bc)
                comp   = info['component']
                if info.get('is_points'):
                    target = info['u_data']
                else:
                    target = (info['const_value'] if info['const_value'] is not None
                              else targets_dict.get(bname, 0.0))
                residuals[bname] = (y_bc[:, comp] - target).flatten()
            return residuals

        def compute_al_loss_weak(params, train_data, lagrange_dict,
                                 weights_dict, targets_dict=None):
            residuals = compute_residuals_weak(params, train_data, targets_dict)
            total_loss = 0.0
            losses = {'bcs': []}
            lc = self._lagrange_constraints
            for name, g in residuals.items():
                lam = lagrange_dict.get(name, jnp.zeros_like(g))
                if len(lam) != len(g):
                    lam = jnp.zeros_like(g)
                use_quad   = self._constraint_uses_quadratic(name)
                use_lambda = (lc is None) or (name in lc)
                penalty    = weights_dict.get(name, 1.0) * jnp.mean(g ** 2) if use_quad else 0.0
                lagrangian = jnp.mean(jax.lax.stop_gradient(lam) * g) if use_lambda else 0.0
                constraint_loss = penalty + lagrangian
                losses[name] = constraint_loss
                losses[f'{name}_penalty']        = penalty
                losses[f'{name}_lagrangian']     = lagrangian
                losses[f'{name}_residual_mean']  = jnp.mean(jnp.abs(g))
                if name not in _inner_term_names:
                    losses['bcs'].append(constraint_loss)
                total_loss = total_loss + constraint_loss
            return total_loss, (losses, residuals)

        return compute_al_loss_weak, compute_residuals_weak

    bc_info = {}
    bc_names = self._get_bc_names()
    from pinns.terms import TermPeriodicBC as _PBC4
    _periodic_types4 = (_PBC4,)

    # ── Pre-sample TermCubicTermPeriodicBC pairs for use in compute_residuals ──────
    _periodic_al_entries = []  # list of dicts with x_a, x_b, name, component, dim, match_deriv
    _n_out_al = len(self.problem.output_names) if (hasattr(self.problem, 'output_names') and self.problem.output_names) else getattr(self.problem, 'n_outputs', 1)
    import numpy as _np_al
    for bc in getattr(getattr(self.problem, 'domain', None), 'boundary_conditions', []):
        if not (_periodic_types4 and isinstance(bc, _periodic_types4)):
            continue
        # New-style TermPeriodicBC: sample from region strings
        if hasattr(bc, 'node_positions_a'):
            _x_a_al = jnp.asarray(bc.node_positions_a, dtype=jnp.float32)
            _x_b_al = jnp.asarray(bc.node_positions_b, dtype=jnp.float32)
        else:
            _rng_al = _np_al.random.default_rng()
            _n_al = bc.n_pairs or 200
            _x_a_al = jnp.asarray(self.problem.domain.sample_boundary(_n_al, region=bc.region_a, rng=_rng_al), dtype=jnp.float32)
            _x_b_al = jnp.asarray(self.problem.domain.sample_boundary(_n_al, region=bc.region_b, rng=_rng_al), dtype=jnp.float32)
        _dim_al = 1
        _comps_al = [bc.component] if bc.component is not None else list(range(_n_out_al))
        for _i_al in _comps_al:
            _sub_name_al = bc.name if bc.component is not None else f'{bc.name}_{_i_al}'
            _periodic_al_entries.append({
                'name':        _sub_name_al,
                'x_a':         _x_a_al,
                'x_b':         _x_b_al,
                'component':   _i_al,
                'dim':         _dim_al,
                'match_deriv': getattr(bc, 'match_x_derivative', False),
            })
    name_idx = 0
    for i, bc in enumerate(getattr(self.problem, 'boundary_conditions', [])):
        if _periodic_types4 and isinstance(bc, _periodic_types4):
            continue
        name = bc_names[name_idx]
        name_idx += 1
        is_neumann = isinstance(bc, (TermNeumannBC, TermRobinBC))
        bc_info[name] = {
            'component': bc.component,
            'is_neumann': is_neumann,
            'is_mesh_neumann': False,
            'is_mesh_time_neumann': False,
            'const_value': bc.value if not callable(bc.value) else None,
            'normal_dim': 0,
            'normal_sign': 1,
        }
        if is_neumann:
            _ninfo = self.problem.domain.get_face_normal_direction(
                getattr(bc, 'region', '')) or (0, 1)
            bc_info[name]['normal_dim'] = _ninfo[0]
            bc_info[name]['normal_sign'] = _ninfo[1]

    # ── ProblemStrong: AL mode implemented directly from _terms ──────────
    from pinns.problems.problem_strong import ProblemStrong as _ProblemStrongAL
    if isinstance(self.problem, _ProblemStrongAL):
        network = self.network
        _terms_al = list(self.problem._terms)

        def _model_apply_al(params, x):
            return network.apply(params, x, params_dict)

        def compute_residuals_strong_al(params, train_data, targets_dict=None):
            deriv_fn = make_derivative_fn(_model_apply_al, params)
            residuals = {}
            for term in _terms_al:
                if term.name not in train_data:
                    continue
                x = train_data[term.name]
                u = _model_apply_al(params, x)
                if term.kind == 'points':
                    output_col = term.output_idx if term.output_idx is not None else 0
                    target = jnp.array(term.u_data, dtype=jnp.float32).flatten()
                    r = (u[:, output_col] - target).flatten()
                elif term.fn is not None and callable(term.fn):
                    r_raw = term.fn(x, u, params_dict, deriv_fn)
                    if term.eq_idx is not None and hasattr(r_raw, 'ndim') and r_raw.ndim == 2:
                        r_raw = r_raw[:, term.eq_idx:term.eq_idx + 1]
                    r = r_raw.flatten()
                elif term.fn is not None:
                    output_col = term.output_idx if term.output_idx is not None else 0
                    r = (u[:, output_col:output_col + 1] - float(term.fn)).flatten()
                else:
                    continue
                residuals[term.name] = r
            return residuals

        def compute_al_loss_strong(params, train_data, lagrange_dict,
                                   weights_dict, targets_dict=None):
            residuals = compute_residuals_strong_al(params, train_data, targets_dict)
            total_loss = jnp.array(0.0)
            losses = {'bcs': []}
            lc = self._lagrange_constraints
            for name, g in residuals.items():
                lam = lagrange_dict.get(name, jnp.zeros_like(g))
                if len(lam) != len(g):
                    lam = jnp.zeros_like(g)
                use_quad   = self._constraint_uses_quadratic(name)
                use_lambda = (lc is None) or (name in lc)
                penalty    = weights_dict.get(name, 1.0) * jnp.mean(g ** 2) if use_quad else 0.0
                lagrangian = jnp.mean(jax.lax.stop_gradient(lam) * g) if use_lambda else 0.0
                constraint_loss = penalty + lagrangian
                losses[name] = constraint_loss
                losses[f'{name}_penalty']       = penalty
                losses[f'{name}_lagrangian']    = lagrangian
                losses[f'{name}_residual_mean'] = jnp.mean(jnp.abs(g))
                total_loss = total_loss + constraint_loss
            return total_loss, (losses, residuals)

        return compute_al_loss_strong, compute_residuals_strong_al

    pde_fn = self.problem.pde_fn
    network = self.network
    pde_accepts_derivative = len(inspect.signature(pde_fn).parameters) >= 4

    def model_apply_with_params(params, x):
        return network.apply(params, x, params_dict)

    def compute_residuals(params, train_data, targets_dict=None):
        targets_dict = {} if targets_dict is None else targets_dict
        residuals = {}
        if 'pde' in train_data:
            x_pde = train_data['pde']
            y_pde = model_apply_with_params(params, x_pde)
            deriv_fn = make_derivative_fn(model_apply_with_params, params)
            if pde_accepts_derivative:
                pde_residual = pde_fn(x_pde, y_pde, params_dict, deriv_fn)
            else:
                set_context(network.apply, params)
                try:
                    pde_residual = pde_fn(x_pde, y_pde, params_dict)
                finally:
                    clear_context()
            residuals['pde'] = sum(r.flatten() for r in pde_residual) if isinstance(pde_residual, (list, tuple)) else pde_residual.flatten()

        for name, info in bc_info.items():
            if name not in train_data:
                continue
            x_bc = train_data[name]
            y_bc = model_apply_with_params(params, x_bc)
            comp = info['component']
            target = info['const_value'] if info['const_value'] is not None else targets_dict.get(name, 0.0)
            if info['is_mesh_neumann']:
                # Edge-based Neumann: per-sample normals stored in train_data
                normals_rt = train_data.get(f'{name}__normals', None)
                def forward_mesh_comp(x):
                    return model_apply_with_params(params, x)[:, comp]
                if normals_rt is not None:
                    _, du_dn = jax.jvp(forward_mesh_comp, (x_bc,), (normals_rt,))
                else:
                    tangent = jnp.ones_like(x_bc) / jnp.sqrt(x_bc.shape[1])
                    _, du_dn = jax.jvp(forward_mesh_comp, (x_bc,), (tangent,))
                residuals[name] = (du_dn - target).flatten()
            elif info['is_mesh_time_neumann']:
                # Time-boundary Neumann: normal along time axis
                def forward_comp(x):
                    return model_apply_with_params(params, x)[:, comp]
                tangent = jnp.zeros_like(x_bc).at[:, info['normal_dim']].set(1.0)
                _, du_dn = jax.jvp(forward_comp, (x_bc,), (tangent,))
                residuals[name] = (info['normal_sign'] * du_dn - target).flatten()
            elif info['is_neumann']:
                def forward_component(x):
                    return model_apply_with_params(params, x)[:, comp]
                tangent = jnp.zeros_like(x_bc)
                tangent = tangent.at[:, info['normal_dim']].set(1.0)
                _, du_dn = jax.jvp(forward_component, (x_bc,), (tangent,))
                residuals[name] = (info['normal_sign'] * du_dn - target).flatten()
            else:
                residuals[name] = (y_bc[:, comp] - target).flatten()

        # ── Periodic BC residuals ───────────────────────────────────────────
        for _pe in _periodic_al_entries:
            _x_a_r = _pe['x_a']
            _x_b_r = _pe['x_b']
            _c_r   = _pe['component']
            _y_a_r = model_apply_with_params(params, _x_a_r)
            _y_b_r = model_apply_with_params(params, _x_b_r)
            _res_u = (_y_a_r[:, _c_r] - _y_b_r[:, _c_r])
            if _pe['match_deriv']:
                _d_r = _pe['dim']
                _tang_a_r = jnp.zeros_like(_x_a_r).at[:, _d_r].set(1.0)
                _tang_b_r = jnp.zeros_like(_x_b_r).at[:, _d_r].set(1.0)
                def _fa_r(x): return model_apply_with_params(params, x)[:, _c_r]
                def _fb_r(x): return model_apply_with_params(params, x)[:, _c_r]
                _, _ux_a_r = jax.jvp(_fa_r, (_x_a_r,), (_tang_a_r,))
                _, _ux_b_r = jax.jvp(_fb_r, (_x_b_r,), (_tang_b_r,))
                _res_ux = _ux_a_r - _ux_b_r
                residuals[_pe['name']] = jnp.concatenate([_res_u, _res_ux])
            else:
                residuals[_pe['name']] = _res_u
        return residuals

    def compute_al_loss(params, train_data, lagrange_dict, weights_dict, targets_dict=None):
        residuals = compute_residuals(params, train_data, targets_dict)
        total_loss = 0.0
        losses = {'bcs': []}
        lc = self._lagrange_constraints
        for name, g in residuals.items():
            lam = lagrange_dict.get(name, jnp.zeros_like(g))
            if len(lam) != len(g):
                lam = jnp.zeros_like(g)
            use_quad = self._constraint_uses_quadratic(name)
            use_lambda = (lc is None) or (name in lc)
            penalty = weights_dict.get(name, 1.0) * jnp.mean(g ** 2) if use_quad else 0.0
            lagrangian = jnp.mean(jax.lax.stop_gradient(lam) * g) if use_lambda else 0.0
            constraint_loss = penalty + lagrangian
            losses[name] = constraint_loss
            losses[f'{name}_penalty'] = penalty
            losses[f'{name}_lagrangian'] = lagrangian
            losses[f'{name}_residual_mean'] = jnp.mean(jnp.abs(g))
            if name != 'pde':
                losses['bcs'].append(constraint_loss)
            total_loss = total_loss + constraint_loss
        return total_loss, (losses, residuals)

    return compute_al_loss, compute_residuals


def _update_lagrange_multipliers_impl(self, residuals):
    lc = self._lagrange_constraints
    for name, g in residuals.items():
        if name not in self.lagrange_multipliers:
            continue
        if lc is not None and name not in lc:
            continue
        n_points = len(g)
        if self._lagrange_optimizer is not None:
            grad = -g / n_points
            updates, new_state = self._lagrange_optimizer.update(
                grad, self._lagrange_opt_states[name], self.lagrange_multipliers[name]
            )
            self.lagrange_multipliers[name] = optax.apply_updates(self.lagrange_multipliers[name], updates)
            self._lagrange_opt_states[name] = new_state
        else:
            self.lagrange_multipliers[name] = self.lagrange_multipliers[name] + self.lagrange_lr * g / n_points
        self.lagrange_multipliers[name] = jnp.clip(self.lagrange_multipliers[name], -self._lagrange_max, self._lagrange_max)


def _train_lagrangian_mode_impl(self):
    epochs = self._epochs
    print_each = self._print_each
    show_plots = self._show_plots
    save_plots = self._save_plots
    params_dict = self._build_params()
    weights_dict = self._list_to_dict_weights(self.weights)
    compute_al_loss, _ = self._make_al_loss_fn(params_dict)

    @jax.jit
    def train_step(params, opt_state, train_data, lagrange_dict, weights_dict, targets_dict):
        (loss, (losses, residuals)), grads = jax.value_and_grad(compute_al_loss, has_aux=True)(
            params, train_data, lagrange_dict, weights_dict, targets_dict
        )
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, losses, residuals

    start_time = time.time()
    start_epoch = self._global_epoch
    resample_each = getattr(self, '_resample_each', 0)
    adaptive_sampling = getattr(self, '_adaptive_sampling', False)
    adaptive_each = getattr(self, '_adaptive_each', 100)
    lr_scheduler = getattr(self, '_lr_scheduler', None)
    train_targets = getattr(self, '_train_targets', {})

    # Initialize live plot (mirrors standard training loop)
    if show_plots:
        needs_recreation = self._fig is None
        if needs_recreation:
            self._fig, self._axes = self._create_figure()
        _, _, self._display_handle = self.plot_progress(
            save_path=None, n_points=self._plot_n_points,
            fig=self._fig, axes=self._axes,
            display_handle=self._display_handle
        )

    print(f"Starting Trainer (JAX, Lagrangian mode) for {epochs} epochs...")

    # Print epoch 0 (before any training — mirrors standard training loop)
    if print_each > 0:
        _, compute_residuals = self._make_al_loss_fn(params_dict)
        residuals0 = compute_residuals(self.network.params, self._train_data, train_targets)
        bc_names = self._get_soft_bc_names()
        pde_mse0 = float(jnp.mean(residuals0['pde'] ** 2)) if 'pde' in residuals0 else 0.0
        bc_mse0 = [float(jnp.mean(residuals0[n] ** 2)) if n in residuals0 else 0.0 for n in bc_names]
        mse0 = float(sum(jnp.mean(g ** 2) for g in residuals0.values()))
        self.history['epoch'].append(start_epoch)
        self.history['loss'].append(mse0)
        self.history['train_loss'].append(mse0)
        self.history['loss_pde'].append([pde_mse0])
        self.history['loss_bcs'].append(bc_mse0)
        if (any(v > 0 for v in (self.test_samples.values() if isinstance(self.test_samples, dict) else self.test_samples)) and self._test_data) or getattr(self, '_weak_test_loss', False):
            metrics_batch_size = self._batch_size if self._batch_size and self._batch_size > 0 else 1000
            _tw_data0 = self._test_data if self._test_data else {}
            test_weights0 = {k: 1.0 for k in _tw_data0.keys()}
            test_total0, _ = self._compute_total_loss_batched(
                _tw_data0, params_dict, test_weights0, batch_size=metrics_batch_size
            )
            self.history['test_loss'].append(float(test_total0))
        bc_losses_str0 = ", ".join(f"{bc_names[i]}: {bc_mse0[i]:.2e}" for i in range(len(bc_names)))
        print(f"Epoch 0/{epochs} | MSE Loss: {mse0:.2e} | PDE: {pde_mse0:.2e} | BCs: [{bc_losses_str0}]")

    for epoch in range(start_epoch, start_epoch + epochs):
        self._outer_epoch = epoch - start_epoch

        # Time-domain curriculum: expand sampling window if stage changed
        self._curriculum_step(epoch)

        if lr_scheduler is not None and self.optimizer_name not in ("lbfgs", "soap") and hasattr(self.opt_state, 'hyperparams'):
            new_lr = lr_scheduler.lr(self.learning_rate, epoch)
            hp = dict(self.opt_state.hyperparams)
            hp['learning_rate'] = new_lr
            self.opt_state = self.opt_state._replace(hyperparams=hp)
            # Scale lagrange_lr by the same ratio
            self.lagrange_lr = self._lagrange_lr_ratio * new_lr
            if self._lagrange_optimizer is not None:
                for k in self._lagrange_opt_states:
                    if hasattr(self._lagrange_opt_states[k], 'hyperparams'):
                        lhp = dict(self._lagrange_opt_states[k].hyperparams)
                        lhp['learning_rate'] = self.lagrange_lr
                        self._lagrange_opt_states[k] = self._lagrange_opt_states[k]._replace(hyperparams=lhp)
        if resample_each > 0 and epoch > start_epoch and epoch % resample_each == 0:
            self._sample_train_data()
            self._reinitialize_lagrange_if_needed()
        if adaptive_sampling and epoch > start_epoch and epoch % adaptive_each == 0:
            self._adaptive_resample(params_dict)
            self._reinitialize_lagrange_if_needed()

        self.network.params, self.opt_state, loss, losses, residuals = train_step(
            self.network.params, self.opt_state, self._train_data, self.lagrange_multipliers, weights_dict, train_targets
        )
        self._update_lagrange_multipliers(residuals)

        if print_each > 0 and ((epoch + 1) % print_each == 0 or epoch == start_epoch + epochs - 1):
            al_loss_val = float(loss)
            mse_loss_val = float(sum(jnp.mean(g ** 2) for g in residuals.values()))
            bc_names = self._get_soft_bc_names()
            pde_mse = float(jnp.mean(residuals['pde'] ** 2)) if 'pde' in residuals else 0.0
            bc_mse_losses = [float(jnp.mean(residuals[name] ** 2)) if name in residuals else 0.0 for name in bc_names]
            self.history['epoch'].append(epoch)
            self.history['loss'].append(mse_loss_val)
            self.history['train_loss'].append(mse_loss_val)
            self.history.setdefault('al_loss', []).append(al_loss_val)
            self.history.setdefault('al_pde_penalty', []).append(float(losses.get('pde_penalty', 0.0)))
            self.history.setdefault('al_pde_lagrangian', []).append(float(losses.get('pde_lagrangian', 0.0)))
            self.history.setdefault('al_bcs_penalty', []).append([float(losses.get(f'{name}_penalty', 0.0)) for name in bc_names])
            self.history.setdefault('al_bcs_lagrangian', []).append([float(losses.get(f'{name}_lagrangian', 0.0)) for name in bc_names])
            self.history['loss_pde'].append([pde_mse])
            self.history['loss_bcs'].append(bc_mse_losses)
            if (any(v > 0 for v in (self.test_samples.values() if isinstance(self.test_samples, dict) else self.test_samples)) and self._test_data) or getattr(self, '_weak_test_loss', False):
                metrics_batch_size = self._batch_size if self._batch_size and self._batch_size > 0 else 1000
                _tw_data = self._test_data if self._test_data else {}
                test_weights = {k: 1.0 for k in _tw_data.keys()}
                test_total, _ = self._compute_total_loss_batched(
                    _tw_data, params_dict, test_weights, batch_size=metrics_batch_size
                )
                self.history['test_loss'].append(float(test_total))

            elapsed = time.time() - start_time
            if self.problem.solution is not None:
                sol_error = self._compute_solution_error()
                self.history['solution_error'].append(sol_error)
            bc_losses_str = ", ".join(f"{bc_names[i]}: {bc_mse_losses[i]:.2e}" for i in range(len(bc_names)))
            msg = (
                f"Epoch {epoch + 1}/{self._epochs + start_epoch} | AL Loss: {al_loss_val:.2e} | "
                f"MSE Loss: {mse_loss_val:.2e} | PDE: {pde_mse:.2e} | BCs: [{bc_losses_str}] | Time: {elapsed:.1f}s"
            )
            if self.problem.solution is not None:
                msg += f" | Error: {self.history['solution_error'][-1]:.2e}"
            print(msg)
            if show_plots:
                _, _, self._display_handle = self.plot_progress(
                    save_path=None, n_points=self._plot_n_points,
                    fig=self._fig, axes=self._axes,
                    display_handle=self._display_handle
                )

    self._global_epoch += epochs
    print(f"Trainer (Lagrangian mode) complete in {time.time() - start_time:.1f}s")
    self._curriculum_restore()
    if is_notebook() and show_plots and self._fig is not None:
        plt.close(self._fig)


def _get_lagrange_statistics_impl(self) -> Dict[str, Dict[str, float]]:
    return {
        name: {
            'mean': float(jnp.mean(lam)),
            'std': float(jnp.std(lam)),
            'min': float(jnp.min(lam)),
            'max': float(jnp.max(lam)),
        }
        for name, lam in self.lagrange_multipliers.items()
    }

def _reset_lagrange_multipliers_impl(self):
    for name in self.lagrange_multipliers:
        self.lagrange_multipliers[name] = jnp.zeros_like(self.lagrange_multipliers[name])


def _reset_betas_impl(self, betas: dict = None):
    if betas is None:
        for i in range(len(self.weights)):
            self.weights[i] = 1.0
    else:
        self.weights = self._convert_dict_to_list(betas, 'weights')
