"""
PINN Trainer (JAX/Optax).

Provides a JAX-based Trainer class for Physics-Informed Neural Networks.
LR schedulers live in pinns.schedulers.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from functools import partial
import time
import inspect
import threading

try:
    import jaxopt
    HAS_JAXOPT = True
except ImportError:
    HAS_JAXOPT = False

from .optimizers import build_optimizer, BaseOptimizer, AdamOptimizer
from .schedulers import Scheduler, is_notebook, SchedulerExponentialDecay, SchedulerReduceLROnPlateau
from ._plotting import TrainPlotter
from ..functional import set_context, clear_context, make_derivative_fn

class Trainer:
    """
    JAX-based PINN Trainer.
    
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
    
    def __init__(self, problem, network, dataset=None, device=None):
        """
        Initialize common trainer attributes.
        
        Args:
            problem: Problem instance defining PDE and boundary conditions.
            network: Neural network to train.
            dataset: Optional :class:`~pinns.Dataset` of fixed observation data.
                     Each registered point set is evaluated every training step
                     as a supervised MSE loss (no domain-based sampling).
            device: Device to use. If None, auto-detect using backend.
        """
        # Reset network parameters so fresh init always starts clean
        if hasattr(network, 'params'):
            network.params = None

        self.problem = problem
        self.dataset  = dataset
        
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
        if isinstance(problem, (_ProblemStrong, _ProblemWeak)):
            _is_weak = isinstance(problem, _ProblemWeak)
            self.train_samples = {
                t.name: (0 if (_is_weak and t.kind == 'inner') or t.kind == 'periodic'
                         else (1000 if t.kind == 'inner' else 10))
                for t in problem._terms
            }
            self.test_samples  = {t.name: 0 for t in problem._terms}
            self.weights       = {t.name: 1.0 for t in problem._terms}
        else:
            raise TypeError(
                f"Unsupported problem type: {type(problem).__name__}. "
                "Use ProblemStrong or ProblemWeak."
            )
        # Dataset: register weights so compile(weights={...}) can override them.
        # Do NOT add to train_samples — dataset points are injected directly in
        # _sample_train_data without going through _sample_points_np.
        if dataset is not None:
            for _dt in dataset._data_terms:
                self.weights[_dt.name] = 1.0

        self._optimizer_obj = AdamOptimizer()
        self.optimizer_name = self._optimizer_obj.name
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
        self._plotter: Optional['TrainPlotter'] = None
        self._batch_size = None
        self._compiled = False

        # Rollout AL mode (BPTT) — minimal state needed
        self.lagrange_lr = 1.0
        self._lagrange_lr_ratio = 1.0
    
    def _setup_network_normalization(self):
        """Set up input/output normalization on the network from problem definition."""
        xmin = np.array(self.problem.domain.xmin)
        xmax = np.array(self.problem.domain.xmax)
        
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

    @property
    def learning_rate(self) -> float:
        """Current learning rate, read from the active optimizer instance."""
        return self._optimizer_obj.learning_rate

    # ==================== Compile ====================
    
    def _compile_base(
        self,
        problem: Optional[Dict[str, Dict]] = None,
        optimizer: Optional[BaseOptimizer] = None,
        epochs: int = 1000,
        batch_size: int = None,
        print_each: int = 100,
        # Training schedulers (resample, adaptive, curriculum, lagrange, lr, …)
        schedulers: Optional[List] = None,
        # Plotting: pass TrainPlotter() to enable; None disables all plots.
        show: Optional[TrainPlotter] = TrainPlotter(),
        # Time-step curriculum for BPTT rollout mode
        # Train epochs_by_time_step epochs with 1 step, then 2 steps, …, up to n_time_steps.
        # Each stage runs epochs_by_time_step epochs.  Set to None to disable.
        epochs_by_time_step: Optional[int] = None,
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
            problem: Per-term configuration dict. Each key is a loss term name,
                each value is a dict with optional keys ``train`` (int),
                ``test`` (int), and ``weight`` (float or array).  Example::

                    problem={
                        "pde":    {"train": 1000, "test": 500,  "weight": 1.0},
                        "bottom": {"train": 100,  "test": 50,   "weight": 100.0},
                    }

            optimizer: Optimizer instance (e.g. ``AdamOptimizer(1e-3)``,
            epochs: Number of training epochs.
            batch_size: Batch size (if applicable).
            print_each: Print progress every N epochs.
            schedulers: List of Scheduler instances (SchedulerResample,
                SchedulerAdaptiveResample, SchedulerCurriculum, SchedulerLagrange,
                ExponentialDecay, ReduceLROnPlateau, etc.) to control training
                behaviour.  Pass ``Scheduler`` subclass instances here to
                schedule the learning rate.
            show: A :class:`~pinns.TrainPlotter` instance controlling all plot
                behaviour.  Pass ``TrainPlotter()`` to display with defaults, or
                ``TrainPlotter(save="./out", style={"theme": "dark"}, ...)`` for
                custom plots.  ``None`` (default) disables all plotting.
            epochs_by_time_step: Time-step curriculum for BPTT rollout mode.
                When set to an integer N, ``train()`` runs N epochs with 1 rollout
                step, then N epochs with 2 rollout steps, …, up to
                ``domain.n_steps`` rollout steps.  Each stage re-JITs the scan
                for the new length.  Set to ``None`` (default) to train on the
                full rollout from the start.
            grad_clip: Clip global gradient norm to this value.  ``None``
                (default) disables clipping.
            face_batch_size: Mini-batch over test functions (faces) for BPTT
                rollout.  ``None`` (default) uses all faces.
        """
        if problem is not None:
            for _term, _cfg in problem.items():
                if not isinstance(_cfg, dict):
                    raise ValueError(
                        f"compile(problem={{...}}) values must be dicts, "
                        f"got {type(_cfg).__name__!r} for term {_term!r}. "
                        "Expected e.g. {'train': 1000, 'test': 500, 'weight': 1.0}"
                    )
                if 'train' in _cfg:
                    self.train_samples[_term] = _cfg['train']
                if 'test' in _cfg:
                    self.test_samples[_term] = _cfg['test']
                if 'weight' in _cfg:
                    self.weights[_term] = _cfg['weight']

        # Update schedulers list FIRST so _create_optimizer can inspect it
        new_schedulers = list(schedulers) if schedulers else []

        optimizer_changed = False
        if optimizer is not None:
            self._optimizer_obj = optimizer
            self.optimizer_name = optimizer.name
            optimizer_changed = True

        # Detect if an lr-aware scheduler presence changed (affects inject_hyperparams)
        old_has_lr_sched = any(hasattr(s, 'lr') for s in getattr(self, '_schedulers', []))
        new_has_lr_sched = any(hasattr(s, 'lr') for s in new_schedulers)
        if old_has_lr_sched != new_has_lr_sched:
            optimizer_changed = True

        self._schedulers = new_schedulers

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
        self._epochs_by_time_step = epochs_by_time_step
        self._batch_size = batch_size

        # ── Plot configuration ────────────────────────────────────────────
        if show is not None:
            if not isinstance(show, TrainPlotter):
                raise TypeError(
                    f"'show' must be a TrainPlotter instance or None, got {type(show).__name__!r}. "
                    "Example: compile(show=TrainPlotter())"
                )
            show._activate(self)
        self._plotter = show
        
        # Backend-specific hook (e.g., presampling in JAX)
        self._after_compile_hook()
        
        self._compiled = True
    def eval(self, x: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """Evaluate the network at given input points.

        Parameters
        ----------
        x : np.ndarray, shape ``(N, n_inputs)``
            Query points.
        batch_size : int, optional
            Chunk size for large inputs to avoid OOM.  Defaults to the
            batch size configured at compile time.

        Returns
        -------
        np.ndarray, shape ``(N, n_outputs)``
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

    def _get_bc_target(self, bc, x):
        """
        Get BC target value as backend tensor.
        
        Args:
            bc: Boundary condition object
            x: Input points (backend-specific tensor)
            
        Returns:
            Target tensor (backend-specific), 1D shape (n_points,)
        """
        if not getattr(bc, 'has_value', False):
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
        from pinns.dataset import TermPoints
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

    def _compute_total_loss_base(self, data: Dict, params_dict: Dict[str, Any], weights_dict: Dict):
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
        bc_names = self._get_bc_names()
        losses['bcs'] = []
        name_idx = 0
        for i, bc in enumerate(self.problem.boundary_conditions):
            if bc.kind == 'periodic':
                continue   # handled by the JIT train step, not here
            name = bc_names[name_idx]
            name_idx += 1
            if name in data:
                x_bc = data[name]
                y_bc = self._call_network(x_bc, params_dict)
                per_output = self._compute_bc_term_loss_dict(bc, x_bc, y_bc, params_dict)
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

        # Dataset: evaluate fixed supervised data terms.
        if self.dataset is not None:
            for _dterm in self.dataset._data_terms:
                _x_d = self._to_tensor(np.asarray(_dterm.inputs, dtype=np.float32))
                _y_d = self._call_network(_x_d, params_dict)
                _col = _dterm.components[0]
                _tgt = self._to_tensor(np.asarray(_dterm.outputs[:, 0]).flatten())
                _res = _y_d[:, _col] - _tgt
                _loss = self._mean_squared(_res)
                _w    = weights_dict.get(_dterm.name, 1.0)
                _wl   = _w * _loss
                losses[_dterm.name] = _wl
                losses['bcs'].append(_wl)
                total_loss = _wl if total_loss is None else total_loss + _wl

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
                continue  # dataset terms are handled separately

            elif term.kind in ('inner', 'initial', 'boundary'):
                # User-provided physics / IC residual
                fn = term.fn
                import inspect as _inspect
                if not callable(fn):
                    # Scalar constant target (e.g. add_initial(1.0, ...))
                    col = getattr(term, 'output_idx', 0) or 0
                    residual = y[:, col:col + 1] - float(fn)
                else:
                    n_p = len(_inspect.signature(fn).parameters)
                    if n_p >= 4:
                        residual = fn(x, y, params_dict, derivative_fn)
                    elif n_p == 3:
                        residual = fn(x, y, params_dict)
                    else:
                        residual = fn(x, y)
                eq_idx = getattr(term, 'eq_idx', None)
                if eq_idx is not None:
                    if hasattr(residual, 'ndim') and residual.ndim == 2:
                        residual = residual[:, eq_idx]

            elif term.kind == 'dirichlet':
                # Structural Dirichlet BC: u_component(x) = value(x)
                residual = self._compute_strong_bc_residual(term, x, y, params_dict, derivative_fn)

            elif term.kind in ('neumann', 'robin'):
                # Residual closure pre-built by add_neumann / add_robin
                fn = getattr(term, 'fn', None)
                if fn is None:
                    raise NotImplementedError(
                        f"No residual function found on term '{term.name}' "
                        f"(kind='{term.kind}'). Use add_boundary() with a callable fn instead."
                    )
                import inspect as _inspect_nr
                n_p = len(_inspect_nr.signature(fn).parameters)
                if n_p >= 4:
                    residual = fn(x, y, params_dict, derivative_fn)
                elif n_p == 3:
                    residual = fn(x, y, params_dict)
                else:
                    residual = fn(x, y)

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

        # Dataset: evaluate fixed supervised data terms.
        if self.dataset is not None:
            for _dterm in self.dataset._data_terms:
                if _dterm.name not in data:
                    continue
                _x_d = data[_dterm.name]
                _y_d = self.network.apply(self.network.params, _x_d, params_dict)
                _col = _dterm.components[0]
                _tgt = self._to_tensor(np.asarray(_dterm.outputs[:, 0]).flatten())
                _res = _y_d[:, _col] - _tgt
                _loss = self._mean_squared(_res)
                _w    = weights_dict.get(_dterm.name, 1.0)
                _wl   = _w * _loss
                losses[_dterm.name] = _wl
                losses['bcs'].append(_wl)
                total_loss = _wl if total_loss is None else total_loss + _wl

        return total_loss, losses

    def _compute_strong_bc_residual(self, term, x, y, params_dict, derivative_fn):
        """Compute residual for structural add_dirichlet terms."""
        col = term.component
        u = y[:, col:col + 1]
        target = term.get_value(x, params_dict)
        if callable(target):
            target = target(x, params_dict)
        target = self._to_tensor(np.asarray(target, dtype=np.float32))
        if hasattr(target, 'ndim') and target.ndim == 2:
            target = target[:, 0:1]
        return u - target

    def _compute_bc_term_loss_dict(self, bc, x, y, params_dict):
        """Compute per-output loss dict for a BC term using kind dispatch."""
        import inspect as _inspect

        kind = bc.kind
        if kind == 'dirichlet':
            col = bc.component
            target = bc.get_value(x, params_dict)
            target = self._to_tensor(np.asarray(target, dtype=np.float32))
            if hasattr(target, 'ndim') and target.ndim == 2:
                target = target[:, 0:1]
            residual = y[:, col:col + 1] - target
            return {bc.name or 'bc': self._mean_squared(residual)}

        elif kind in ('inner', 'initial', 'boundary'):
            fn = bc.fn
            n_p = len(_inspect.signature(fn).parameters)
            if n_p >= 4:
                derivative_fn = self._get_derivative_fn(params_dict)
                residual = fn(x, y, params_dict, derivative_fn)
            elif n_p == 3:
                residual = fn(x, y, params_dict)
            else:
                residual = fn(x, y)
            eq_idx = getattr(bc, 'eq_idx', None)
            if eq_idx is not None and hasattr(residual, 'ndim') and residual.ndim == 2:
                residual = residual[:, eq_idx]
            if not isinstance(residual, (list, tuple)):
                residual = [residual]
            base = bc.name or 'bc'
            names = bc.output_names or ([base] if len(residual) == 1
                                         else [f'{base}_{k}' for k in range(len(residual))])
            return {n: self._mean_squared(r) for n, r in zip(names, residual)}

        elif kind == 'neumann':
            # ProblemWeak assembles Neumann/traction terms inside the weak-form
            # residual_fn (boundary integrals are baked into the cubature data).
            # No separate point-evaluation is needed or possible here.
            from pinns.problems.problem_weak import ProblemWeak as _PWNeu
            if isinstance(self.problem, _PWNeu):
                return {}
            raise NotImplementedError(
                f"Neumann BC '{bc.name}' cannot be auto-evaluated. "
                "Use add_boundary() with a callable fn."
            )

        else:
            return {}

    def _compute_total_loss_batched_base(self, data: Dict, params_dict: Dict[str, Any], 
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

        Handles ProblemStrong and ProblemWeak via duck-typing on the BC
        object's attributes.  This is the single sampling entry point.
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

        # ── ProblemWeak: look up term by kind, route interior vs BC ──
        term = next((t for t in self.problem._terms if t.name == name), None)
        if term is not None and term.kind == 'inner':
            from pinns.problems.problem_weak import ProblemWeak as _PWeak
            from pinns.domain.domain_mesh import DomainMesh as _DMWeak
            if isinstance(self.problem, _PWeak) and isinstance(domain, _DMWeak):
                # Sample N (node_idx, time) pairs — stochastic Galerkin test functions
                _free = getattr(self.problem, 'free_nodes', None)
                _region = getattr(term, 'region', None)
                if _region in ('all', 'inner', None):
                    # No spatial restriction — sample from all free nodes
                    return domain.sample_nodes(n_samples, rng=rng, node_pool=_free)
                else:
                    # Region-restricted: pass region (and optionally intersect with free)
                    return domain.sample_nodes(n_samples, rng=rng,
                                               node_pool=_free, region=_region)
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

        # ProblemWeak mesh BCs — delegate to domain.sample_boundary_bc
        # (geometry is looked up from domain._boundary_regions[bc.region])
        from pinns.problems.problem_weak import ProblemWeak as _PWSample
        from pinns.domain.domain_mesh import DomainMesh as _DMSample
        if isinstance(self.problem, _PWSample) and isinstance(domain, _DMSample):
            pts, idx = domain.sample_boundary_bc(bc, n_samples, rng=rng)
            # Store per-sample normals for Neumann BCs from domain
            _region = getattr(bc, 'region', None)
            if _region == 'all':
                _normals = (domain._infer_edge_outward_normals(domain._bnd_edges)
                            if domain._bnd_edges is not None else None)
            elif _region and _region in domain._boundary_regions:
                _normals = domain._boundary_regions[_region].get('normals')
            else:
                _normals = None
            bc._sampled_normals = _normals[idx] if _normals is not None else None
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
    
    def _sample_train_data(self):
        """Sample training data and store as backend tensors.
        
        Also precomputes target values for BCs with callable value functions.
        """
        self._train_data = {}
        self._train_targets = {}  # Store precomputed target values
        for name, n in self.train_samples.items():
            if n > 0:
                np_data = self._sample_points_np(name, n)
                self._train_data[name] = self._to_tensor(np_data)
        # Dataset: always include all fixed observation points.
        if self.dataset is not None:
            for _dt in self.dataset._data_terms:
                self._train_data[_dt.name] = self._to_tensor(
                    np.asarray(_dt.inputs, dtype=np.float32)
                )
    
    def _sample_test_data(self):
        """Sample test data and store as backend tensors."""
        self._test_data = {}
        self._test_targets = {}  # Store precomputed target values
        for name, n in self.test_samples.items():
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
        """Get list of boundary condition / term names."""
        from pinns.problems.problem_strong import ProblemStrong as _PSbc
        if isinstance(self.problem, _PSbc):
            return [t.name for t in self.problem._terms]
        names = []
        for i, bc in enumerate(self.problem.boundary_conditions):
            if bc.kind == 'periodic':
                continue   # handled separately; no sampled data needed
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
        if self._plotter is not None:
            self._plotter._fig = None
            self._plotter._axes = None
            self._plotter._display_handle = None
            self._plotter._colorbars = []
        self._compiled = False
    
    # ==================== Plotting Methods ====================
    # (see pinns/trainer/_plotting.py — TrainPlotter)

    
    # ==================== Solution Error Computation ====================

    def _is_mesh_domain(self):
        """Return True when the problem domain is a DomainMesh."""
        try:
            from pinns.domain import DomainMesh as _DomainMesh
            return isinstance(self.problem.domain, _DomainMesh)
        except ImportError:
            return False

    def _call_solution(self, x: np.ndarray) -> np.ndarray:
        """Call problem.solution with either 1-arg or 2-arg signature."""
        try:
            return self.problem.solution(x, self._build_params())
        except TypeError:
            return self.problem.solution(x)

    def _compute_solution_error_base(self, n_points: int = 1000) -> Optional[float]:
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

        y_pred = self.eval(x)
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

    def _problem_uses_lagrange(self) -> bool:
        from .schedulers.scheduler_lagrange import SchedulerLagrange as _SLag
        lag = next((s for s in getattr(self, '_schedulers', []) if isinstance(s, _SLag)), None)
        return lag is not None and bool(getattr(lag, 'constraints', None))

    def _resolve_problem_lagrange_constraints(self) -> Optional[List[str]]:
        from .schedulers.scheduler_lagrange import SchedulerLagrange as _SLag
        lag = next((s for s in getattr(self, '_schedulers', []) if isinstance(s, _SLag)), None)
        if lag is None:
            return None
        # Prefer already-resolved list (populated in on_compile); fall back to raw constraints.
        resolved = getattr(lag, '_resolved_constraints', None)
        if resolved is not None:
            return list(resolved)
        raw = getattr(lag, 'constraints', None)
        if not raw:
            return None
        if callable(raw):
            return None  # dynamic; scheduler will resolve on_compile
        return list(raw)

    def _get_soft_bc_names(self) -> list:
        """BC names to show in plots/history — hard-constrained BCs excluded for ProblemWeak."""
        return self._get_bc_names()

    def _get_bc_plot_names(self) -> list:
        """Use soft (filtered) names for plot labels."""
        return self._get_soft_bc_names()

    def compile(
        self,
        *args,
        step_weight_exp: float = 0.0,
        schedulers: Optional[List] = None,
        **kwargs,
    ):
        """
        Compile trainer.

        Parameters
        ----------
        problem : dict, optional
            Per-term config. Keys are loss term names; values are dicts with
            ``train`` (int), ``test`` (int), and/or ``weight`` (float/array)::

                problem={
                    "pde":    {"train": 1000, "test": 500, "weight": 1.0},
                    "bottom": {"train": 100,  "test": 50,  "weight": 100.0},
                }
        schedulers : list, optional
            List of Scheduler instances to use during training.
        """
        # For ProblemWeak: strip 'pde' from test_samples before base class sees it.
        # The base class would try to create collocation test points, which makes no
        # sense for ProblemWeak — node batching is handled separately below.
        from pinns.problems.problem_weak import ProblemWeak as _PW
        _is_weak = isinstance(self.problem, _PW)
        self._step_weight_exp = float(step_weight_exp)

        self._compile_base(*args, schedulers=schedulers, **kwargs)

        # For ProblemWeak: rollout IC BCs are handled separately — no soft loss removal needed here.
        if _is_weak:
            _dom = self.problem.domain
            _t_min = getattr(_dom, '_t_min', None)
            _rollout: set = set()
            if _dom._time_mode == 'discrete' and _t_min is not None:
                for _bc in self.problem.boundary_conditions:
                    if _bc.kind != 'dirichlet': continue
                    _bname = getattr(_bc, 'name', None)
                    _tw = getattr(_bc, 'time_window', None)
                    if _bname is None or _tw is None: continue
                    _pts = [float(v) for v in _tw]
                    if len(_pts) <= 2 and abs(_pts[0] - _t_min) < 1e-10 and abs(_pts[-1] - _t_min) < 1e-10:
                        _rollout.add(_bname)
            for _k in _rollout:
                self._train_data.pop(_k, None)
                self._train_targets.pop(_k, None)
                if self._test_data:
                    self._test_data.pop(_k, None)
                    self._test_targets.pop(_k, None)

        self._train_samples = {}

        # For ProblemWeak transient: time sampling is handled per-step via
        # domain.sample_nodes() → sample_data['pde'] = (N, 2) [node_idx, time].
        # No compile-time cubature tiling needed.

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
        """Create the underlying optimizer from the stored BaseOptimizer instance."""
        has_lr_sched = any(hasattr(s, 'lr') for s in getattr(self, '_schedulers', []))
        return self._optimizer_obj.build(
            grad_clip=getattr(self, '_grad_clip', None),
            lr_scheduler=has_lr_sched,
        )
    
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
        # Inlined from former BaseTrainer._after_compile_hook:
        self._sample_train_data()
        _ts = self.test_samples
        if any(v > 0 for v in (_ts.values() if isinstance(_ts, dict) else _ts)):
            self._sample_test_data()
        for s in getattr(self, '_schedulers', []):
            s.on_compile(self)
        
        # Precompute sparse FBPINN data if network is FBPINN
        # Skip when batching is enabled (indices change)
        use_batching = self._batch_size is not None and self._batch_size > 0

        from ..models.model_partitioned import ModelPartitioned as _MP
        if self._use_sparse_fbpinn and isinstance(self.network, _MP) and not use_batching:
            self._precompute_sparse_data()
        elif use_batching:
            # Clear any precomputed sparse data when batching
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

        # ProblemWeak has no strong-form residual
        return jnp.zeros((x.shape[0], 1))

    def _compute_custom_bc_losses_dict(self, bc, x, y, params_dict, weights_dict=None):
        """Return {output_name: weighted_loss} with full JAX autodiff."""
        if getattr(bc, 'is_weak', False):
            out_names = bc.output_names or [bc.name]
            return {oname: 0.0 for oname in out_names}
        model_apply = lambda p, xin: self.network.apply(p, xin, params_dict)
        deriv_fn = make_derivative_fn(model_apply, self.network.params)
        import inspect as _inspect
        sig = _inspect.signature(bc.fn)
        n_params = len(sig.parameters)
        if n_params >= 4:
            residual = bc.fn(x, y, params_dict, deriv_fn)
        elif n_params == 3:
            residual = bc.fn(x, y, params_dict)
        else:
            residual = bc.fn(x, y)
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
        sig = _inspect.signature(bc.fn)
        n_params = len(sig.parameters)
        if n_params >= 4:
            residual = bc.fn(x, y, params_dict, deriv_fn)
        elif n_params == 3:
            residual = bc.fn(x, y, params_dict)
        else:
            residual = bc.fn(x, y)
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
    
    def _make_compute_loss_fn(self, weights, params_dict, schedulers=None):
        """Build ``compute_loss(params, train_data, targets_dict, scheduler_states)`` closure.

        Both ``ProblemStrong`` and ``ProblemWeak`` expose the same interface:
        ``make_residual_fn(network)`` → ``fn(params, data)``.  A single unified
        branch handles both; the only problem-type-specific side-effect is
        storing ``_weak_loss_fn`` / ``_weak_residual_fn`` on ``self`` so that
        ``_compute_total_loss_batched`` can use them for logging.

        The optional ``scheduler_states`` argument is a dict ``{idx: state}``
        produced by ``Scheduler.get_jit_state()`` for each scheduler.  It must
        be passed as an explicit JIT argument (not captured) so that per-epoch
        updates to arrays like Lagrange multipliers are picked up correctly.

        Returns ``(compute_loss, pde_accepts_derivative=True)``.
        """
        from pinns.problems.problem_weak import ProblemWeak as _ProblemWeak

        _weights    = weights
        _net_losses = list(getattr(self.network, 'network_losses', []))
        _residual_fn = self.problem.make_residual_fn(self.network)
        self._residual_fn = _residual_fn
        self._residual_fn_jit = jax.jit(_residual_fn)
        _schedulers = list(schedulers) if schedulers else []

        def compute_loss(params, train_data, targets_dict, scheduler_states=None):
            residuals  = _residual_fn(params, train_data)
            total_loss = jnp.array(0.0)
            for name, R in residuals.items():
                total_loss = total_loss + _weights.get(name, 1.0) * jnp.mean(R ** 2)
            for nloss in _net_losses:
                x_nl = nloss.x if nloss.x is not None else (train_data or {}).get('pde')
                if x_nl is not None:
                    total_loss = (total_loss
                                  + _weights.get(nloss.name, nloss.weight)
                                  * nloss.fn(params, x_nl))
            # Scheduler extra-loss contributions (e.g. Lagrange terms λᵀr)
            if scheduler_states:
                for idx, s in enumerate(_schedulers):
                    s_state = scheduler_states.get(idx, {})
                    if s_state:
                        total_loss = total_loss + s.extra_loss(residuals, s_state)
            return total_loss, residuals

        return compute_loss, True
    
    # ==================== Solution error (rollout override) ====================

    def _make_jit_train_step(self, weights, params_dict):
        """Create JIT-compiled Adam training step, built on the shared compute_loss core.

        The compiled ``train_step`` accepts ``scheduler_states`` as an explicit
        argument so that per-epoch JAX-array updates (e.g. Lagrange multipliers)
        are reflected without recompilation.
        """
        _schedulers = list(getattr(self, '_schedulers', []))
        compute_loss, pde_accepts_derivative = self._make_compute_loss_fn(
            weights, params_dict, schedulers=_schedulers)
        optim = self.optimizer
        if pde_accepts_derivative:
            @jax.jit
            def train_step(params, opt_state, train_data, targets_dict, scheduler_states):
                (loss, residuals), grads = jax.value_and_grad(
                    compute_loss, has_aux=True
                )(params, train_data, targets_dict, scheduler_states)
                updates, new_opt_state = optim.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return new_params, new_opt_state, loss, residuals
            return train_step, True, False
        else:
            grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
            @jax.jit
            def apply_updates(params, grads, opt_state):
                updates, new_opt_state = optim.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return new_params, new_opt_state
            return (grad_fn, apply_updates), False, False


    def _compute_solution_error(self, n_points: int = 1000):
        """Override: for BPTT rollout mode unroll the full trajectory and compare."""
        from pinns.problems.problem_weak import ProblemWeak as _PW
        _is_rollout = (
            isinstance(self.problem, _PW)
            and getattr(self.problem.domain, '_time_mode', None) == 'discrete'
            and self.problem.solution is not None
            and hasattr(self.network, 'predict_rollout')
        )
        if not _is_rollout:
            return self._compute_solution_error_base(n_points)

        import numpy as _np
        domain = self.problem.domain
        # Use the current curriculum stage count if set, else full domain horizon
        n_steps = getattr(self, '_curriculum_n_steps', None) or domain.n_steps
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
        if isinstance(self.problem, _ProblemWeak) and hasattr(self, '_residual_fn'):
            _inner_names_ct = {t.name for t in getattr(self.problem, '_inner_terms', []) or []} | {'pde'}
            bc_data = {k: v for k, v in data.items() if k not in _inner_names_ct}
            return self._compute_total_loss_base(bc_data, params_dict, weights_dict)
        return self._compute_total_loss_base(data, params_dict, weights_dict)

    def _compute_total_loss_batched(self, data, params_dict, weights_dict, batch_size=1000):
        """Override so the weak-form PDE loss is computed once, not per-batch."""
        from pinns.problems.problem_weak import ProblemWeak as _ProblemWeak
        if isinstance(self.problem, _ProblemWeak) and hasattr(self, '_residual_fn'):
            # Strip inner-term keys (handled by weak assembler) and internal '_'-prefixed keys.
            _inner_names_ctb = ({t.name for t in getattr(self.problem, '_inner_terms', []) or []}
                                | {'pde'})
            bc_data = {k: v for k, v in data.items()
                       if k not in _inner_names_ctb and not k.startswith('_')}
            if bc_data:
                total_loss, losses = self._compute_total_loss_batched_base(
                    bc_data, params_dict, weights_dict, batch_size)
            else:
                total_loss, losses = 0.0, {}
            residuals = (getattr(self, '_last_residuals', None)
                         or self._residual_fn(self.network.params, None))
            weak_pde_loss = 0.0
            for name, r in residuals.items():
                _w = weights_dict.get(name, 1.0)
                _term_loss = float(_w * jnp.mean(r ** 2))
                losses[name] = _term_loss
                weak_pde_loss += _term_loss
            total_loss = (total_loss or 0.0) + weak_pde_loss
            return total_loss, losses
        return self._compute_total_loss_batched_base(data, params_dict, weights_dict, batch_size)

    # ==================== Shared training helpers ====================

    def _setup_training_plot(self, show_plots, save_plots):
        """Build auto-save path, recreate figure if needed, show epoch-0 plot.

        Returns auto_save_path (str or None).
        """
        auto_save_path = None
        if show_plots and not save_plots and not is_notebook():
            import glob as _gl
            import os as _os
            existing = _gl.glob('./pinn_progress_*.png')
            if existing:
                nums = []
                for _f in existing:
                    _base = _os.path.basename(_f)
                    _parts = _base.replace('pinn_progress_', '').replace('.png', '').split('_')
                    try:
                        nums.append(int(_parts[0]))
                    except ValueError:
                        pass
                next_num = max(nums) + 1 if nums else 0
            else:
                next_num = 0
            auto_save_path = f'./pinn_progress_{next_num}.png'

        if show_plots and self._plotter is not None:
            p = self._plotter
            n_zoom_regions = len(p.regions)
            needs_recreation = p._fig is None
            if not needs_recreation and p._axes is not None and n_zoom_regions > 0:
                if 'zoom_0_0' not in p._axes:
                    needs_recreation = True
            if needs_recreation:
                p._fig, p._axes = p._create_figure()
            _, _, p._display_handle = p.plot_progress(
                save_path=None, n_points=p.n_points,
                fig=p._fig, axes=p._axes,
                display_handle=p._display_handle
            )
        return auto_save_path

    def _log_epoch(self, epoch_key, epoch_display, ep_total, reported_loss,
                   elapsed, weights, params_dict,
                   show_plots, save_plots, auto_save_path,
                   metrics_batch_size=1000, stage_prefix='',
                   callback=None, do_plot=True):
        """Record history for one epoch and print/optionally plot progress.

        Parameters
        ----------
        epoch_key       : value appended to history['epoch']
        epoch_display   : epoch number shown in the printed message
        ep_total        : total epochs shown in the printed message
        reported_loss   : loss value recorded in history['train_loss'] / ['loss'];
                          if None, the recomputed full_train_loss is used
        elapsed         : wall-clock seconds since training start
        """
        full_train_loss, individual_losses = self._compute_total_loss_batched(
            self._train_data, params_dict, weights, batch_size=metrics_batch_size
        )
        pde_loss = float(individual_losses.get('pde', 0.0))
        bc_losses = [
            float(individual_losses.get(name, 0.0))
            for name in self._train_data.keys()
            if name != 'pde' and not name.startswith('_')
        ]
        bc_names = self._get_soft_bc_names()
        bc_losses_str = ", ".join(
            f"{name}: {individual_losses.get(name, 0.0):.2e}" for name in bc_names
        )

        _loss_to_record = float(reported_loss) if reported_loss is not None else float(full_train_loss)
        self.history['epoch'].append(epoch_key)
        self.history['train_loss'].append(_loss_to_record)
        self.history['loss'].append(_loss_to_record)
        self.history['loss_pde'].append(pde_loss)
        self.history['loss_bcs'].append(bc_losses)

        _has_test = (
            (any(v > 0 for v in (self.test_samples.values() if isinstance(self.test_samples, dict) else self.test_samples))
             and self._test_data)
            or getattr(self, '_weak_test_loss', False)
        )
        if _has_test:
            _tw_data = self._test_data if self._test_data else {}
            _tw = {k: 1.0 for k in _tw_data.keys()}
            test_total, _ = self._compute_total_loss_batched(
                _tw_data, params_dict, _tw, batch_size=metrics_batch_size
            )
            self.history['test_loss'].append(float(test_total))

        if self.problem.solution is not None:
            self.history['solution_error'].append(self._compute_solution_error())

        msg = (stage_prefix +
               f"Epoch {epoch_display}/{ep_total} | "
               f"Loss: {_loss_to_record:.2e} | "
               f"MSE Loss: {full_train_loss:.2e} | "
               f"PDE: {pde_loss:.2e} | "
               f"BCs: [{bc_losses_str}] | "
               f"Time: {elapsed:.1f}s")
        if self.history['test_loss']:
            msg += f" | Test Loss: {self.history['test_loss'][-1]:.2e}"
        if self.problem.solution is not None:
            msg += f" | Error: {self.history['solution_error'][-1]:.2e}"
        print(msg)

        if callback is not None:
            callback(epoch_display, self)

        if do_plot and (show_plots or save_plots):
            if save_plots:
                plot_path = f"{save_plots}_epoch{epoch_key:05d}.png"
            elif auto_save_path:
                plot_path = auto_save_path
            else:
                plot_path = None
            if self._plotter is not None:
                _, _, self._plotter._display_handle = self._plotter.plot_progress(
                    save_path=plot_path, n_points=self._plotter.n_points,
                    fig=self._plotter._fig, axes=self._plotter._axes,
                    display_handle=self._plotter._display_handle
                )

    # ==================== Public Scheduler / User API ====================

    def resample(self, term: Optional[str] = None, kind: str = "train", n: Optional[int] = None):
        """Resample training or test points.

        Parameters
        ----------
        term : str or None
            Name of the loss term to resample.  ``None`` resamples all terms.
        kind : ``"train"`` | ``"test"``
            Which data split to resample.
        n : int or None
            Number of points.  ``None`` keeps the current count.
        """
        kind = kind.lower()
        if kind not in ("train", "test"):
            raise ValueError(f"kind must be 'train' or 'test', got {kind!r}")

        data_dict    = self._train_data    if kind == "train" else self._test_data
        target_dict  = self._train_targets if kind == "train" else self._test_targets
        samples_dict = self.train_samples  if kind == "train" else self.test_samples

        def _resample_one(name):
            current_n = samples_dict.get(name, 0)
            new_n = n if n is not None else current_n
            if new_n <= 0:
                return
            new_n = int(new_n)
            samples_dict[name] = new_n
            np_data = self._sample_points_np(name, new_n)
            data_dict[name] = self._to_tensor(np_data)
            # Recompute targets for BCs with callable values
            bc = self._get_bc_by_name(name)
            if bc is not None and callable(getattr(bc, 'value', None)):
                params = self._build_params()
                target_np = bc.value(np_data, params)
                target_dict[name] = self._to_tensor(np.atleast_2d(target_np).T
                                                    if target_np.ndim == 1
                                                    else target_np)

        if term is None:
            for name in list(samples_dict.keys()):
                _resample_one(name)
        else:
            _resample_one(term)

    def add_samples(self, term: str, points: np.ndarray, targets=None, kind: str = "train"):
        """Append extra points to an existing term's data tensors.

        Parameters
        ----------
        term : str
            Name of the loss term.
        points : np.ndarray
            New collocation points, shape ``(N, n_inputs)``.
        targets : np.ndarray or None
            Corresponding target values (for supervised BCs).
        kind : ``"train"`` | ``"test"``
        """
        kind = kind.lower()
        data_dict   = self._train_data    if kind == "train" else self._test_data
        target_dict = self._train_targets if kind == "train" else self._test_targets

        new_tensor = self._to_tensor(np.asarray(points, dtype=np.float32))
        if term in data_dict:
            import jax.numpy as _jnp
            data_dict[term] = _jnp.concatenate([data_dict[term], new_tensor], axis=0)
        else:
            data_dict[term] = new_tensor

        if targets is not None:
            t_arr = np.asarray(targets, dtype=np.float32)
            new_t = self._to_tensor(t_arr)
            if term in target_dict:
                import jax.numpy as _jnp
                target_dict[term] = _jnp.concatenate([target_dict[term], new_t], axis=0)
            else:
                target_dict[term] = new_t

        # Keep sample count in sync
        samples = self.train_samples if kind == "train" else self.test_samples
        samples[term] = int(len(data_dict[term]))

    def eval_residuals(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the PDE residual magnitude at given points.

        Parameters
        ----------
        points : np.ndarray, shape ``(N, n_inputs)``

        Returns
        -------
        np.ndarray, shape ``(N,)`` — mean absolute residual across outputs.
        """
        residuals = self._compute_residuals(np.asarray(points, dtype=np.float32))
        # residuals is a list of arrays (one per output) or a single array
        if isinstance(residuals, (list, tuple)):
            return np.mean(np.stack([np.abs(r).flatten() for r in residuals], axis=1), axis=1)
        return np.abs(np.asarray(residuals)).mean(axis=-1).flatten()

    def eval_term_residuals(self, term: str) -> np.ndarray:
        """Return the raw residual vector for *term* at current training points.

        Uses the same ``_residual_fn`` closure built by ``_make_compute_loss_fn``
        so the result is consistent with what enters the JIT training step.

        Parameters
        ----------
        term : str
            Loss-term name (e.g. ``'pde'``, ``'boundary_up'``).

        Returns
        -------
        np.ndarray, shape ``(N,)`` – signed residual values ``r_k``.
        """
        if not hasattr(self, '_residual_fn') or self._residual_fn is None:
            raise RuntimeError(
                "eval_term_residuals() called before compile(). "
                "Call trainer.compile() first."
            )
        # Prefer residuals cached from the most recent JIT training step
        # (zero extra compute).  Fall back to JIT-compiled forward pass
        # only when cache is empty (e.g. on_compile before first step).
        _cached = getattr(self, '_last_residuals', None)
        if _cached is not None and term in _cached:
            return np.asarray(_cached[term]).flatten()
        _eval_fn = getattr(self, '_residual_fn_jit', None) or self._residual_fn
        residuals_dict = _eval_fn(self.network.params, self._train_data)
        if term not in residuals_dict:
            raise KeyError(
                f"Term {term!r} not found in residuals. "
                f"Available: {list(residuals_dict.keys())}"
            )
        return np.asarray(residuals_dict[term]).flatten()

    def set_weights(self, weights: Dict[str, float]):
        """Update loss weights for one or more terms.

        Parameters
        ----------
        weights : dict[str, float]
            Mapping of term name → new weight.  Unspecified terms are unchanged.
        """
        self.weights.update(weights)

    def set_learning_rate(self, lr: float):
        """Update the optimizer learning rate in-place.

        Parameters
        ----------
        lr : float
            New learning rate.
        """
        self._optimizer_obj.learning_rate = float(lr)
        if hasattr(self.opt_state, 'hyperparams') and 'learning_rate' in self.opt_state.hyperparams:
            new_hp = dict(self.opt_state.hyperparams)
            new_hp['learning_rate'] = float(lr)
            self.opt_state = self.opt_state._replace(hyperparams=new_hp)

    # ── Getters ───────────────────────────────────────────────────────────

    def get_epoch(self) -> int:
        """Return the number of epochs completed in the current ``train()`` call.

        Returns 0 when called outside of training.
        """
        return getattr(self, '_current_epoch', 0)

    def get_global_epoch(self) -> int:
        """Return total epochs trained across all ``train()`` calls."""
        return self._global_epoch

    def get_time(self) -> float:
        """Return total wall-clock training time in seconds (sum of epoch times)."""
        times = self.history.get('epoch_times', [])
        return float(sum(times))

    def get_samples(self, term: str, kind: str = "train") -> np.ndarray:
        """Return current collocation points for a term as a NumPy array.

        Parameters
        ----------
        term : str
        kind : ``"train"`` | ``"test"``
        """
        kind = kind.lower()
        data_dict = self._train_data if kind == "train" else self._test_data
        if term not in data_dict:
            raise KeyError(f"No {kind!r} data for term {term!r}. "
                           f"Available: {list(data_dict.keys())}")
        return self._to_numpy(data_dict[term])

    def get_optimizer(self):
        """Return the current :class:`~pinns.BaseOptimizer` instance."""
        return self._optimizer_obj

    def get_learning_rate(self) -> float:
        """Return the current learning rate."""
        return self._optimizer_obj.learning_rate

    def get_loss_history(self) -> Dict:
        """Return the full training history dict.

        Keys include ``'epoch'``, ``'loss'``, ``'train_loss'``, ``'test_loss'``,
        ``'loss_pde'``, ``'loss_bcs'``, ``'solution_error'``, ``'epoch_times'``.
        """
        return dict(self.history)

    # ==================== Training ====================

    def train(self):
        """
        Train the model.
        
        For full JIT compilation with JAX, define your PDE function with 4 arguments:
            def my_pde(X, U, params, derivative):
                u_x = derivative(U, X, 0, (0,))
                ...
        """
        # ── Time-step curriculum (BPTT rollout mode only) ────────────────
        # If epochs_by_time_step is set, run progressive stages:
        #   stage 1 → unroll 1 step  for epochs_by_time_step epochs
        #   stage 2 → unroll 2 steps for epochs_by_time_step epochs
        #   …
        #   stage N → unroll N_TIME steps for epochs_by_time_step epochs
        # Each stage re-JITs the rollout loss fn for the new scan length.
        _ebt = getattr(self, '_epochs_by_time_step', None)
        _n_time_max = (
            self.problem.domain.n_steps
            if getattr(self.problem.domain, '_time_mode', None) == 'discrete'
            else None
        )
        _in_curriculum = getattr(self, '_curriculum_running', False)
        if _ebt is not None and _n_time_max is not None and not _in_curriculum:
            self._curriculum_running = True
            self._curriculum_total_epochs = _n_time_max * _ebt   # for display
            saved_epochs = self._epochs
            try:
                for _stage_steps in range(1, _n_time_max + 1):
                    # Store the current curriculum scan length on the trainer so
                    # _compute_solution_error and the rollout loss builder can read it.
                    self._curriculum_n_steps = _stage_steps
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
                self._curriculum_n_steps = None
                self._epochs = saved_epochs
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
        show_plots = self._plotter is not None
        save_plots = self._plotter.save if self._plotter else None
        
        params_dict = self._build_params()
        weights = self.weights

        # Optimizer-specific loop (e.g. L-BFGS) — returns True if handled
        if self._optimizer_obj.train_loop(
                self, epochs, print_each, show_plots, save_plots, params_dict, weights):
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
        
        auto_save_path = self._setup_training_plot(show_plots, save_plots)
        
        # Initialize RNG key for shuffling
        shuffle_key = jax.random.PRNGKey(self.rng.integers(0, 2**31))

        # Print epoch 0 (before any training)
        if print_each > 0:
            metrics_batch_size = self._batch_size if self._batch_size and self._batch_size > 0 else 1000
            params_dict = self._build_params({'global_step': start_epoch, 'step': 0})
            _cur_stage  = getattr(self, '_curriculum_stage', None)
            _n_stages   = getattr(self, '_curriculum_n_stages', None)
            _tot_epochs = getattr(self, '_curriculum_total_epochs', None)
            _stage_pfx  = f"Stage {_cur_stage}/{_n_stages} | " if _cur_stage is not None else ""
            _ep_total   = _tot_epochs if _tot_epochs is not None else epochs
            self._log_epoch(
                start_epoch, start_epoch, _ep_total, None, 0.0,
                weights, params_dict, show_plots, save_plots, auto_save_path,
                metrics_batch_size=metrics_batch_size, stage_prefix=_stage_pfx,
                do_plot=False,
            )
        
        # ── Hoist static per-epoch checks outside the loop ─────────────────────
        _schedulers_list   = getattr(self, '_schedulers', [])
        _has_schedulers    = bool(_schedulers_list)
        _rfb               = getattr(self, '_rollout_face_batch', None)
        _rnf               = getattr(self, '_rollout_n_faces', None)
        _has_rollout_faces = (_rfb is not None and _rnf is not None)
        _rts_n_nodes       = getattr(self, '_rts_n_nodes', None)
        _has_rts           = _rts_n_nodes is not None
        _al_step_fn        = getattr(self, '_rollout_al_train_step', None)
        _train_targets     = getattr(self, '_train_targets', {})

        self._current_epoch = 0
        for epoch in range(epochs):
            self._current_epoch = epoch
            global_epoch = start_epoch + epoch

            # Scheduler on_epoch_start hooks (resample, adaptive, curriculum, lr, etc.)
            if _has_schedulers:
                for _s in _schedulers_list:
                    _s.on_epoch_start(self, epoch)

            # Collect per-scheduler JIT states (e.g. Lagrange multipliers)
            if _has_schedulers:
                _sched_states = {
                    i: s.get_jit_state()
                    for i, s in enumerate(_schedulers_list)
                }
            else:
                _sched_states = {}

            # ── Rollout face mini-batching: sample fresh indices each epoch ──────────
            if _has_rollout_faces:
                _face_idx = np.random.choice(_rnf, size=_rfb, replace=False).astype(np.int32)
                self._train_data['_rollout_face_idx'] = jnp.array(_face_idx)

            # ── Node mini-batching (train_samples={'pde': N}) ──────────────────────
            if _has_rts:
                _n_free = self._rts_n_free_nodes
                _node_idx = np.random.choice(_n_free, size=_rts_n_nodes, replace=_rts_n_nodes > _n_free).astype(np.int32)
                self._train_data['_node_idx'] = jnp.array(_node_idx)
                # One independent random time per sampled node
                _t_per_node = (self._rts_t_min + np.random.uniform(0, 1, _rts_n_nodes) * (self._rts_t_max - self._rts_t_min)).astype(np.float32)
                self._train_data['_t_vals'] = jnp.array(_t_per_node)

            if use_batching:
                # Shuffle data and targets at the start of each epoch
                shuffle_key, subkey = jax.random.split(shuffle_key)
                shuffled_train_data = {}
                shuffled_train_targets = {}
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
                    if name in _train_targets:
                        shuffled_train_targets[name] = _train_targets[name][perm]
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
                        self.network.params, self.opt_state, loss, _step_res = train_step(
                                self.network.params, self.opt_state, batch_data, batch_targets,
                                _sched_states,
                            )
                        self._last_residuals = _step_res
                    else:
                        (loss, _step_res), grads = grad_fn(self.network.params, batch_data, batch_targets, _sched_states)
                        self._last_residuals = _step_res
                        self.network.params, self.opt_state = apply_updates(self.network.params, grads, self.opt_state)
                    
                    epoch_loss += float(loss)
                
                loss = epoch_loss / n_batches
            else:
                # Full-batch training
                if is_full_jit:
                    # ── Per-step Lagrangian rollout (AL dual-ascent) ──────────
                    if _al_step_fn is not None:
                        self.network.params, self.opt_state, loss, _step_res = _al_step_fn(
                            self.network.params, self.opt_state, self._rollout_lambdas
                        )
                        # Dual ascent: λ ← λ + lr * R̂  (plain numpy, outside JIT)
                        # step_res are already normalised by node_norm inside the
                        # loss fn, so the update is scale-free w.r.t. mesh size.
                        self._rollout_lambdas = self._rollout_lambdas + self.lagrange_lr * _step_res
                    else:
                        self.network.params, self.opt_state, loss, _step_res = train_step(
                                self.network.params, self.opt_state, self._train_data, _train_targets,
                                _sched_states,
                            )
                        self._last_residuals = _step_res
                else:
                    (loss, _step_res), grads = grad_fn(self.network.params, self._train_data, _train_targets, _sched_states)
                    self._last_residuals = _step_res
                    self.network.params, self.opt_state = apply_updates(self.network.params, grads, self.opt_state)

            # Scheduler on_epoch_end hooks (lr plateau detection, etc.)
            if _has_schedulers:
                _loss_float = float(loss)  # one host-device sync when schedulers need it
                for _s in _schedulers_list:
                    _s.on_epoch_end(self, epoch, _loss_float)

            if print_each > 0 and ((global_epoch + 1) % print_each == 0 or epoch == epochs - 1):
                elapsed = time.time() - start_time
                metrics_batch_size = self._batch_size if self._batch_size and self._batch_size > 0 else 1000
                _cur_stage  = getattr(self, '_curriculum_stage', None)
                _n_stages   = getattr(self, '_curriculum_n_stages', None)
                _tot_epochs = getattr(self, '_curriculum_total_epochs', None)
                _stage_pfx  = f"Stage {_cur_stage}/{_n_stages} | " if _cur_stage is not None else ""
                _ep_total   = _tot_epochs if _tot_epochs is not None else epochs + start_epoch
                self._log_epoch(
                    global_epoch, global_epoch + 1, _ep_total, None, elapsed,
                    weights, params_dict, show_plots, save_plots, auto_save_path,
                    metrics_batch_size=metrics_batch_size, stage_prefix=_stage_pfx,
callback=self._plotter.callback if self._plotter else None,
            )
        
        self._global_epoch += epochs
        _total_elapsed = time.time() - start_time
        # Record per-epoch average time so get_time() stays accurate
        _avg_epoch_time = _total_elapsed / epochs if epochs > 0 else 0.0
        self.history['epoch_times'].extend([_avg_epoch_time] * epochs)
        if not getattr(self, '_curriculum_running', False):
            print(f"Training complete in {_total_elapsed:.1f}s")
            for _s in _schedulers_list:
                _s.on_training_end(self)
        self._curriculum_restore()
        
        # Close figure to prevent duplicate display in notebooks
        if is_notebook() and show_plots and self._plotter is not None and self._plotter._fig is not None:
            plt.close(self._plotter._fig)

    def _curriculum_restore(self):
        """Clear per-stage display attributes after a stage's train() call ends."""
        # Only clear when not inside an active curriculum loop.
        # The outer curriculum dispatcher will set _curriculum_stage for the next stage.
        if not getattr(self, '_curriculum_running', False):
            self._curriculum_stage         = None
            self._curriculum_n_stages      = None
            self._curriculum_total_epochs  = None

    def get_history(self) -> Dict:
        """Get training history."""
        return self.history
