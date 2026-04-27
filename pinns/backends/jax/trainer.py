"""
JAX/Optax implementation of PINN Trainer.

Inherits common functionality from BaseTrainer.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from functools import partial
import time
import inspect

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

from ..base_trainer import BaseTrainer, is_notebook
from .functional import set_context, clear_context, make_derivative_fn
from .networks import FBPINN, FNN


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

        # Single-class AL state
        self._is_lagrangian_mode = False
        self.lagrange_multipliers = {}
        self.lagrange_lr = 1.0
        self._lagrange_max = 1e6
        self._lagrange_constraints = None
        self._lagrange_optimizer = None
        self._lagrange_opt_states = {}
        self._lagrange_optimizer_name = 'adam'

    def _problem_uses_lagrange(self) -> bool:
        lagrange = getattr(self.problem, 'lagrange_multipliers', None)
        return bool(lagrange)

    def _resolve_problem_lagrange_constraints(self) -> Optional[List[str]]:
        from pinns.problem_weak import ProblemWeak as _ProblemWeak
        lagrange = getattr(self.problem, 'lagrange_multipliers', None)
        if not lagrange:
            return None
        # For ProblemWeak the list is already in the right format
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

    def compile(
        self,
        *args,
        auto_lagrangian: bool = True,
        lagrange_constraints: list = None,
        lagrange_lr: float = 1.0,
        lagrange_max: float = 1e6,
        lagrange_optimizer: str = "adam",
        **kwargs,
    ):
        """
        Compile trainer in a single-class setup.

        Standard or AL mode is selected within this class (no delegation).
        """
        super().compile(*args, **kwargs)

        resolved_constraints = lagrange_constraints
        if resolved_constraints is None:
            resolved_constraints = self._resolve_problem_lagrange_constraints()

        has_al_request = (
            resolved_constraints is not None
            or self._problem_uses_lagrange()
        )
        self._is_lagrangian_mode = bool(auto_lagrangian and has_al_request)

        self.lagrange_lr = lagrange_lr
        self._lagrange_max = lagrange_max
        self._lagrange_optimizer_name = lagrange_optimizer

        prev_constraints = self._lagrange_constraints
        self._lagrange_constraints = resolved_constraints

        if self._is_lagrangian_mode:
            constraints_changed = (resolved_constraints != prev_constraints)
            first_time = not bool(self.lagrange_multipliers)
            if first_time or constraints_changed:
                _initialize_lagrange_multipliers_impl(self)
            else:
                # Lagrange lr may have changed — rebuild optimizer but keep λ values
                if self._lagrange_optimizer_name == 'adam':
                    self._lagrange_optimizer = optax.adam(self.lagrange_lr)
                elif self._lagrange_optimizer_name == 'sgd':
                    self._lagrange_optimizer = optax.sgd(self.lagrange_lr)
                else:
                    self._lagrange_optimizer = None

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

    def _initialize_lagrange_multipliers(self):
        return _initialize_lagrange_multipliers_impl(self)

    def _reinitialize_lagrange_if_needed(self):
        return _reinitialize_lagrange_if_needed_impl(self)

    def _make_al_loss_fn(self, params_dict):
        return _make_al_loss_fn_impl(self, params_dict)

    def _update_lagrange_multipliers(self, residuals):
        return _update_lagrange_multipliers_impl(self, residuals)

    def get_lagrange_statistics(self) -> Dict[str, Dict[str, float]]:
        return _get_lagrange_statistics_impl(self)

    def reset_lagrange_multipliers(self):
        return _reset_lagrange_multipliers_impl(self)

    def reset_betas(self, betas: dict = None):
        return _reset_betas_impl(self, betas)
    
    # ==================== Device Detection ====================
    
    def _auto_detect_device(self) -> str:
        """Auto-detect the best available device using JAX."""
        return jax.devices()[0].platform  # 'cpu', 'gpu', 'tpu'
    
    # ==================== Optimizer ====================
    
    def _create_optimizer(self):
        """Create the optimizer with injectable hyperparameters for LR scheduling."""
        lr_scheduler = getattr(self, '_lr_scheduler', None)
        
        if self.optimizer_name == "adam":
            if lr_scheduler is not None:
                # Use inject_hyperparams to allow LR updates during training
                return optax.inject_hyperparams(optax.adam)(learning_rate=self.learning_rate)
            return optax.adam(self.learning_rate)
        elif self.optimizer_name == "sgd":
            if lr_scheduler is not None:
                return optax.inject_hyperparams(optax.sgd)(learning_rate=self.learning_rate)
            return optax.sgd(self.learning_rate)
        elif self.optimizer_name == "rmsprop":
            if lr_scheduler is not None:
                return optax.inject_hyperparams(optax.rmsprop)(learning_rate=self.learning_rate)
            return optax.rmsprop(self.learning_rate)
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
        
        if self._use_sparse_fbpinn and isinstance(self.network, FBPINN) and not use_batching and not use_resampling:
            self._precompute_sparse_data()
        elif use_batching or use_resampling:
            # Clear any precomputed sparse data when batching or resampling
            self._precomputed_pde = None
            self._precomputed_bcs = {}
    
    def _precompute_sparse_data(self):
        """Precompute sparse training data for FBPINN."""
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
        """Compute PDE residual using JAX autodiff - supports both 3-arg and 4-arg PDEs."""
        model_apply = lambda p, xin: self.network.apply(p, xin, params_dict)
        
        # Check if PDE accepts derivative function (4-arg signature)
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
        """Evaluate a MeshCustomBC residual with full JAX autodiff."""
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
    
    # ==================== JIT Training Step ====================
    
    def _make_jit_train_step(self, weights, params_dict):
        """Create a JIT-compiled training step function."""
        from pinns.boundary import NeumannBC, MeshNodeBC, MeshCustomBC

        # Pre-extract BC info as static data
        bc_info = []
        dirichlet_bcs = []
        neumann_bcs = []
        mesh_neumann_bcs = []   # MeshNodeBC with bc_type="neumann"
        custom_bc_list = []     # MeshCustomBC entries
        
        # Get precomputed targets for callable BCs
        train_targets = getattr(self, '_train_targets', {})
        
        for name in self._train_data.keys():
            if name == 'pde':
                continue
            bc = self._get_bc_by_name(name)
            if bc is not None:
                if isinstance(bc, MeshNodeBC):
                    if bc.bc_type == "dirichlet":
                        bc_data = {
                            'name': name,
                            'component': bc.component,
                            'is_neumann': False,
                            'normal_dim': 0,
                            'normal_sign': 1,
                            'const_value': bc.value if not callable(bc.value) else None,
                            'has_callable_value': callable(bc.value),
                            'weight': weights.get(name, 1.0),
                        }
                        dirichlet_bcs.append(bc_data)
                    else:  # neumann
                        bc_data = {
                            'name': name,
                            'component': bc.component,
                            'const_value': bc.value if not callable(bc.value) else None,
                            'has_callable_value': callable(bc.value),
                            'weight': weights.get(name, 1.0),
                        }
                        mesh_neumann_bcs.append(bc_data)
                    continue

                # MeshCustomBC: captured for the custom-BC loss block below
                if isinstance(bc, MeshCustomBC):
                    import inspect as _insp
                    _n = len(_insp.signature(bc.f).parameters)
                    custom_bc_list.append((bc, name, weights.get(name, 1.0), _n))
                    continue

                is_neumann = isinstance(bc, NeumannBC)
                normal_info = bc.get_normal_direction() if is_neumann else (0, 1)
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
        
        from pinns.problem_weak import ProblemWeak as _ProblemWeak
        _is_weak = isinstance(self.problem, _ProblemWeak)

        model_apply = self.network.apply
        pde_fn = None if _is_weak else self.problem.pde_fn
        pde_weight = weights.get('pde', 1.0)

        # ── Weak-form: pre-build and JIT the FEM assembler loss ──────────
        if _is_weak:
            _network = self.network
            _n_out = self.problem.n_outputs
            if _n_out == 1:
                def _u_and_grad(params, xy):
                    def u_single(z):
                        return _network.apply(params, z[None])[0, 0]
                    return jax.value_and_grad(u_single)(xy)
            else:
                # Multi-output: return full Jacobian (n_out, n_dims)
                def _u_and_grad(params, xy):
                    def u_vec(z):
                        return _network.apply(params, z[None])[0]  # (n_out,)
                    u = u_vec(xy)
                    jac = jax.jacobian(u_vec)(xy)  # (n_out, n_dims)
                    return u, jac
            _weak_loss_fn = jax.jit(self.problem.make_loss_fn(_u_and_grad, bc_weights=weights))
            self._weak_loss_fn = _weak_loss_fn
            self._weak_residual_fn = jax.jit(
                self.problem.make_residual_vector_fn(_u_and_grad))
        else:
            _weak_loss_fn = None

        def model_apply_with_params(params, x):
            return self.network.apply(params, x, params_dict)
        
        # Check if we should use sparse FBPINN
        use_sparse = (self._use_sparse_fbpinn and 
                      isinstance(self.network, FBPINN) and 
                      self._precomputed_bcs)
        
        # Check if we have sparse PDE indices for differentiable sparse forward
        use_sparse_pde = (self._use_sparse_fbpinn and 
                          isinstance(self.network, FBPINN) and 
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

        def compute_loss(params, train_data, targets_dict, lm_params=None):
            total_loss = 0.0

            # ===== PDE / Weak-form Loss =====
            if _is_weak:
                # Weak-form: cubature assembly, no collocation points needed
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
                
                # Get target: const_value for scalar BCs, precomputed targets for callable BCs
                if bc_data['const_value'] is not None:
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

            # ===== Custom Residual BC Loss (MeshCustomBC) =====
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
    
    # ==================== Loss Computation (weak-form override) ====================

    def _compute_total_loss(self, data, params_dict, weights_dict):
        """Override to include weak-form PDE residual in metrics for ProblemWeak."""
        from pinns.problem_weak import ProblemWeak as _ProblemWeak
        if isinstance(self.problem, _ProblemWeak) and hasattr(self, '_weak_loss_fn'):
            # Strip 'pde' so base never tries to call problem.pde_fn
            bc_data = {k: v for k, v in data.items() if k != 'pde'}
            total_loss, losses = super()._compute_total_loss(bc_data, params_dict, weights_dict)
            # Add weak PDE residual
            pde_weight = weights_dict.get('pde', 1.0)
            weak_pde_loss = float(self._weak_loss_fn(self.network.params))
            losses['pde'] = pde_weight * weak_pde_loss
            extra = pde_weight * weak_pde_loss
            total_loss = extra if total_loss is None else total_loss + extra
            return total_loss, losses
        return super()._compute_total_loss(data, params_dict, weights_dict)

    def _compute_total_loss_batched(self, data, params_dict, weights_dict, batch_size=1000):
        """Override so the weak-form PDE loss is computed once, not per-batch."""
        from pinns.problem_weak import ProblemWeak as _ProblemWeak
        if isinstance(self.problem, _ProblemWeak) and hasattr(self, '_weak_loss_fn'):
            # Strip 'pde' so base never tries to call problem.pde_fn
            bc_data = {k: v for k, v in data.items() if k != 'pde'}
            total_loss, losses = super()._compute_total_loss_batched(
                bc_data, params_dict, weights_dict, batch_size)
            # Add weak residual exactly once
            pde_weight = weights_dict.get('pde', 1.0)
            weak_pde_loss = float(self._weak_loss_fn(self.network.params))
            losses['pde'] = pde_weight * weak_pde_loss
            total_loss = (total_loss or 0.0) + pde_weight * weak_pde_loss
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
        if self._is_lagrangian_mode:
            _train_lagrangian_mode_impl(self)
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
        
        if is_full_jit:
            train_step = result
            if use_batching:
                print(f"Starting training for {epochs} epochs, {n_batches} batches/epoch (JIT-compiled)...")
            else:
                print(f"Starting training for {epochs} epochs (JIT-compiled)...")
        else:
            grad_fn, apply_updates = result
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
        
        # Get resample interval and check if using pool
        resample_each = getattr(self, '_resample_each', 0)
        pool_refresh_each = getattr(self, '_pool_refresh_each', 0)
        has_pool = hasattr(self, '_train_pool') and self._train_pool is not None
        pool_size = getattr(self, '_resample_pool_size', 10)
        
        # Adaptive sampling parameters
        adaptive_sampling = getattr(self, '_adaptive_sampling', False)
        adaptive_each = getattr(self, '_adaptive_each', 100)
        
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
            bc_names = self._get_bc_names()
            bc_losses_str = ", ".join(
                f"{name}: {individual_losses.get(name, 0.0):.2e}" 
                for name in bc_names
            )
            
            self.history['epoch'].append(start_epoch)
            self.history['train_loss'].append(float(full_train_loss))
            self.history['loss'].append(float(full_train_loss))
            self.history['loss_pde'].append(pde_loss)
            bc_losses = [float(individual_losses.get(name, 0.0)) for name in self._train_data.keys() if name != 'pde']
            self.history['loss_bcs'].append(bc_losses)
            
            if any(s > 0 for s in self.test_samples) and self._test_data:
                test_weights = {k: 1.0 for k in self._test_data.keys()}
                test_total, _ = self._compute_total_loss_batched(
                    self._test_data, params_dict, test_weights, batch_size=metrics_batch_size
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

            # Time-domain curriculum: expand sampling window if stage changed
            self._curriculum_step(global_epoch)

            # Update learning rate if scheduler is provided
            # Skip for SOAP (has its own LR handling) and L-BFGS
            if lr_scheduler is not None and self.optimizer_name not in ("lbfgs", "soap"):
                new_lr = lr_scheduler.lr(self.learning_rate, global_epoch)
                # Update the hyperparameter in opt_state (inject_hyperparams stores it there)
                # InjectHyperparamsState is immutable, so we need to create a new state
                if hasattr(self.opt_state, 'hyperparams'):
                    new_hyperparams = dict(self.opt_state.hyperparams)
                    new_hyperparams['learning_rate'] = new_lr
                    self.opt_state = self.opt_state._replace(hyperparams=new_hyperparams)
            
            # Refresh pool with fresh samples if interval reached
            if pool_refresh_each > 0 and has_pool and epoch > 0 and epoch % pool_refresh_each == 0:
                self._sample_pool_data(pool_size)
            
            # Resample collocation points if interval reached
            if resample_each > 0 and epoch > 0 and epoch % resample_each == 0:
                if has_pool:
                    # Fast: select from pre-sampled pool
                    self._select_from_pool()
                else:
                    # Slow: full resampling
                    self._sample_train_data()
                    if any(s > 0 for s in self.test_samples):
                        self._sample_test_data()
            
            # Adaptive resampling based on residuals
            if adaptive_sampling and epoch > 0 and epoch % adaptive_each == 0:
                self._adaptive_resample(params_dict)
                # Recalculate n_batches if dataset size changed (adaptive_mode="add")
                if use_batching and getattr(self, '_adaptive_mode', 'replace') == 'add':
                    n_batches = self._get_n_batches()
            
            if use_batching:
                # Shuffle data and targets at the start of each epoch
                shuffle_key, subkey = jax.random.split(shuffle_key)
                shuffled_train_data = {}
                shuffled_train_targets = {}
                train_targets = getattr(self, '_train_targets', {})
                for name, data in self._train_data.items():
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
                bc_losses = [float(individual_losses.get(name, 0.0)) for name in self._train_data.keys() if name != 'pde']
                
                self.history['epoch'].append(global_epoch)
                self.history['train_loss'].append(float(full_train_loss))
                self.history['loss'].append(float(full_train_loss))
                self.history['loss_pde'].append(pde_loss)
                self.history['loss_bcs'].append(bc_losses)
                
                # Test loss (if test data available)
                if any(s > 0 for s in self.test_samples) and self._test_data:
                    test_weights = {k: 1.0 for k in self._test_data.keys()}
                    test_total, _ = self._compute_total_loss_batched(
                        self._test_data, params_dict, test_weights, batch_size=metrics_batch_size
                    )
                    self.history['test_loss'].append(float(test_total))
                
                if self.problem.solution is not None:
                    sol_error = self._compute_solution_error()
                    self.history['solution_error'].append(sol_error)
                
                bc_names = self._get_bc_names()
                bc_losses_str = ", ".join(
                    f"{name}: {individual_losses.get(name, 0.0):.2e}" 
                    for name in bc_names
                )
                
                msg = (f"Epoch {global_epoch + 1}/{epochs + start_epoch} | "
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
        print(f"Training complete in {time.time() - start_time:.1f}s")
        self._curriculum_restore()
        
        # Close figure to prevent duplicate display in notebooks
        if is_notebook() and show_plots and self._fig is not None:
            plt.close(self._fig)
    
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
            bc_names = self._get_bc_names()
            bc_losses_str = ", ".join(
                f"{name}: {individual_losses.get(name, 0.0):.2e}" 
                for name in bc_names
            )
            
            self.history['epoch'].append(start_epoch)
            self.history['train_loss'].append(float(full_train_loss))
            self.history['loss'].append(float(full_train_loss))
            self.history['loss_pde'].append(pde_loss)
            bc_losses = [float(individual_losses.get(name, 0.0)) for name in self._train_data.keys() if name != 'pde']
            self.history['loss_bcs'].append(bc_losses)
            
            if any(s > 0 for s in self.test_samples) and self._test_data:
                test_weights = {k: 1.0 for k in self._test_data.keys()}
                test_total, _ = self._compute_total_loss_batched(
                    self._test_data, params_dict, test_weights, batch_size=metrics_batch_size
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
                bc_losses = [float(individual_losses.get(name, 0.0)) for name in self._train_data.keys() if name != 'pde']
                
                self.history['epoch'].append(global_epoch)
                self.history['train_loss'].append(float(loss))
                self.history['loss'].append(float(loss))
                self.history['loss_pde'].append(pde_loss)
                self.history['loss_bcs'].append(bc_losses)
                
                # Test loss
                if any(s > 0 for s in self.test_samples) and self._test_data:
                    test_weights = {k: 1.0 for k in self._test_data.keys()}
                    test_total, _ = self._compute_total_loss_batched(
                        self._test_data, params_dict, test_weights, batch_size=metrics_batch_size
                    )
                    self.history['test_loss'].append(float(test_total))
                
                if self.problem.solution is not None:
                    sol_error = self._compute_solution_error()
                    self.history['solution_error'].append(sol_error)
                
                bc_names = self._get_bc_names()
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
        from .networks import FBPINN
        
        # Pre-extract BC info
        dirichlet_bcs = []
        neumann_bcs = []
        mesh_neumann_bcs = []   # MeshNodeBC with bc_type="neumann"
        bc_names = self._get_bc_names()
        
        # Get precomputed targets for callable BCs
        train_targets = getattr(self, '_train_targets', {})
        
        for i, bc in enumerate(self.problem.boundary_conditions):
            from pinns.boundary import DirichletBC, NeumannBC, RobinBC, MeshNodeBC
            name = bc_names[i]

            if isinstance(bc, MeshNodeBC):
                if bc.bc_type == "dirichlet":
                    dirichlet_bcs.append({
                        'name': name,
                        'component': bc.component,
                        'weight': weights.get(name, 1.0),
                        'const_value': bc.value if not callable(bc.value) else None,
                        'has_callable_value': callable(bc.value),
                    })
                else:  # neumann
                    mesh_neumann_bcs.append({
                        'name': name,
                        'component': bc.component,
                        'weight': weights.get(name, 1.0),
                        'const_value': bc.value if not callable(bc.value) else None,
                        'has_callable_value': callable(bc.value),
                    })
                continue

            is_neumann = isinstance(bc, (NeumannBC, RobinBC))
            
            bc_data = {
                'name': name,
                'component': bc.component,
                'weight': weights.get(name, 1.0),
                'const_value': bc.value if not callable(bc.value) else None,
                'has_callable_value': callable(bc.value),
                'dim': bc.boundary.index(1) if 1 in bc.boundary else bc.boundary.index(0),
            }
            
            if is_neumann:
                normal_sign = 1.0 if 1 in bc.boundary else -1.0
                bc_data['normal_sign'] = normal_sign
                if isinstance(bc, NeumannBC):
                    neumann_bcs.append(bc_data)
            else:
                dirichlet_bcs.append(bc_data)
        
        pde_fn = self.problem.pde_fn
        pde_weight = weights.get('pde', 1.0)
        
        # Check if sparse FBPINN
        use_sparse_pde = (self._use_sparse_fbpinn and 
                          isinstance(self.network, FBPINN) and 
                          self._precomputed_pde is not None)
        use_sparse = (self._use_sparse_fbpinn and 
                      isinstance(self.network, FBPINN) and 
                      self._precomputed_bcs)
        
        precomputed_bcs = self._precomputed_bcs if use_sparse else {}
        precomputed_pde = self._precomputed_pde if use_sparse_pde else None
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
                
                # Get target: const_value for scalar BCs, precomputed targets for callable BCs
                if bc_data['const_value'] is not None:
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
        self._lagrange_optimizer = optax.adam(self.lagrange_lr)
    elif self._lagrange_optimizer_name == 'sgd':
        self._lagrange_optimizer = optax.sgd(self.lagrange_lr)
    else:
        self._lagrange_optimizer = None

    if 'pde' in self._train_data and ((self._lagrange_constraints is None) or ('pde' in self._lagrange_constraints)):
        n = len(self._train_data['pde'])
        self.lagrange_multipliers['pde'] = jnp.zeros(n)
        if self._lagrange_optimizer is not None:
            self._lagrange_opt_states['pde'] = self._lagrange_optimizer.init(self.lagrange_multipliers['pde'])

    # For ProblemWeak the PDE residual has size n_free_nodes * n_outputs
    # (all component residuals are concatenated: [R1_free; R2_free; ...])
    from pinns.problem_weak import ProblemWeak as _ProblemWeak
    if isinstance(self.problem, _ProblemWeak):
        if ((self._lagrange_constraints is None) or ('pde' in self._lagrange_constraints)):
            n = self.problem.n_free_nodes * self.problem.n_outputs
            self.lagrange_multipliers['pde'] = jnp.zeros(n)
            if self._lagrange_optimizer is not None:
                self._lagrange_opt_states['pde'] = self._lagrange_optimizer.init(self.lagrange_multipliers['pde'])

    for name in self._get_bc_names():
        if name in self._train_data and ((self._lagrange_constraints is None) or (name in self._lagrange_constraints)):
            n = len(self._train_data[name])
            self.lagrange_multipliers[name] = jnp.zeros(n)
            if self._lagrange_optimizer is not None:
                self._lagrange_opt_states[name] = self._lagrange_optimizer.init(self.lagrange_multipliers[name])


def _reinitialize_lagrange_if_needed_impl(self):
    for name, data in self._train_data.items():
        if name in self.lagrange_multipliers and len(self.lagrange_multipliers[name]) != len(data):
            self.lagrange_multipliers[name] = jnp.zeros(len(data))
            if self._lagrange_optimizer is not None:
                self._lagrange_opt_states[name] = self._lagrange_optimizer.init(self.lagrange_multipliers[name])


def _make_al_loss_fn_impl(self, params_dict):
    from pinns.boundary import NeumannBC, RobinBC, MeshNodeBC
    from pinns.problem_weak import ProblemWeak as _ProblemWeak
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

        _weak_res_fn = jax.jit(self.problem.make_residual_vector_fn(_u_and_grad))
        # Store for the residual plot
        self._weak_residual_fn = _weak_res_fn
        _n_dofs  = self.problem.n_dofs
        _n_comp  = self.problem.n_outputs
        # For multi-component: residual vector is [R1; R2; ...] of length n_dofs*n_comp
        # Free indices span all components
        _free_base = jnp.array(self.problem.free_nodes, dtype=jnp.int32)
        _free_nodes_jax = jnp.concatenate(
            [_free_base + k * _n_dofs for k in range(_n_comp)]
        ) if _n_comp > 1 else _free_base

        def _model_apply(p, x):
            return network.apply(p, x)

        train_targets = getattr(self, '_train_targets', {})
        bc_names = self._get_bc_names()

        # Collect Dirichlet BC info (same as strong form)
        _bc_point_info = []
        for i, bc in enumerate(self.problem.boundary_conditions):
            name = bc_names[i]
            if isinstance(bc, MeshNodeBC) and bc.bc_type == 'dirichlet':
                _bc_point_info.append({
                    'name': name,
                    'component': bc.component,
                    'const_value': bc.value if not callable(bc.value) else None,
                })

        def compute_residuals_weak(params, train_data, targets_dict=None):
            targets_dict = {} if targets_dict is None else targets_dict
            residuals = {}
            # PDE: weak-form R_free (shape n_free_nodes)
            R_full = _weak_res_fn(params)
            residuals['pde'] = R_full[_free_nodes_jax]
            # BCs: point evaluation  u(x_k) − g
            for info in _bc_point_info:
                bname = info['name']
                if bname not in train_data:
                    continue
                x_bc = train_data[bname]
                y_bc = _model_apply(params, x_bc)
                comp   = info['component']
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
                if name != 'pde':
                    losses['bcs'].append(constraint_loss)
                total_loss = total_loss + constraint_loss
            return total_loss, (losses, residuals)

        return compute_al_loss_weak, compute_residuals_weak

    bc_info = {}
    bc_names = self._get_bc_names()
    for i, bc in enumerate(self.problem.boundary_conditions):
        name = bc_names[i]
        if isinstance(bc, MeshNodeBC):
            is_mesh_neumann = (bc.bc_type == "neumann") and (bc.edge_normals is not None)
            is_mesh_time_neumann = (bc.bc_type == "neumann") and (bc.t_mode in ("t_min", "t_max"))
            bc_info[name] = {
                'component': bc.component,
                'is_neumann': False,
                'is_mesh_neumann': is_mesh_neumann,
                'is_mesh_time_neumann': is_mesh_time_neumann,
                'const_value': bc.value if not callable(bc.value) else None,
                'normal_dim': self.problem.domain._spatial_dims if is_mesh_time_neumann else 0,
                'normal_sign': (-1 if bc.t_mode == "t_min" else 1) if is_mesh_time_neumann else 1,
            }
        else:
            is_neumann = isinstance(bc, (NeumannBC, RobinBC))
            bc_info[name] = {
                'component': bc.component,
                'is_neumann': is_neumann,
                'is_mesh_neumann': False,
                'is_mesh_time_neumann': False,
                'const_value': bc.value if not callable(bc.value) else None,
            }
            if is_neumann:
                normal_dim, normal_sign = bc.get_normal_direction()
                bc_info[name]['normal_dim'] = normal_dim
                bc_info[name]['normal_sign'] = normal_sign

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
        bc_names = self._get_bc_names()
        pde_mse0 = float(jnp.mean(residuals0['pde'] ** 2)) if 'pde' in residuals0 else 0.0
        bc_mse0 = [float(jnp.mean(residuals0[n] ** 2)) if n in residuals0 else 0.0 for n in bc_names]
        mse0 = float(sum(jnp.mean(g ** 2) for g in residuals0.values()))
        self.history['epoch'].append(start_epoch)
        self.history['loss'].append(mse0)
        self.history['train_loss'].append(mse0)
        self.history['loss_pde'].append([pde_mse0])
        self.history['loss_bcs'].append(bc_mse0)
        if any(s > 0 for s in self.test_samples) and self._test_data:
            metrics_batch_size = self._batch_size if self._batch_size and self._batch_size > 0 else 1000
            test_weights0 = {k: 1.0 for k in self._test_data.keys()}
            test_total0, _ = self._compute_total_loss_batched(
                self._test_data, params_dict, test_weights0, batch_size=metrics_batch_size
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
            bc_names = self._get_bc_names()
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
            if any(s > 0 for s in self.test_samples) and self._test_data:
                metrics_batch_size = self._batch_size if self._batch_size and self._batch_size > 0 else 1000
                test_weights = {k: 1.0 for k in self._test_data.keys()}
                test_total, _ = self._compute_total_loss_batched(
                    self._test_data, params_dict, test_weights, batch_size=metrics_batch_size
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

    