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
        # Initialize base class (handles network.to(), normalization, defaults)
        super().__init__(problem, network, device)
    
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
        from pinns.boundary import NeumannBC
        
        # Pre-extract BC info as static data
        bc_info = []
        dirichlet_bcs = []
        neumann_bcs = []
        
        # Get precomputed targets for callable BCs
        train_targets = getattr(self, '_train_targets', {})
        
        for name in self._train_data.keys():
            if name == 'pde':
                continue
            bc = self._get_bc_by_name(name)
            if bc is not None:
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
        
        model_apply = self.network.apply
        pde_fn = self.problem.pde_fn
        pde_weight = weights.get('pde', 1.0)
        
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
        
        sig = inspect.signature(pde_fn)
        pde_accepts_derivative = len(sig.parameters) >= 4
        
        # Precompute n_dims from first BC if available
        n_dims = self.problem.n_dims
        
        def compute_loss(params, train_data, targets_dict):
            total_loss = 0.0
            
            # ===== PDE Loss =====
            if 'pde' in train_data:
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
            
            return total_loss
        
        if pde_accepts_derivative:
            @jax.jit
            def train_step(params, opt_state, train_data, targets_dict):
                loss, grads = jax.value_and_grad(compute_loss)(params, train_data, targets_dict)
                updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return new_params, new_opt_state, loss
            
            return train_step, True
        else:
            grad_fn = jax.value_and_grad(compute_loss)
            
            @jax.jit
            def apply_updates(params, grads, opt_state):
                updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return new_params, new_opt_state
            
            return (grad_fn, apply_updates), False
    
    # ==================== Training ====================
    
    def train(self):
        """
        Train the model.
        
        For full JIT compilation with JAX, define your PDE function with 4 arguments:
            def my_pde(X, U, params, derivative):
                u_x = derivative(U, X, 0, (0,))
                ...
        """
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
        
        result, is_full_jit = self._make_jit_train_step(weights, params_dict)
        
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
        bc_names = self._get_bc_names()
        
        # Get precomputed targets for callable BCs
        train_targets = getattr(self, '_train_targets', {})
        
        for i, bc in enumerate(self.problem.boundary_conditions):
            from pinns.boundary import DirichletBC, NeumannBC, RobinBC
            name = bc_names[i]
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
            
            return total_loss
        
        return compute_loss

    def get_history(self) -> Dict:
        """Get training history."""
        return self.history


class ALTrainer(Trainer):
    """
    Augmented Lagrangian Trainer for Physics-Informed Neural Networks (JAX).
    
    Implements the augmented Lagrangian method where the loss function is:
        L = ||data - NN(x_i)||_L2 + sum_i [β_i ||g_i(x_j)||^2 + λ_{ij} g_i(x_j)]
    
    where:
        - The first term is the data loss (supervised)
        - β_i are fixed penalty parameters for each constraint i (PDE, boundaries)
        - g_i(x_j) are the constraint residuals at points j
        - λ_{ij} are Lagrange multipliers for each constraint i at point j
    
    The optimization is a min-max problem:
        - Minimize over NN parameters
        - Maximize over Lagrange multipliers λ
    
    Algorithm:
        1. Initialize λ_{ij} = 0 for all constraints and points
        2. For each epoch:
           a. Fix λ, minimize over NN params (primal update)
           b. Update λ: λ_{ij} = λ_{ij} + λ_lr * g_i(x_j) (dual update)
        3. Repeat until convergence
    
    Example:
        trainer = ALTrainer(problem, network)
        trainer.compile(
            optimizer="adam",
            learning_rate=1e-3,
            weights={'pde': 1.0, 'left': 500.0, 'right': 500.0},
            lagrange_lr=1.0,
        )
        trainer.train()
    
    Attributes:
        lagrange_multipliers: Dict of Lagrange multipliers λ_i for each constraint (per point)
        lagrange_lr: Learning rate for dual (λ) updates
    """
    
    def __init__(
        self,
        problem,
        network,
        device=None,
    ):
        """
        Initialize ALTrainer.
        
        Args:
            problem: The problem to solve (domain, PDE, BCs, params).
            network: The neural network to train.
            device: Device to use ('cpu', 'gpu', 'tpu'). Default: auto-detect.
        """
        super().__init__(problem, network, device)
        
        # Augmented Lagrangian specific attributes
        self.lagrange_multipliers = {}  # Lagrange multipliers λ_i (per constraint, per point)
        self.lagrange_lr = 1.0  # Dual learning rate
        self._outer_epoch = 0
        self._lagrange_max = 1e6  # Maximum absolute value for λ (clipping)
        self._lagrange_constraints = None  # Which constraints get λ (None = all)
        self._lagrange_optimizer = None  # Optimizer for λ updates
        self._lagrange_opt_states = {}  # Optimizer states for each λ
    
    def compile(
        self,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        train_samples: int = 1000,
        test_samples: int = 0,
        epochs: int = 10000,
        weights: dict = None,
        lagrange_lr: float = 1.0,
        lagrange_max: float = 1e6,
        lagrange_constraints: list = None,
        lagrange_optimizer: str = "adam",
        print_each: int = 100,
        show_plots: bool = False,
        save_plots: str = None,
        resample_each: int = 0,
        plot_n_points: int = 10000,
        plot_regions: list = None,
        adaptive_sampling: bool = False,
        adaptive_each: int = 100,
        adaptive_ratio: float = 0.5,
        adaptive_mode: str = 'replace',
        lr_scheduler = None,
        batch_size: int = None,
        **kwargs
    ):
        """
        Configure training parameters for ALTrainer.
        
        Args:
            optimizer: Optimizer name ('adam', 'sgd', 'rmsprop').
            learning_rate: Learning rate for the optimizer.
            train_samples: Number of training samples per constraint.
            test_samples: Number of test samples (0 to disable).
            epochs: Total number of epochs.
            weights: Penalty parameters (weights) for each constraint.
                   Dict with keys 'pde' and BC names, values are floats.
                   Example: {'pde': 1.0, 'left': 500.0, 'right': 500.0}
                   These serve as β_i in the augmented Lagrangian loss.
            lagrange_lr: Learning rate for Lagrange multiplier updates.
            lagrange_max: Maximum absolute value for λ (clipping for stability).
            lagrange_constraints: List of constraint names to apply λ to. Default None = all.
                   Use e.g. ['left', 'right', 'bottom', 'top'] to only apply λ to boundaries.
            lagrange_optimizer: Optimizer for λ updates ('adam', 'sgd', or 'none'). Default 'adam'.
                   'adam' matches the AL-PINNs paper using Adam with momentum for stable λ updates.
                   'none' uses simple gradient ascent: λ = λ + lr * g / N.
            print_each: Print progress every N epochs.
            show_plots: Show live plots during training.
            save_plots: Path prefix for saving plots.
            resample_each: Resample collocation points every N epochs.
            plot_n_points: Number of points for plotting.
            plot_regions: List of zoom regions for plots.
            adaptive_sampling: Enable adaptive sampling based on residuals.
            adaptive_each: Frequency of adaptive resampling.
            adaptive_ratio: Fraction of points to resample adaptively.
            adaptive_mode: 'replace' or 'add' for adaptive sampling.
            lr_scheduler: Learning rate scheduler.
            batch_size: Batch size for mini-batch training.
            **kwargs: Additional arguments passed to parent compile().
        """
        # Store AL-specific parameters
        self.lagrange_lr = lagrange_lr
        self._lagrange_max = lagrange_max
        self._lagrange_constraints = lagrange_constraints  # None means all constraints
        self._lagrange_optimizer_name = lagrange_optimizer  # 'adam', 'sgd', or 'none'
        
        # Call parent compile (handles optimizer, sampling, weights, etc.)
        super().compile(
            optimizer=optimizer,
            learning_rate=learning_rate,
            train_samples=train_samples,
            test_samples=test_samples,
            epochs=epochs,
            weights=weights,
            print_each=print_each,
            show_plots=show_plots,
            save_plots=save_plots,
            resample_each=resample_each,
            plot_n_points=plot_n_points,
            plot_regions=plot_regions,
            adaptive_sampling=adaptive_sampling,
            adaptive_each=adaptive_each,
            adaptive_ratio=adaptive_ratio,
            adaptive_mode=adaptive_mode,
            lr_scheduler=lr_scheduler,
            batch_size=batch_size,
            **kwargs
        )
    
    def _after_compile_hook(self):
        """Initialize Lagrange multipliers after data is sampled."""
        super()._after_compile_hook()
        self._initialize_lagrange_multipliers()
    
    def _initialize_lagrange_multipliers(self):
        """Initialize Lagrange multipliers λ for each constraint and point."""
        self.lagrange_multipliers = {}
        self._lagrange_opt_states = {}
        
        # Create λ optimizer (matching paper: Adam with lagrange_lr)
        if self._lagrange_optimizer_name == 'adam':
            self._lagrange_optimizer = optax.adam(self.lagrange_lr)
        elif self._lagrange_optimizer_name == 'sgd':
            self._lagrange_optimizer = optax.sgd(self.lagrange_lr)
        else:
            self._lagrange_optimizer = None  # Simple gradient ascent
        
        # For PDE constraint
        if 'pde' in self._train_data:
            n_pde_points = len(self._train_data['pde'])
            self.lagrange_multipliers['pde'] = jnp.zeros(n_pde_points)
            if self._lagrange_optimizer is not None:
                self._lagrange_opt_states['pde'] = self._lagrange_optimizer.init(self.lagrange_multipliers['pde'])
        
        # For each BC constraint
        bc_names = self._get_bc_names()
        for name in bc_names:
            if name in self._train_data:
                n_bc_points = len(self._train_data[name])
                self.lagrange_multipliers[name] = jnp.zeros(n_bc_points)
                if self._lagrange_optimizer is not None:
                    self._lagrange_opt_states[name] = self._lagrange_optimizer.init(self.lagrange_multipliers[name])
    
    def _reinitialize_lagrange_if_needed(self):
        """Reinitialize λ if data size changed (e.g., after resampling)."""
        for name, data in self._train_data.items():
            if name in self.lagrange_multipliers:
                if len(self.lagrange_multipliers[name]) != len(data):
                    self.lagrange_multipliers[name] = jnp.zeros(len(data))
                    if self._lagrange_optimizer is not None:
                        self._lagrange_opt_states[name] = self._lagrange_optimizer.init(self.lagrange_multipliers[name])
    
    def _make_al_loss_fn(self, params_dict):
        """
        Create augmented Lagrangian loss function for JIT compilation.
        
        Returns a function that computes:
            L = sum_i [β_i ||g_i||^2 + λ_i · g_i]
        """
        from pinns.boundary import NeumannBC, DirichletBC, RobinBC
        
        # Pre-extract BC info
        bc_info = {}
        bc_names = self._get_bc_names()
        
        for i, bc in enumerate(self.problem.boundary_conditions):
            name = bc_names[i]
            is_neumann = isinstance(bc, (NeumannBC, RobinBC))
            
            bc_info[name] = {
                'component': bc.component,
                'is_neumann': is_neumann,
                'const_value': bc.value if not callable(bc.value) else None,
                'has_callable_value': callable(bc.value),
            }
            
            if is_neumann:
                normal_dim, normal_sign = bc.get_normal_direction()
                bc_info[name]['normal_dim'] = normal_dim
                bc_info[name]['normal_sign'] = normal_sign
        
        pde_fn = self.problem.pde_fn
        network = self.network
        
        sig = inspect.signature(pde_fn)
        pde_accepts_derivative = len(sig.parameters) >= 4
        
        def model_apply_with_params(params, x):
            return network.apply(params, x, params_dict)
        
        def compute_residuals(params, train_data, targets_dict=None):
            """Compute all constraint residuals."""
            residuals = {}
            if targets_dict is None:
                targets_dict = {}
            
            # PDE residual
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
                
                if isinstance(pde_residual, (list, tuple)):
                    residuals['pde'] = sum(r.flatten() for r in pde_residual)
                else:
                    residuals['pde'] = pde_residual.flatten()
            
            # BC residuals
            for name, info in bc_info.items():
                if name in train_data:
                    x_bc = train_data[name]
                    y_bc = model_apply_with_params(params, x_bc)
                    comp = info['component']
                    u = y_bc[:, comp]
                    
                    if info['is_neumann']:
                        # Compute du/dn
                        normal_dim = info['normal_dim']
                        normal_sign = info['normal_sign']
                        
                        def forward_component(x):
                            return model_apply_with_params(params, x)[:, comp]
                        
                        tangent = jnp.zeros_like(x_bc)
                        tangent = tangent.at[:, normal_dim].set(1.0)
                        _, du_dn = jax.jvp(forward_component, (x_bc,), (tangent,))
                        
                        # Get target: const_value for scalar BCs, targets_dict for callable BCs
                        if info['const_value'] is not None:
                            target = info['const_value']
                        elif name in targets_dict:
                            target = targets_dict[name]
                        else:
                            target = 0.0
                        residuals[name] = (normal_sign * du_dn - target).flatten()
                    else:
                        # Dirichlet
                        # Get target: const_value for scalar BCs, targets_dict for callable BCs
                        if info['const_value'] is not None:
                            target = info['const_value']
                        elif name in targets_dict:
                            target = targets_dict[name]
                        else:
                            target = 0.0
                        residuals[name] = (u - target).flatten()
            
            return residuals
        
        def compute_al_loss(params, train_data, lagrange_dict, weights_dict, targets_dict=None):
            """
            Compute augmented Lagrangian loss.
            
            L = Σ_i [w_i * (1/N_i) Σ_j ||g_ij||² + (1/N_i) Σ_j λ_ij · g_ij]
            
            Where w_i are the weights (penalty parameters β).
            All terms are normalized by the number of points per constraint.
            λ terms are only added for constraints in lagrange_constraints (or all if None).
            """
            if targets_dict is None:
                targets_dict = {}
            residuals = compute_residuals(params, train_data, targets_dict)
            
            total_loss = 0.0
            losses = {}
            
            # Get lagrange_constraints from outer scope
            lc = self._lagrange_constraints
            
            for name, g in residuals.items():
                weight = weights_dict.get(name, 1.0)
                lam = lagrange_dict.get(name, jnp.zeros_like(g))
                
                # Ensure λ size matches
                if len(lam) != len(g):
                    lam = jnp.zeros_like(g)
                
                # Penalty term: w_i * (1/N_i) Σ_j ||g_ij||²
                penalty = weight * jnp.mean(g ** 2)
                
                # Lagrangian term: only if this constraint uses λ
                use_lambda = (lc is None) or (name in lc)
                if use_lambda:
                    lagrangian = jnp.mean(jax.lax.stop_gradient(lam) * g)
                else:
                    lagrangian = 0.0
                
                constraint_loss = penalty + lagrangian
                losses[name] = constraint_loss
                losses[f'{name}_penalty'] = penalty
                losses[f'{name}_lagrangian'] = lagrangian
                losses[f'{name}_residual_mean'] = jnp.mean(jnp.abs(g))
                total_loss = total_loss + constraint_loss
            
            return total_loss, (losses, residuals)
        
        return compute_al_loss, compute_residuals
    
    def _update_lagrange_multipliers(self, residuals):
        """
        Update Lagrange multipliers (dual ascent step).
        
        When using optimizer (default 'adam', matching AL-PINNs paper):
            The gradient of mean(λ·g) w.r.t. λ is g/N.
            For gradient ascent (maximization), we negate: grad = -g/N
            Then apply optimizer.update() for adaptive updates.
        
        When optimizer='none' (simple gradient ascent):
            λ_{ij} = λ_{ij} + lagrange_lr * g_i(x_j) / N_i
        
        Only updates λ for constraints in lagrange_constraints (or all if None).
        """
        lc = self._lagrange_constraints
        for name, g in residuals.items():
            # Only update λ for specified constraints
            use_lambda = (lc is None) or (name in lc)
            if name in self.lagrange_multipliers and use_lambda:
                n_points = len(g)
                
                if self._lagrange_optimizer is not None:
                    # Optimizer-based update (paper's approach)
                    # Gradient for ascent: negate the gradient g/N
                    grad = -g / n_points
                    updates, new_state = self._lagrange_optimizer.update(
                        grad, self._lagrange_opt_states[name], self.lagrange_multipliers[name]
                    )
                    self.lagrange_multipliers[name] = optax.apply_updates(self.lagrange_multipliers[name], updates)
                    self._lagrange_opt_states[name] = new_state
                else:
                    # Simple gradient ascent
                    self.lagrange_multipliers[name] = self.lagrange_multipliers[name] + self.lagrange_lr * g / n_points
                
                # Clip λ for stability
                self.lagrange_multipliers[name] = jnp.clip(
                    self.lagrange_multipliers[name], 
                    -self._lagrange_max, 
                    self._lagrange_max
                )
    
    def train(self):
        """
        Run augmented Lagrangian training loop.
        
        Each epoch:
            1. Primal optimization step (minimize over NN params)
            2. Dual update (λ = λ + lr * g)
        """
        epochs = self._epochs
        print_each = self._print_each
        show_plots = self._show_plots
        save_plots = self._save_plots
        
        params_dict = self._build_params()
        weights_dict = self._list_to_dict_weights(self.weights)
        
        # Create AL loss function
        compute_al_loss, compute_residuals = self._make_al_loss_fn(params_dict)
        
        # JIT compile the training step
        @jax.jit
        def train_step(params, opt_state, train_data, lagrange_dict, weights_dict, targets_dict):
            (loss, (losses, residuals)), grads = jax.value_and_grad(
                compute_al_loss, has_aux=True
            )(params, train_data, lagrange_dict, weights_dict, targets_dict)
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss, losses, residuals
        
        # JIT compile residual computation for dual update
        @jax.jit
        def compute_residuals_jit(params, train_data, targets_dict):
            def model_apply_with_params(p, x):
                return self.network.apply(p, x, params_dict)
            
            residuals = {}
            
            if 'pde' in train_data:
                x_pde = train_data['pde']
                y_pde = model_apply_with_params(params, x_pde)
                deriv_fn = make_derivative_fn(model_apply_with_params, params)
                
                sig = inspect.signature(self.problem.pde_fn)
                if len(sig.parameters) >= 4:
                    pde_residual = self.problem.pde_fn(x_pde, y_pde, params_dict, deriv_fn)
                else:
                    set_context(self.network.apply, params)
                    try:
                        pde_residual = self.problem.pde_fn(x_pde, y_pde, params_dict)
                    finally:
                        clear_context()
                
                if isinstance(pde_residual, (list, tuple)):
                    residuals['pde'] = sum(r.flatten() for r in pde_residual)
                else:
                    residuals['pde'] = pde_residual.flatten()
            
            return residuals
        
        start_time = time.time()
        start_epoch = self._global_epoch
        
        # Auto-save path for script mode
        auto_save_path = None
        if show_plots and not save_plots and not is_notebook():
            import glob
            import os
            existing = glob.glob('./al_progress_*.png')
            nums = [int(os.path.basename(f).replace('al_progress_', '').replace('.png', '').split('_')[0]) 
                    for f in existing if f.replace('al_progress_', '').replace('.png', '').split('_')[0].isdigit()]
            next_num = max(nums) + 1 if nums else 0
            auto_save_path = f'./al_progress_{next_num}.png'
        
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
        
        resample_each = getattr(self, '_resample_each', 0)
        adaptive_sampling = getattr(self, '_adaptive_sampling', False)
        adaptive_each = getattr(self, '_adaptive_each', 100)
        lr_scheduler = getattr(self, '_lr_scheduler', None)
        
        print(f"Starting ALTrainer (JAX) for {epochs} epochs...")
        print(f"Weights (penalty parameters): {weights_dict}")
        print(f"λ optimizer: {self._lagrange_optimizer_name}, lr: {self.lagrange_lr}")
        
        # Print epoch 0 (before any training)
        train_targets = getattr(self, '_train_targets', {})
        if print_each > 0:
            _, (losses_init, residuals_init) = compute_al_loss(
                self.network.params, self._train_data, 
                self.lagrange_multipliers, weights_dict, train_targets
            )
            al_loss_init = float(losses_init.get('total', 0.0))
            mse_loss_init = float(losses_init.get('mse_total', 0.0))
            pde_mse_init = float(jnp.mean(residuals_init['pde'] ** 2)) if 'pde' in residuals_init else 0.0
            bc_names = self._get_bc_names()
            bc_mse_losses_init = [float(jnp.mean(residuals_init[name] ** 2)) if name in residuals_init else 0.0 
                                  for name in bc_names]
            bc_losses_str = ", ".join(
                f"{bc_names[i]}: {bc_mse_losses_init[i]:.2e}"
                for i in range(len(bc_names))
            )
            
            self.history['epoch'].append(start_epoch)
            self.history['train_loss'].append(mse_loss_init)
            self.history['loss'].append(al_loss_init)
            if 'al_loss' not in self.history:
                self.history['al_loss'] = []
            self.history['al_loss'].append(al_loss_init)
            if 'mse_loss' not in self.history:
                self.history['mse_loss'] = []
            self.history['mse_loss'].append(mse_loss_init)
            self.history['loss_pde'].append([pde_mse_init])
            self.history['loss_bcs'].append(bc_mse_losses_init)
            
            # Initialize AL-specific history lists for epoch 0
            if 'al_pde_penalty' not in self.history:
                self.history['al_pde_penalty'] = []
                self.history['al_pde_lagrangian'] = []
                self.history['al_bcs_penalty'] = []
                self.history['al_bcs_lagrangian'] = []
            
            # At epoch 0, penalty and lagrangian terms are ~0 (no training yet)
            pde_penalty_init = float(losses_init.get('pde_penalty', 0.0))
            pde_lagrangian_init = float(losses_init.get('pde_lagrangian', 0.0))
            self.history['al_pde_penalty'].append(pde_penalty_init)
            self.history['al_pde_lagrangian'].append(pde_lagrangian_init)
            bc_penalty_init = [float(losses_init.get(f'{name}_penalty', 0.0)) for name in bc_names]
            bc_lagrangian_init = [float(losses_init.get(f'{name}_lagrangian', 0.0)) for name in bc_names]
            self.history['al_bcs_penalty'].append(bc_penalty_init)
            self.history['al_bcs_lagrangian'].append(bc_lagrangian_init)
            
            test_targets = getattr(self, '_test_targets', {})
            if any(s > 0 for s in self.test_samples) and self._test_data:
                _, (_, test_residuals) = compute_al_loss(
                    self.network.params, self._test_data, 
                    {k: jnp.zeros(len(v)) for k, v in self._test_data.items()},
                    {k: 1.0 for k in self._test_data.keys()},
                    test_targets
                )
                test_mse = sum(jnp.mean(g ** 2) for g in test_residuals.values())
                self.history['test_loss'].append(float(test_mse))
            
            if self.problem.solution is not None:
                sol_error = self._compute_solution_error()
                self.history['solution_error'].append(sol_error)
            
            msg = (
                f"Epoch 0/{self._epochs + start_epoch} | "
                f"AL Loss: {al_loss_init:.2e} | "
                f"MSE Loss: {mse_loss_init:.2e} | "
                f"PDE: {pde_mse_init:.2e} | "
                f"BCs: [{bc_losses_str}] | "
                f"Time: 0.0s"
            )
            if self.history['test_loss']:
                msg += f" | Test: {self.history['test_loss'][-1]:.2e}"
            if self.problem.solution is not None:
                msg += f" | Error: {self.history['solution_error'][-1]:.2e}"
            print(msg)
        
        for epoch in range(start_epoch, start_epoch + epochs):
            self._outer_epoch = epoch - start_epoch
            
            # Update learning rate if scheduler is provided
            if lr_scheduler is not None and self.optimizer_name not in ("lbfgs", "soap"):
                new_lr = lr_scheduler.lr(self.learning_rate, epoch)
                if hasattr(self.opt_state, 'hyperparams'):
                    new_hyperparams = dict(self.opt_state.hyperparams)
                    new_hyperparams['learning_rate'] = new_lr
                    self.opt_state = self.opt_state._replace(hyperparams=new_hyperparams)
            
            if resample_each > 0 and epoch > start_epoch and epoch % resample_each == 0:
                self._sample_train_data()
                self._reinitialize_lagrange_if_needed()
            
            if adaptive_sampling and epoch > start_epoch and epoch % adaptive_each == 0:
                self._adaptive_resample(params_dict)
                self._reinitialize_lagrange_if_needed()
            
            # ==================== Primal Optimization Step ====================
            self.network.params, self.opt_state, loss, losses, residuals = train_step(
                self.network.params, self.opt_state, self._train_data, 
                self.lagrange_multipliers, weights_dict, train_targets
            )
            
            # ==================== Dual Update (Lagrange Multipliers) ====================
            self._update_lagrange_multipliers(residuals)
            
            # ==================== Record History and Print ====================
            # Store both AL loss (for training dynamics) and MSE loss (for comparison)
            al_loss_val = float(loss)
            mse_loss = sum(jnp.mean(g ** 2) for g in residuals.values())
            mse_loss_val = float(mse_loss)
            
            self.history['epoch'].append(epoch)
            self.history['loss'].append(mse_loss_val)  # MSE loss for comparison
            self.history['train_loss'].append(mse_loss_val)
            
            # Store AL loss separately
            if 'al_loss' not in self.history:
                self.history['al_loss'] = []
            self.history['al_loss'].append(al_loss_val)
            
            # Store AL loss components (penalty and lagrangian)
            if 'al_pde_penalty' not in self.history:
                self.history['al_pde_penalty'] = []
                self.history['al_pde_lagrangian'] = []
                self.history['al_bcs_penalty'] = []
                self.history['al_bcs_lagrangian'] = []
            
            self.history['al_pde_penalty'].append(float(losses.get('pde_penalty', 0.0)))
            self.history['al_pde_lagrangian'].append(float(losses.get('pde_lagrangian', 0.0)))
            
            bc_names = self._get_bc_names()
            bc_penalty_list = [float(losses.get(f'{name}_penalty', 0.0)) for name in bc_names]
            bc_lagrangian_list = [float(losses.get(f'{name}_lagrangian', 0.0)) for name in bc_names]
            self.history['al_bcs_penalty'].append(bc_penalty_list)
            self.history['al_bcs_lagrangian'].append(bc_lagrangian_list)
            
            # Compute per-constraint MSE losses
            pde_mse = float(jnp.mean(residuals['pde'] ** 2)) if 'pde' in residuals else 0.0
            self.history['loss_pde'].append([pde_mse])
            
            bc_mse_losses = [float(jnp.mean(residuals[name] ** 2)) if name in residuals else 0.0 
                            for name in bc_names]
            self.history['loss_bcs'].append(bc_mse_losses)
            
            if print_each > 0 and ((epoch + 1) % print_each == 0 or epoch == self._epochs + start_epoch - 1):
                test_targets = getattr(self, '_test_targets', {})
                if any(s > 0 for s in self.test_samples) and self._test_data:
                    # Compute test MSE loss (without λ terms)
                    _, (_, test_residuals) = compute_al_loss(
                        self.network.params, self._test_data, 
                        {k: jnp.zeros(len(v)) for k, v in self._test_data.items()},
                        {k: 1.0 for k in self._test_data.keys()},
                        test_targets
                    )
                    test_mse = sum(jnp.mean(g ** 2) for g in test_residuals.values())
                    self.history['test_loss'].append(float(test_mse))
                
                if self.problem.solution is not None:
                    sol_error = self._compute_solution_error()
                    self.history['solution_error'].append(sol_error)
                
                elapsed_time = time.time() - start_time
                
                bc_losses_str = ", ".join(
                    f"{bc_names[i]}: {bc_mse_losses[i]:.2e}"
                    for i in range(len(bc_names))
                )
                
                # Print both AL loss and MSE loss
                msg = (
                    f"Epoch {epoch + 1}/{self._epochs + start_epoch} | "
                    f"AL Loss: {al_loss_val:.2e} | "
                    f"MSE Loss: {mse_loss_val:.2e} | "
                    f"PDE: {pde_mse:.2e} | "
                    f"BCs: [{bc_losses_str}] | "
                    f"Time: {elapsed_time:.1f}s"
                )
                if self.history['test_loss']:
                    msg += f" | Test: {self.history['test_loss'][-1]:.2e}"
                if self.problem.solution is not None:
                    msg += f" | Error: {self.history['solution_error'][-1]:.2e}"
                print(msg)
                
                if show_plots or save_plots:
                    if save_plots:
                        plot_path = f"{save_plots}_epoch{epoch:05d}.png"
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
        print(f"ALTrainer complete in {time.time() - start_time:.1f}s")
        
        # Close figure to prevent duplicate display in notebooks
        if is_notebook() and show_plots and self._fig is not None:
            plt.close(self._fig)
    
    def get_lagrange_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics of Lagrange multipliers.
        
        Returns:
            Dict mapping constraint names to dicts with 'mean', 'std', 'min', 'max'.
        """
        stats = {}
        for name, lam in self.lagrange_multipliers.items():
            stats[name] = {
                'mean': float(jnp.mean(lam)),
                'std': float(jnp.std(lam)),
                'min': float(jnp.min(lam)),
                'max': float(jnp.max(lam)),
            }
        return stats
    
    def reset_lagrange_multipliers(self):
        """Reset all Lagrange multipliers to zero."""
        for name in self.lagrange_multipliers:
            self.lagrange_multipliers[name] = jnp.zeros_like(self.lagrange_multipliers[name])
    
    def _plot_losses(self, ax):
        """Plot AL loss and its components (penalty + lagrangian) on 'losses' axis."""
        import numpy as np
        
        epochs = self.history['epoch']
        if not epochs:
            return
        
        # Total AL Loss
        al_loss_data = self.history.get('al_loss', [])
        if al_loss_data:
            ax.semilogy(epochs, al_loss_data, 'k-', label='AL Total', linewidth=2)
        
        # PDE penalty and lagrangian
        pde_penalty = self.history.get('al_pde_penalty', [])
        pde_lagrangian = self.history.get('al_pde_lagrangian', [])
        if pde_penalty:
            ax.semilogy(epochs, np.abs(pde_penalty) + 1e-20, 'b--', label='PDE penalty', linewidth=1.5)
        if pde_lagrangian:
            ax.semilogy(epochs, np.abs(pde_lagrangian) + 1e-20, 'b:', label='PDE lagrangian', linewidth=1.5)
        
        # BC penalty and lagrangian
        bc_names = self._get_bc_names()
        bc_penalty = self.history.get('al_bcs_penalty', [])
        bc_lagrangian = self.history.get('al_bcs_lagrangian', [])
        
        colors = ['r', 'g', 'c', 'm', 'y', 'tab:orange']
        if bc_penalty and len(bc_penalty) > 0:
            bc_penalty_arr = np.array(bc_penalty)
            bc_lagrangian_arr = np.array(bc_lagrangian) if bc_lagrangian else None
            
            for i in range(bc_penalty_arr.shape[1]):
                color = colors[i % len(colors)]
                bc_label = bc_names[i] if i < len(bc_names) else f'BC{i+1}'
                ax.semilogy(epochs, np.abs(bc_penalty_arr[:, i]) + 1e-20, '--', 
                           color=color, label=f'{bc_label} penalty', linewidth=1.5)
                if bc_lagrangian_arr is not None:
                    ax.semilogy(epochs, np.abs(bc_lagrangian_arr[:, i]) + 1e-20, ':', 
                               color=color, label=f'{bc_label} lagr.', linewidth=1.5)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AL Loss')
        ax.set_title('Augmented Lagrangian Loss (penalty + lagrangian)')
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7)
        ax.grid(True, alpha=0.3)
    
    def _plot_mse_losses(self, ax):
        """Plot MSE losses for comparison on 'mse_losses' axis."""
        import numpy as np
        
        epochs = self.history['epoch']
        if not epochs:
            return
        
        # Total MSE loss
        loss_data = self.history.get('loss', self.history.get('train_loss', []))
        if loss_data:
            ax.semilogy(epochs, loss_data, 'k-', label='MSE Total', linewidth=2)
        
        # PDE MSE
        pde_losses = self.history.get('loss_pde', [])
        if len(pde_losses) > 0:
            if isinstance(pde_losses[0], (list, tuple)):
                pde_array = np.array(pde_losses)
                for i in range(pde_array.shape[1]):
                    ax.semilogy(epochs, pde_array[:, i], 'b--', label=f'PDE eq{i+1}')
            else:
                ax.semilogy(epochs, pde_losses, 'b--', label='PDE')
        
        # BC MSE
        bc_names = self._get_bc_names()
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
            ax.semilogy(test_epochs, test_loss, 'r:', marker='o', markersize=4, label='Test MSE', linewidth=2)
        
        # Solution error
        sol_error = self.history.get('solution_error', [])
        if len(sol_error) > 0:
            n_err = len(sol_error)
            err_epochs = np.linspace(epochs[0], epochs[-1], n_err).astype(int) if n_err > 1 else [epochs[-1]]
            ax.semilogy(err_epochs, sol_error, 'm-', marker='s', markersize=4, label='Solution Error', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title('MSE Losses (for comparison with other methods)')
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7)
        ax.grid(True, alpha=0.3)

    def reset_betas(self, betas: dict = None):
        """
        Reset penalty parameters.
        
        Args:
            betas: New beta values. If None, reset to 1.0.
        """
        if betas is None:
            for name in self.betas:
                self.betas[name] = 1.0
        else:
            for name, val in betas.items():
                if name in self.betas:
                    self.betas[name] = val

    