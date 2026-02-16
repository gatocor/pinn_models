"""
JAX/Optax implementation of PINN Trainer.

Inherits common functionality from BaseTrainer.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from functools import partial
import time
import inspect

try:
    import jaxopt
    HAS_JAXOPT = True
except ImportError:
    HAS_JAXOPT = False

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
        """Create the optimizer."""
        if self.optimizer_name == "adam":
            return optax.adam(self.learning_rate)
        elif self.optimizer_name == "sgd":
            return optax.sgd(self.learning_rate)
        elif self.optimizer_name == "rmsprop":
            return optax.rmsprop(self.learning_rate)
        elif self.optimizer_name == "lbfgs":
            if not HAS_JAXOPT:
                raise ImportError(
                    "L-BFGS requires jaxopt. Install with: pip install jaxopt"
                )
            # Return None - L-BFGS solver is created per training run
            return None
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
        # Call parent to sample train/test data
        super()._after_compile_hook()
        
        # Precompute sparse FBPINN data if network is FBPINN
        # Skip when batching is enabled (indices don't match batch slices)
        use_batching = self._batch_size is not None and self._batch_size > 0
        if self._use_sparse_fbpinn and isinstance(self.network, FBPINN) and not use_batching:
            self._precompute_sparse_data()
        elif use_batching:
            # Clear any precomputed sparse data when batching
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
        
        def compute_loss(params, train_data):
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
                
                if bc_data['const_value'] is not None:
                    target = bc_data['const_value']
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
                
                if bc_data['const_value'] is not None:
                    target = bc_data['const_value']
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
            def train_step(params, opt_state, train_data):
                loss, grads = jax.value_and_grad(compute_loss)(params, train_data)
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
        
        for epoch in range(epochs):
            epoch_start = time.time()
            global_epoch = start_epoch + epoch
            
            if use_batching:
                # Mini-batch training
                epoch_loss = 0.0
                for batch_idx in range(n_batches):
                    # Create batch data
                    batch_data = {}
                    for name, data in self._train_data.items():
                        n_points = len(data)
                        start_idx, end_idx = self._get_batch_indices(n_points, batch_idx, n_batches)
                        batch_data[name] = data[start_idx:end_idx]
                    
                    if is_full_jit:
                        self.network.params, self.opt_state, loss = train_step(
                            self.network.params, self.opt_state, batch_data
                        )
                    else:
                        loss, grads = grad_fn(self.network.params, batch_data)
                        self.network.params, self.opt_state = apply_updates(self.network.params, grads, self.opt_state)
                    
                    epoch_loss += float(loss)
                
                loss = epoch_loss / n_batches
            else:
                # Full-batch training
                if is_full_jit:
                    self.network.params, self.opt_state, loss = train_step(
                        self.network.params, self.opt_state, self._train_data
                    )
                else:
                    loss, grads = grad_fn(self.network.params, self._train_data)
                    self.network.params, self.opt_state = apply_updates(self.network.params, grads, self.opt_state)
            
            epoch_time = time.time() - epoch_start
            self.history['epoch_times'].append(epoch_time)
            
            if print_each > 0 and (epoch % print_each == 0 or epoch == epochs - 1):
                elapsed = time.time() - start_time
                
                _, individual_losses = self._compute_total_loss(self._train_data, params_dict, weights)
                pde_loss = float(individual_losses.get('pde', 0.0))
                bc_losses = [float(individual_losses[name]) for name in self._train_data.keys() if name != 'pde']
                
                self.history['epoch'].append(global_epoch)
                self.history['train_loss'].append(float(loss))
                self.history['loss'].append(float(loss))
                self.history['loss_pde'].append(pde_loss)
                self.history['loss_bcs'].append(bc_losses)
                
                # Test loss (if test data available)
                if any(s > 0 for s in self.test_samples) and self._test_data:
                    test_weights = {k: 1.0 for k in self._test_data.keys()}
                    test_total, _ = self._compute_total_loss(self._test_data, params_dict, test_weights)
                    self.history['test_loss'].append(float(test_total))
                
                if self.problem.solution is not None:
                    sol_error = self._compute_solution_error()
                    self.history['solution_error'].append(sol_error)
                
                bc_names = self._get_bc_names()
                bc_losses_str = ", ".join(
                    f"{name}: {individual_losses.get(name, 0.0):.2e}" 
                    for name in bc_names
                )
                
                msg = (f"Epoch {epoch}/{epochs + start_epoch} | "
                       f"Loss: {loss:.6f} | "
                       f"PDE: {pde_loss:.2e} | "
                       f"BCs: [{bc_losses_str}] | "
                       f"Time: {elapsed:.1f}s")
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
        print(f"Training complete in {time.time() - start_time:.1f}s")
    
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
        
        # Initialize solver state
        state = solver.init_state(self.network.params, self._train_data)
        
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
            
            if print_each > 0 and (epoch % print_each == 0 or epoch == epochs - 1):
                elapsed = time.time() - start_time
                
                _, individual_losses = self._compute_total_loss(self._train_data, params_dict, weights)
                pde_loss = float(individual_losses.get('pde', 0.0))
                bc_losses = [float(individual_losses[name]) for name in self._train_data.keys() if name != 'pde']
                
                self.history['epoch'].append(global_epoch)
                self.history['train_loss'].append(float(loss))
                self.history['loss'].append(float(loss))
                self.history['loss_pde'].append(pde_loss)
                self.history['loss_bcs'].append(bc_losses)
                
                # Test loss
                if any(s > 0 for s in self.test_samples) and self._test_data:
                    test_weights = {k: 1.0 for k in self._test_data.keys()}
                    test_total, _ = self._compute_total_loss(self._test_data, params_dict, test_weights)
                    self.history['test_loss'].append(float(test_total))
                
                if self.problem.solution is not None:
                    sol_error = self._compute_solution_error()
                    self.history['solution_error'].append(sol_error)
                
                bc_names = self._get_bc_names()
                bc_losses_str = ", ".join(
                    f"{name}: {individual_losses.get(name, 0.0):.2e}" 
                    for name in bc_names
                )
                
                msg = (f"Epoch {epoch}/{epochs + start_epoch} | "
                       f"Loss: {loss:.6f} | "
                       f"PDE: {pde_loss:.2e} | "
                       f"BCs: [{bc_losses_str}] | "
                       f"Time: {elapsed:.1f}s")
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
        print(f"L-BFGS training complete in {time.time() - start_time:.1f}s")
    
    def _make_lbfgs_loss_fn(self, weights, params_dict):
        """Create a loss function suitable for L-BFGS optimization."""
        # Similar to _make_jit_train_step but returns only loss function
        from .networks import FBPINN
        
        # Pre-extract BC info
        dirichlet_bcs = []
        neumann_bcs = []
        bc_names = self._get_bc_names()
        
        for i, bc in enumerate(self.problem.boundary_conditions):
            from pinns.boundary import DirichletBC, NeumannBC, RobinBC
            name = bc_names[i]
            is_neumann = isinstance(bc, (NeumannBC, RobinBC))
            
            bc_data = {
                'name': name,
                'component': bc.component,
                'weight': weights.get(name, 1.0),
                'const_value': bc.value if not callable(bc.value) else None,
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
                
                target = bc_data['const_value'] if bc_data['const_value'] is not None else 0.0
                bc_loss = jnp.mean((y_bc[:, bc_data['component']] - target) ** 2)
                total_loss = total_loss + bc_data['weight'] * bc_loss
            
            # Neumann BC Loss
            for bc_data in neumann_bcs:
                bc_name = bc_data['name']
                x_bc = train_data[bc_name]
                dim = bc_data['dim']
                component = bc_data['component']
                normal_sign = bc_data['normal_sign']
                target = bc_data['const_value'] if bc_data['const_value'] is not None else 0.0
                
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
    