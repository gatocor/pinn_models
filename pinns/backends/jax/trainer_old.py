"""
JAX/Optax implementation of PINN Trainer.

API compatible with PyTorch backend.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import Dict, List, Tuple, Callable, Optional, Any
from functools import partial
import time
import matplotlib.pyplot as plt

from .functional import set_context, clear_context
from .networks import FBPINN, FNN

def _is_notebook():
    """Check if we're running in a Jupyter notebook."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        return False
    except:
        return False

class Trainer:
    """
    JAX-based trainer for Physics-Informed Neural Networks.
    
    API compatible with PyTorch backend - same method signatures.
    
    Example:
        # Same API as PyTorch:
        trainer = Trainer(problem, model)
        trainer.compile(train_samples={'pde': 1000}, epochs=10000)
        trainer.train()  # No need to pass params
    """
    
    def __init__(self, problem, model):
        """
        Initialize trainer.
        
        Args:
            problem: Problem instance defining PDE and boundary conditions
            model: FBPINN or FNN model instance (Flax module)
        """
        self.problem = problem
        self.model = model
        self.history = {
            'train_loss': [], 
            'test_loss': [], 
            'epoch_times': [],
            'epoch': [],
            'loss_pde': [],
            'loss_bcs': [],
            'solution_error': []
        }
        
        # Initialize model parameters with dummy input
        dummy_input = jnp.ones((1, problem.n_dims))
        self.params = model.init(jax.random.PRNGKey(0), dummy_input)
        
        # Training configuration
        self.train_samples = None
        self.test_samples = None
        self.weights = None
        self.optimizer = None
        self.opt_state = None
        self.learning_rate = 1e-3
        self.epochs = 1000
        self._batch_size = None
        self.print_each = 100
        
        # Sampled data
        self._train_data = None
        self._test_data = None
        
        # Plotting
        self._show_plots = False
        self._fig = None
        self._axes = None
        self._display_handle = None
        self._colorbars = []
        self._global_epoch = 0
        
        # Device (for API compatibility)
        self.device = 'cpu'  # JAX handles device placement automatically
    
    def compile(self,
                train_samples: Dict[str, int],
                test_samples: Optional[Dict[str, int]] = None,
                weights: Optional[Dict[str, float]] = None,
                optimizer: str = 'adam',
                learning_rate: float = 1e-3,
                epochs: int = 1000,
                batch_size: Optional[int] = None,
                print_each: int = 100,
                show_plots: bool = False,
                **kwargs):
        """
        Configure training parameters.
        
        Same API as PyTorch backend.
        """
        self.train_samples = train_samples
        self.test_samples = test_samples or {}
        self.weights = weights or {k: 1.0 for k in train_samples}
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._batch_size = batch_size
        self._show_plots = show_plots
        self.print_each = print_each
        
        # Create optimizer
        if optimizer.lower() == 'adam':
            self.optimizer = optax.adam(learning_rate)
        elif optimizer.lower() == 'sgd':
            self.optimizer = optax.sgd(learning_rate)
        elif optimizer.lower() == 'rmsprop':
            self.optimizer = optax.rmsprop(learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        # Initialize optimizer state
        self.opt_state = self.optimizer.init(self.params)
        
        # Set input range from domain bounds
        if hasattr(self.model, 'set_input_range'):
            xmin = np.array(self.problem.xmin)
            xmax = np.array(self.problem.xmax)
            self.model.set_input_range(xmin, xmax)
        
        # Set output range from problem definition
        if hasattr(self.model, 'set_output_range'):
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
            self.model.set_output_range(ymin, ymax)
        
        # Sample initial data
        self._sample_train_data()
        if self.test_samples:
            self._sample_test_data()
    
    def _build_params_dict(self) -> Dict:
        """Build params dict for transforms."""
        return {'fixed': self.problem.params}
    
    def _sample_points(self, name: str, n_samples: int) -> jnp.ndarray:
        """Sample points for a given loss term."""
        params_dict = {
            "fixed": self.problem.params,
            "infer": {},
            "internal": {}
        }
        
        if name == 'pde':
            points = self.problem.domain.sample_interior(n_samples, params=params_dict)
        else:
            bc = self._get_bc_by_name(name)
            if bc is not None:
                points = self._sample_boundary(bc, n_samples, params_dict)
            else:
                raise ValueError(f"Unknown loss term: {name}")
        return jnp.array(points)
    
    def _sample_boundary(self, bc, n_points: int, params_dict: dict) -> np.ndarray:
        """Sample boundary points for a specific boundary condition."""
        from pinns.boundary import PointsetBC
        
        if isinstance(bc, PointsetBC):
            return bc.points
        
        # Get boundary specification and domain
        boundary = bc.boundary
        domain = self.problem.domain
        
        # Get sampling method and transform from BC (or defaults)
        method = getattr(bc, 'sampling_method', 'uniform')
        transform = getattr(bc, 'sampling_transform', None)
        
        # Find which dimension and side this boundary is on
        for dim, side in enumerate(boundary):
            if side is not None:
                points = domain.sample_boundary(
                    n_points, dim, side,
                    method=method, transform=transform, params=params_dict
                )
                
                # Apply subdomain constraint if specified
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
    
    def _get_bc_by_name(self, name: str):
        """Get boundary condition by name."""
        for bc in self.problem.boundary_conditions:
            if hasattr(bc, 'name') and bc.name == name:
                return bc
        return None
    
    def _sample_train_data(self):
        """Sample training data."""
        self._train_data = {}
        for name, n in self.train_samples.items():
            self._train_data[name] = self._sample_points(name, n)
    
    def _sample_test_data(self):
        """Sample test data."""
        self._test_data = {}
        for name, n in self.test_samples.items():
            self._test_data[name] = self._sample_points(name, n)
    
    def _apply_fn(self, params, x):
        """Network apply function for derivatives."""
        return self.model.apply(params, x)
    
    def _compute_pde_loss(self, params, x_pde, params_dict):
        """Compute PDE residual loss with context set for derivatives."""
        import inspect
        
        # Check if model accepts params_dict
        model_sig = inspect.signature(self.model.apply)
        model_accepts_params_dict = len(model_sig.parameters) >= 3
        
        if model_accepts_params_dict:
            model_apply = lambda p, x: self.model.apply(p, x, params_dict)
        else:
            model_apply = self.model.apply
        
        # Set context so derivative() works with same API as PyTorch
        set_context(model_apply, params)
        
        try:
            # Forward pass
            y = model_apply(params, x_pde)
            
            # Compute residual - PDE function uses same API as PyTorch
            residual = self.problem.pde_fn(x_pde, y, params_dict)
            
            # Handle multiple residuals
            if isinstance(residual, (list, tuple)):
                total_res = 0.0
                for r in residual:
                    total_res = total_res + jnp.mean(r**2)
                return total_res / len(residual)
            else:
                return jnp.mean(residual**2)
        finally:
            clear_context()
    
    def _get_bc_value(self, bc, x_bc):
        """Get boundary condition value (JAX-compatible)."""
        if callable(bc.value):
            # Call the function
            result = bc.value(x_bc)
            if not isinstance(result, jnp.ndarray):
                result = jnp.array(result)
            return result
        else:
            # Constant value - create array
            return jnp.full((x_bc.shape[0],), bc.value)
    
    def _compute_bc_loss(self, params, x_bc, bc, params_dict):
        """Compute boundary condition loss."""
        from pinns.boundary import DirichletBC, NeumannBC, RobinBC, PointsetBC
        import inspect
        
        # Check if model accepts params_dict
        model_sig = inspect.signature(self.model.apply)
        model_accepts_params_dict = len(model_sig.parameters) >= 3
        
        if model_accepts_params_dict:
            y = self.model.apply(params, x_bc, params_dict)
            model_apply = lambda p, x: self.model.apply(p, x, params_dict)
        else:
            y = self.model.apply(params, x_bc)
            model_apply = self.model.apply
        
        if isinstance(bc, DirichletBC):
            target = self._get_bc_value(bc, x_bc)
            return jnp.mean((y[:, bc.component] - target) ** 2)
        
        elif isinstance(bc, NeumannBC):
            normal_dim, normal_sign = bc.get_normal_direction()
            
            # Compute gradient using forward-mode AD (jvp) for efficiency
            n_dims = x_bc.shape[1]
            eye = jnp.eye(n_dims)
            
            def single_grad_fwd(xi):
                def scalar_output(x):
                    return model_apply(params, x.reshape(1, -1))[0, bc.component]
                # Use jvp with unit tangent vector for the normal direction
                _, du_dn = jax.jvp(scalar_output, (xi,), (eye[normal_dim],))
                return du_dn
            
            du_dn = jax.vmap(single_grad_fwd)(x_bc)
            target = self._get_bc_value(bc, x_bc)
            return jnp.mean((normal_sign * du_dn - target) ** 2)
        
        elif isinstance(bc, RobinBC):
            normal_dim, normal_sign = bc.get_normal_direction()
            
            # Use forward-mode AD (jvp) for efficiency
            n_dims = x_bc.shape[1]
            eye = jnp.eye(n_dims)
            
            def single_grad_fwd(xi):
                def scalar_output(x):
                    return model_apply(params, x.reshape(1, -1))[0, bc.component]
                _, du_dn = jax.jvp(scalar_output, (xi,), (eye[normal_dim],))
                return du_dn
            
            du_dn = jax.vmap(single_grad_fwd)(x_bc)
            
            alpha, beta, gamma = bc.get_coefficients(x_bc)
            residual = alpha * y[:, bc.component] + beta * normal_sign * du_dn - gamma
            return jnp.mean(residual ** 2)
        
        elif isinstance(bc, PointsetBC):
            target = jnp.array(bc.values)
            return jnp.mean((y[:, bc.component] - target) ** 2)
        
        else:
            raise ValueError(f"Unknown BC type: {type(bc)}")
    
    def _loss_fn(self, params, data, weights, params_dict):
        """Compute total weighted loss."""
        total_loss = 0.0
        losses = {}
        
        # PDE loss
        if 'pde' in data:
            pde_loss = self._compute_pde_loss(params, data['pde'], params_dict)
            losses['pde'] = pde_loss
            total_loss = total_loss + weights.get('pde', 1.0) * pde_loss
        
        # BC losses
        for name, x_bc in data.items():
            if name == 'pde':
                continue
            bc = self._get_bc_by_name(name)
            if bc is not None:
                bc_loss = self._compute_bc_loss(params, x_bc, bc, params_dict)
                losses[name] = bc_loss
                total_loss = total_loss + weights.get(name, 1.0) * bc_loss
        
        return total_loss, losses
    
    def _make_jit_train_step(self, weights, params_dict):
        """Create a fully JIT-compiled training step function.
        
        For full JIT compilation, the PDE function must accept a 4th argument:
        the derivative function. This is the JAX-idiomatic approach.
        """
        from .functional import make_derivative_fn
        import inspect
        
        # Pre-extract BC info as static data
        bc_info = []
        for name in self._train_data.keys():
            if name == 'pde':
                continue
            bc = self._get_bc_by_name(name)
            if bc is not None:
                from pinns.boundary import NeumannBC
                is_neumann = isinstance(bc, NeumannBC)
                normal_info = bc.get_normal_direction() if is_neumann else (0, 1)
                bc_info.append({
                    'name': name,
                    'component': bc.component,
                    'is_neumann': is_neumann,
                    'normal_dim': normal_info[0],
                    'normal_sign': normal_info[1],
                    'const_value': bc.value if not callable(bc.value) else None,
                })
        
        model_apply = self.model.apply
        pde_fn = self.problem.pde_fn
        
        # Check if model.apply accepts params_dict (FBPINN does, FNN doesn't)
        # Must be checked outside compute_loss for JIT compatibility
        model_sig = inspect.signature(self.model.apply)
        model_accepts_params_dict = len(model_sig.parameters) >= 3
        
        # Wrap model_apply to include params_dict for output_transform support
        if model_accepts_params_dict:
            def model_apply_with_params(params, x):
                return self.model.apply(params, x, params_dict)
        else:
            def model_apply_with_params(params, x):
                return self.model.apply(params, x)
        
        # Check if PDE function accepts derivative argument
        sig = inspect.signature(pde_fn)
        pde_accepts_derivative = len(sig.parameters) >= 4
        
        def compute_loss(params, train_data):
            total_loss = 0.0
            
            # PDE loss
            if 'pde' in train_data:
                x_pde = train_data['pde']
                y = model_apply_with_params(params, x_pde)
                
                # Create derivative function for current params (JIT-traceable)
                deriv_fn = make_derivative_fn(model_apply_with_params, params)
                
                # Call PDE with or without derivative argument
                if pde_accepts_derivative:
                    residual = pde_fn(x_pde, y, params_dict, deriv_fn)
                else:
                    # Fallback: inject derivative via module (not JIT-compatible)
                    set_context(model_apply, params)
                    try:
                        residual = pde_fn(x_pde, y, params_dict)
                    finally:
                        clear_context()
                
                if isinstance(residual, (list, tuple)):
                    pde_loss = sum(jnp.mean(r**2) for r in residual) / len(residual)
                else:
                    pde_loss = jnp.mean(residual**2)
                total_loss = total_loss + weights.get('pde', 1.0) * pde_loss
            
            # BC losses (fully JIT-compatible)
            for bc_data in bc_info:
                x_bc = train_data[bc_data['name']]
                y = model_apply_with_params(params, x_bc)
                
                if bc_data['const_value'] is not None:
                    target = jnp.full((x_bc.shape[0],), bc_data['const_value'])
                else:
                    target = jnp.zeros(x_bc.shape[0])
                
                if bc_data['is_neumann']:
                    comp = bc_data['component']
                    n_dims = x_bc.shape[1]
                    eye = jnp.eye(n_dims)
                    normal_dim = bc_data['normal_dim']
                    
                    # Use forward-mode AD (jvp) for efficiency
                    def make_single_grad_fwd(component):
                        def single_grad_fwd(xi):
                            def f(x):
                                return model_apply_with_params(params, x.reshape(1, -1))[0, component]
                            _, du_dn = jax.jvp(f, (xi,), (eye[normal_dim],))
                            return du_dn
                        return single_grad_fwd
                    
                    du_dn = jax.vmap(make_single_grad_fwd(comp))(x_bc)
                    bc_loss = jnp.mean((bc_data['normal_sign'] * du_dn - target) ** 2)
                else:
                    bc_loss = jnp.mean((y[:, bc_data['component']] - target) ** 2)
                
                total_loss = total_loss + weights.get(bc_data['name'], 1.0) * bc_loss
            
            return total_loss
        
        # JIT compile the full training step IF PDE accepts derivative arg
        if pde_accepts_derivative:
            @jax.jit
            def train_step(params, opt_state, train_data):
                loss, grads = jax.value_and_grad(compute_loss)(params, train_data)
                updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return new_params, new_opt_state, loss
            
            return train_step, True  # True = fully JIT
        else:
            # Non-JIT path for backward compatibility
            grad_fn = jax.value_and_grad(compute_loss)
            
            @jax.jit
            def apply_updates(params, grads, opt_state):
                updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return new_params, new_opt_state
            
            return (grad_fn, apply_updates), False  # False = partial JIT
    
    def train(self):
        """
        Train the model.
        
        Same API as PyTorch - no need to pass params.
        
        For full JIT compilation with JAX, define your PDE function with 4 arguments:
            def my_pde(X, U, params, derivative):
                u_x = derivative(U, X, 0, (0,))
                ...
        """
        params_dict = self._build_params_dict()
        weights = {k: float(v) for k, v in self.weights.items()}
        
        # Create training step
        result, is_full_jit = self._make_jit_train_step(weights, params_dict)
        
        if is_full_jit:
            train_step = result
            print(f"Starting training for {self.epochs} epochs (JIT-compiled)...")
        else:
            grad_fn, apply_updates = result
            print(f"Starting training for {self.epochs} epochs...")
            print("Note: For faster training, define PDE with 4th 'derivative' argument")
        
        start_time = time.time()
        start_epoch = self._global_epoch
        show_plots = self._show_plots
        
        # Setup figure for live plotting in notebooks
        if show_plots and _is_notebook():
            if self._fig is None:
                self._fig, self._axes = self._create_figure()
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            global_epoch = start_epoch + epoch
            
            if is_full_jit:
                self.params, self.opt_state, loss = train_step(
                    self.params, self.opt_state, self._train_data
                )
            else:
                loss, grads = grad_fn(self.params, self._train_data)
                self.params, self.opt_state = apply_updates(self.params, grads, self.opt_state)
            
            # Record basic history (fast)
            epoch_time = time.time() - epoch_start
            self.history['epoch_times'].append(epoch_time)
            
            # Print progress and plot (at epoch 0 and every print_each epochs)
            if epoch % self.print_each == 0:
                elapsed = time.time() - start_time
                
                # Compute individual losses for display (only at print intervals - slow)
                _, individual_losses = self._loss_fn(self.params, self._train_data, weights, params_dict)
                pde_loss = float(individual_losses.get('pde', 0.0))
                bc_losses = [float(individual_losses[name]) for name in self._train_data.keys() if name != 'pde']
                
                # Record detailed history at print intervals
                self.history['epoch'].append(global_epoch)
                self.history['train_loss'].append(float(loss))
                self.history['loss_pde'].append(pde_loss)
                self.history['loss_bcs'].append(bc_losses)
                
                # Compute solution error if available
                if self.problem.solution is not None:
                    sol_error = self._compute_solution_error()
                    self.history['solution_error'].append(sol_error)
                
                bc_losses_str = ", ".join(
                    f"{name}: {individual_losses[name]:.2e}" 
                    for name in self._train_data.keys() if name != 'pde'
                )
                
                msg = (f"Epoch {epoch}/{self.epochs} | "
                       f"Loss: {loss:.6f} | "
                       f"PDE: {pde_loss:.2e} | "
                       f"BCs: [{bc_losses_str}] | "
                       f"Time: {elapsed:.1f}s")
                if self.problem.solution is not None:
                    msg += f" | Error: {self.history['solution_error'][-1]:.2e}"
                print(msg)
                
                # Generate plots
                if show_plots:
                    _, _, self._display_handle = self.plot_progress(
                        fig=self._fig, axes=self._axes, 
                        display_handle=self._display_handle
                    )
        
        self._global_epoch += self.epochs
        print(f"Training complete in {time.time() - start_time:.1f}s")
    
    def predict(self, x: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Make predictions.
        
        Same API as PyTorch - no need to pass params.
        """
        x_jax = jnp.array(x)
        
        if batch_size is None:
            batch_size = self._batch_size
        
        # Build params_dict for output_transform support
        params_dict = self._build_params_dict()
        
        # Check if model accepts params_dict
        import inspect
        model_sig = inspect.signature(self.model.apply)
        model_accepts_params_dict = len(model_sig.parameters) >= 3
        
        def apply_model(x):
            if model_accepts_params_dict:
                return self.model.apply(self.params, x, params_dict)
            return self.model.apply(self.params, x)
        
        if batch_size is None or batch_size >= len(x):
            y = apply_model(x_jax)
            return np.array(y)
        else:
            results = []
            for start in range(0, len(x), batch_size):
                end = min(start + batch_size, len(x))
                x_batch = x_jax[start:end]
                y_batch = apply_model(x_batch)
                results.append(np.array(y_batch))
            return np.vstack(results)
    
    def get_history(self) -> Dict:
        """Get training history."""
        return self.history
    
    # ============== Plotting Methods ==============
    
    def _get_bc_names(self) -> List[str]:
        """Get list of boundary condition names."""
        names = []
        for i, bc in enumerate(self.problem.boundary_conditions):
            if hasattr(bc, 'name') and bc.name is not None:
                names.append(bc.name)
            else:
                names.append(f'bc_{i}')
        return names
    
    def _build_params(self) -> Dict:
        """Build params dict for solution evaluation."""
        return {
            'fixed': self.problem.params,
            'infer': {},
            'internal': {'step': self._global_epoch}
        }
    
    def _get_output_name(self, idx: int) -> str:
        """Get output variable name."""
        if hasattr(self.problem, 'output_names') and self.problem.output_names:
            return self.problem.output_names[idx]
        return f'u_{idx}'
    
    def _get_input_name(self, idx: int) -> str:
        """Get input variable name."""
        if hasattr(self.problem, 'input_names') and self.problem.input_names:
            return self.problem.input_names[idx]
        return f'x_{idx}'
    
    def _compute_solution_error(self) -> float:
        """Compute L2 error against true solution if available."""
        if self.problem.solution is None:
            return 0.0
        
        # Sample points
        n_points = 1000
        x = np.random.uniform(
            self.problem.xmin, 
            self.problem.xmax, 
            (n_points, self.problem.n_dims)
        )
        
        # Get predictions
        y_pred = self.predict(x)
        
        # Get true solution
        y_true = self.problem.solution(x, self._build_params())
        if isinstance(y_true, (list, tuple)):
            y_true = np.column_stack(y_true)
        elif y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        
        # L2 relative error
        error = np.sqrt(np.mean((y_pred - y_true)**2)) / (np.sqrt(np.mean(y_true**2)) + 1e-10)
        return float(error)
    
    def _create_figure(self):
        """Create figure and axes for plotting."""
        n_dims = self.problem.n_dims
        n_outputs = self.problem.n_outputs
        has_solution = self.problem.solution is not None
        
        # For 1D: losses, then for each output: solution + residuals + error (if solution exists)
        if n_dims == 1:
            if has_solution:
                n_cols = 3  # solution, residuals, error
            else:
                n_cols = 2  # solution, residuals
            
            fig = plt.figure(figsize=(5 * n_cols, 3.5 * (1 + n_outputs)))
            gs = fig.add_gridspec(1 + n_outputs, n_cols)
            
            axes = {}
            axes['losses'] = fig.add_subplot(gs[0, :])
            
            for i in range(n_outputs):
                axes[f'sol_{i}'] = fig.add_subplot(gs[1 + i, 0])
                axes[f'res_{i}'] = fig.add_subplot(gs[1 + i, 1])
                if has_solution:
                    axes[f'err_{i}'] = fig.add_subplot(gs[1 + i, 2])
        else:
            # For 2D+: simplified layout
            fig = plt.figure(figsize=(12, 4))
            axes = {'losses': fig.add_subplot(1, 1, 1)}
        
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
    
    def _plot_losses(self, ax):
        """Plot loss curves."""
        epochs = self.history['epoch']
        if not epochs:
            return
        
        ax.semilogy(epochs, self.history['train_loss'], 'k-', label='Total', linewidth=2)
        
        # PDE loss
        if self.history['loss_pde']:
            ax.semilogy(epochs, self.history['loss_pde'], '--', label='PDE')
        
        # BC losses
        bc_names = self._get_bc_names()
        bc_losses = np.array(self.history['loss_bcs']) if self.history['loss_bcs'] else None
        if bc_losses is not None and len(bc_losses) > 0:
            for i in range(bc_losses.shape[1]):
                label = bc_names[i] if i < len(bc_names) else f'BC {i+1}'
                ax.semilogy(epochs, bc_losses[:, i], '--', label=label)
        
        # Solution error
        if self.history['solution_error']:
            ax.semilogy(epochs, self.history['solution_error'], 'm-', marker='s', 
                       markersize=4, label='Solution Error', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_solution_1d(self, ax, output_idx, n_points=200):
        """Plot 1D solution."""
        x = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points).reshape(-1, 1)
        y = self.predict(x)
        
        if self.problem.solution is not None:
            y_true = self.problem.solution(x, self._build_params())
            if isinstance(y_true, (list, tuple)):
                y_true = np.column_stack(y_true)
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
    
    def _plot_residuals_1d(self, ax, output_idx, n_points=200):
        """Plot 1D PDE residuals."""
        from .functional import make_derivative_fn
        
        x_np = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points).reshape(-1, 1)
        x_jax = jnp.array(x_np)
        
        params_dict = self._build_params_dict()
        
        # Check if model accepts params_dict
        import inspect
        model_sig = inspect.signature(self.model.apply)
        model_accepts_params_dict = len(model_sig.parameters) >= 3
        
        if model_accepts_params_dict:
            y = self.model.apply(self.params, x_jax, params_dict)
            model_apply_fn = lambda p, x: self.model.apply(p, x, params_dict)
        else:
            y = self.model.apply(self.params, x_jax)
            model_apply_fn = self.model.apply
        
        # Compute residual using the derivative function
        deriv_fn = make_derivative_fn(model_apply_fn, self.params)
        
        sig = inspect.signature(self.problem.pde_fn)
        if len(sig.parameters) >= 4:
            residual = self.problem.pde_fn(x_jax, y, params_dict, deriv_fn)
        else:
            set_context(model_apply_fn, self.params)
            try:
                residual = self.problem.pde_fn(x_jax, y, params_dict)
            finally:
                clear_context()
        
        if isinstance(residual, (list, tuple)):
            if output_idx < len(residual):
                res = np.array(residual[output_idx]).flatten()
            else:
                res = np.zeros(n_points)
        else:
            res = np.array(residual).flatten()
        
        ax.plot(x_np, np.abs(res), 'm-', linewidth=2)
        
        output_name = self._get_output_name(output_idx)
        input_name = self._get_input_name(0)
        ax.set_xlabel(input_name)
        ax.set_ylabel(f'|Residual| ({output_name})')
        ax.set_title(f'PDE Residual ({output_name})')
        ax.grid(True, alpha=0.3)
    
    def _plot_error_1d(self, ax, output_idx, n_points=200):
        """Plot 1D absolute error."""
        x = np.linspace(self.problem.xmin[0], self.problem.xmax[0], n_points).reshape(-1, 1)
        y = self.predict(x)
        
        y_true = self.problem.solution(x, self._build_params())
        if isinstance(y_true, (list, tuple)):
            y_true = np.column_stack(y_true)
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
        
        if _is_notebook() and save_path is None:
            from IPython.display import display, update_display
            if display_handle is None:
                display_handle = display(fig, display_id=True)
            else:
                display_handle.update(fig)
        elif save_path is None:
            plt.show()
        
        return fig, axes, display_handle
