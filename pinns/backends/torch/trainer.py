"""
PyTorch implementation of PINN Trainer.

Inherits common functionality from BaseTrainer.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Union, Any
import time

from ..base_trainer import BaseTrainer, is_notebook
from pinns.problem import Problem
from pinns.boundary import DirichletBC, NeumannBC, RobinBC, PointsetBC
from .functional import derivative
from .networks import FBPINN


class Trainer(BaseTrainer):
    """
    PyTorch-based trainer for Physics-Informed Neural Networks.
    
    Inherits plotting, history management, and utilities from BaseTrainer.
    Implements PyTorch-specific training loop and autodiff.
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
            network: The neural network to train (e.g., FBPINN, VanillaNetwork).
            device: Device to use ('cpu', 'cuda', 'mps'). Default: auto-detect.
        """
        # Initialize base class (handles network.to(), normalization, defaults)
        super().__init__(problem, network, device)
        
        # PyTorch-specific: ensure dtype is set
        self.dtype = torch.float32
    
    # ==================== Device Detection ====================
    
    def _auto_detect_device(self) -> str:
        """Auto-detect the best available device using PyTorch."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    # ==================== Optimizer ====================
    
    def _create_optimizer(self):
        """Create the optimizer."""
        if self.optimizer_name == "adam":
            return torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "lbfgs":
            # Use L-BFGS parameters from compile()
            max_iter = getattr(self, '_lbfgs_max_iter', 5)
            history_size = getattr(self, '_lbfgs_history_size', 50)
            tolerance = getattr(self, '_lbfgs_tolerance', 1e-9)
            line_search = getattr(self, '_lbfgs_line_search', 'strong_wolfe')
            
            return torch.optim.LBFGS(
                self.network.parameters(),
                lr=self.learning_rate,
                max_iter=max_iter,
                history_size=history_size,
                line_search_fn=line_search,
                tolerance_grad=tolerance,
                tolerance_change=tolerance * 1e-3
            )
        elif self.optimizer_name == "sgd":
            return torch.optim.SGD(self.network.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
    
    def _init_optimizer_state(self):
        """PyTorch optimizers don't need separate state initialization."""
        pass
    
    # ==================== Sparse FBPINN Precomputation ====================
    
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
        
        # Precompute for PDE data
        if 'pde' in self._train_data:
            x_pde = self._train_data['pde']
            self._precomputed_pde = self.network.precompute_training_data(
                x_pde, threshold=self._sparse_threshold, params=params_dict
            )
        
        # Precompute for each BC
        self._precomputed_bcs = {}
        for name, x_bc in self._train_data.items():
            if name != 'pde':
                self._precomputed_bcs[name] = self.network.precompute_training_data(
                    x_bc, threshold=self._sparse_threshold, params=params_dict
                )

    # ==================== Tensor Conversion ====================
    
    def _to_tensor(self, np_array: np.ndarray):
        """Convert numpy array to PyTorch tensor."""
        return torch.tensor(np_array, device=self.device, dtype=self.dtype, requires_grad=True)
    
    def _to_numpy(self, tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy."""
        return tensor.detach().cpu().numpy()
    
    def _index_tensor(self, tensor, indices):
        """Index a PyTorch tensor with numpy indices."""
        return tensor[indices]

    # ==================== Residual (Abstract Implementation) ====================
    
    def _get_pde_residual_tensor(self, x, y, params_dict):
        """Compute PDE residual using PyTorch autograd."""
        return self.problem.pde_fn(x, y, params_dict)
    
    def _mean_squared(self, tensor):
        """Compute mean squared value of a PyTorch tensor."""
        return torch.mean(tensor ** 2)
    
    def _compute_directional_derivative(self, x, component: int, dim: int, params_dict):
        """Compute derivative of y[component] w.r.t. x[dim] using PyTorch autograd."""
        y = self.network.forward(x, params_dict)
        u = y[:, component:component + 1]
        grads = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        return grads[:, dim]
    
    # ==================== Training ======================================
    
    def train(self):
        """Run training loop."""
        epochs = self._epochs
        print_each = self._print_each
        show_plots = self._show_plots
        save_plots = self._save_plots
        
        self.network.train()
        
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
        
        start_epoch = self._global_epoch
        training_start_time = time.time()
        
        # Calculate number of batches (mini-batching not used with L-BFGS)
        n_batches = self._get_n_batches() if self.optimizer_name != "lbfgs" else 1
        use_batching = n_batches > 1
        
        if use_batching:
            print(f"Starting training for {epochs} epochs, {n_batches} batches/epoch...")
        else:
            print(f"Starting training for {epochs} epochs...")
        
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
        
        # Display initial plot before training starts
        if show_plots:
            self.plot_progress(epoch=start_epoch, internal={'global_step': start_epoch, 'step': 0},
                             weights_dict=self._list_to_dict_weights(self.weights),
                             params_dict=self._build_params({'global_step': start_epoch, 'step': 0}))
        
        # Print epoch 0 (before any training)
        if print_each > 0:
            def to_float(val):
                if hasattr(val, 'item'):
                    return val.item()
                return float(val)
            
            params_dict_init = self._build_params({'global_step': start_epoch, 'step': 0})
            weights_dict_init = self._list_to_dict_weights(self.weights)
            metrics_batch_size = self._batch_size if self._batch_size and self._batch_size > 0 else 1000
            full_train_loss, individual_losses = self._compute_total_loss_batched(
                self._train_data, params_dict_init, weights_dict_init, batch_size=metrics_batch_size
            )
            pde_loss_val = to_float(individual_losses.get('pde', 0.0))
            bc_names = self._get_bc_names()
            bc_losses_str = ", ".join(
                f"{name}: {to_float(individual_losses.get(name, 0.0)):.2e}" 
                for name in bc_names
            )
            
            self.history['epoch'].append(start_epoch)
            self.history['train_loss'].append(float(full_train_loss))
            self.history['loss'].append(float(full_train_loss))
            self.history['loss_pde'].append([pde_loss_val])
            bcs = [to_float(individual_losses.get(name, 0.0)) for name in bc_names]
            self.history['loss_bcs'].append(bcs)
            
            if any(s > 0 for s in self.test_samples) and self._test_data:
                test_weights = {k: 1.0 for k in self._test_data.keys()}
                test_total, _ = self._compute_total_loss_batched(
                    self._test_data, params_dict_init, test_weights, batch_size=metrics_batch_size
                )
                self.history['test_loss'].append(float(test_total))
            
            if self.problem.solution is not None:
                sol_error = self._compute_solution_error()
                self.history['solution_error'].append(sol_error)
            
            msg = (
                f"Epoch 0/{self._epochs + start_epoch} | "
                f"Loss: {float(full_train_loss):.2e} | "
                f"MSE Loss: {float(full_train_loss):.2e} | "
                f"PDE: {pde_loss_val:.2e} | "
                f"BCs: [{bc_losses_str}] | "
                f"Time: 0.0s"
            )
            if self.history['test_loss']:
                msg += f" | Test: {self.history['test_loss'][-1]:.2e}"
            if self.problem.solution is not None:
                msg += f" | Error: {self.history['solution_error'][-1]:.2e}"
            print(msg)
        
        for local_epoch in range(epochs):
            epoch = start_epoch + local_epoch
            internal = {'global_step': epoch, 'step': local_epoch}
            weights_dict = self._list_to_dict_weights(self.weights)
            params_dict = self._build_params(internal)
            
            # Update learning rate if scheduler is provided
            if lr_scheduler is not None and self.optimizer_name != "lbfgs":
                new_lr = lr_scheduler.lr(self.learning_rate, epoch)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
            
            # Refresh pool with fresh samples if interval reached
            if pool_refresh_each > 0 and has_pool and local_epoch > 0 and local_epoch % pool_refresh_each == 0:
                self._sample_pool_data(pool_size)
            
            # Resample collocation points if interval reached
            if resample_each > 0 and local_epoch > 0 and local_epoch % resample_each == 0:
                if has_pool:
                    # Fast: select from pre-sampled pool
                    self._select_from_pool()
                else:
                    # Slow: full resampling
                    self._sample_train_data()
                    if any(s > 0 for s in self.test_samples):
                        self._sample_test_data()
            
            # Adaptive resampling based on residuals
            if adaptive_sampling and local_epoch > 0 and local_epoch % adaptive_each == 0:
                self._adaptive_resample(params_dict)
                # Recalculate n_batches if dataset size changed (adaptive_mode="add")
                if use_batching and getattr(self, '_adaptive_mode', 'replace') == 'add':
                    n_batches = self._get_n_batches()
            
            if self.optimizer_name == "lbfgs":
                def closure():
                    self.optimizer.zero_grad()
                    loss, _ = self._compute_total_loss(self._train_data, params_dict, weights_dict)
                    loss.backward()
                    return loss
                
                self.optimizer.step(closure)
                loss, losses = self._compute_total_loss(self._train_data, params_dict, weights_dict)
            elif use_batching:
                # Shuffle data at the start of each epoch
                shuffled_train_data = {}
                for name, data in self._train_data.items():
                    n_points = len(data)
                    perm = torch.randperm(n_points, device=data.device)
                    shuffled_train_data[name] = data[perm]
                
                # Mini-batch training
                epoch_loss = 0.0
                for batch_idx in range(n_batches):
                    # Create batch data
                    batch_data = {}
                    for name, data in shuffled_train_data.items():
                        n_points = len(data)
                        start_idx, end_idx = self._get_batch_indices(n_points, batch_idx, n_batches)
                        batch_data[name] = data[start_idx:end_idx]
                    
                    self.optimizer.zero_grad()
                    batch_loss, batch_losses = self._compute_total_loss(batch_data, params_dict, weights_dict)
                    batch_loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += float(batch_loss.item())
                
                # Recompute on full data for metrics (detached, batched to avoid OOM)
                with torch.no_grad():
                    _, losses = self._compute_total_loss_batched(
                        self._train_data, params_dict, weights_dict, 
                        batch_size=self._batch_size if self._batch_size and self._batch_size > 0 else 1000
                    )
                loss = epoch_loss / n_batches  # Report average batch loss
            else:
                self.optimizer.zero_grad()
                loss, losses = self._compute_total_loss(self._train_data, params_dict, weights_dict)
                loss.backward()
                self.optimizer.step()
            
            # Update ReduceLROnPlateau scheduler with current loss
            if lr_scheduler is not None and hasattr(lr_scheduler, 'step'):
                loss_val_for_scheduler = loss.item() if hasattr(loss, 'item') else float(loss)
                lr_scheduler.step(loss_val_for_scheduler, epoch)
            
            # Batch size for metrics computation (avoid OOM on large datasets)
            metrics_batch_size = self._batch_size if self._batch_size and self._batch_size > 0 else 1000
            
            # Record history (handle both tensor and float values)
            def to_float(val):
                return val.item() if hasattr(val, 'item') else float(val)
            
            self.history['epoch'].append(epoch)
            loss_val = to_float(loss)
            self.history['loss'].append(loss_val)
            self.history['train_loss'].append(loss_val)
            pde_individual = losses.get('pde_individual', [losses['pde']])
            if isinstance(pde_individual, list):
                self.history['loss_pde'].append([to_float(l) for l in pde_individual])
            else:
                self.history['loss_pde'].append([to_float(pde_individual)])
            bcs = losses.get('bcs', [])
            self.history['loss_bcs'].append([to_float(l) for l in bcs] if bcs else [])
            
            # Print progress
            if print_each > 0 and ((epoch + 1) % print_each == 0 or local_epoch == epochs - 1):
                if any(s > 0 for s in self.test_samples) and self._test_data:
                    # Batched test loss to avoid OOM
                    test_weights = {k: 1.0 for k in self._test_data.keys()}
                    test_total, _ = self._compute_total_loss_batched(
                        self._test_data, params_dict, test_weights, batch_size=metrics_batch_size
                    )
                    self.history['test_loss'].append(float(test_total))
                
                if self.problem.solution is not None:
                    sol_error = self._compute_solution_error()
                    self.history['solution_error'].append(sol_error)
                
                elapsed_time = time.time() - training_start_time
                
                bc_names = self._get_bc_names()
                pde_loss_val = to_float(losses['pde'])
                bcs_list = losses.get('bcs', [])
                bc_losses_str = ", ".join(
                    f"{bc_names[i]}: {to_float(l):.2e}"
                    for i, l in enumerate(bcs_list)
                )
                
                msg = (
                    f"Epoch {epoch + 1}/{self._epochs + start_epoch} | "
                    f"Loss: {loss_val:.2e} | "
                    f"MSE Loss: {loss_val:.2e} | "
                    f"PDE: {pde_loss_val:.2e} | "
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
        print(f"Training complete in {time.time() - training_start_time:.1f}s")
        
        # Close figure to prevent duplicate display in notebooks
        if is_notebook() and show_plots and self._fig is not None:
            plt.close(self._fig)


class ALTrainer(Trainer):
    """
    Augmented Lagrangian Trainer for Physics-Informed Neural Networks.
    
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
           b. Update λ: λ_{ij} = λ_{ij} + lagrange_lr * g_i(x_j) (dual update)
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
            device: Device to use ('cpu', 'cuda', 'mps'). Default: auto-detect.
        """
        super().__init__(problem, network, device)
        
        # Augmented Lagrangian specific attributes
        self.lagrange_multipliers = {}  # Lagrange multipliers λ_i (per constraint, per point)
        self.lagrange_lr = 1.0  # Dual learning rate
        self._outer_epoch = 0
        self._lagrange_max = 1e6  # Maximum absolute value for λ (clipping)
        self._lagrange_constraints = None  # Which constraints get λ (None = all)
        self._lagrange_optimizer = None  # Optimizer for λ updates
        self._lagrange_optimizer_name = 'adam'  # 'adam', 'sgd', or 'none'
    
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
            optimizer: Optimizer name ('adam', 'lbfgs', 'sgd').
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
        
        # For PDE constraint
        if 'pde' in self._train_data:
            n_pde_points = len(self._train_data['pde'])
            self.lagrange_multipliers['pde'] = torch.zeros(n_pde_points, device=self.device, dtype=self.dtype)
        
        # For each BC constraint
        bc_names = self._get_bc_names()
        for name in bc_names:
            if name in self._train_data:
                n_bc_points = len(self._train_data[name])
                self.lagrange_multipliers[name] = torch.zeros(n_bc_points, device=self.device, dtype=self.dtype)
        
        # Create λ optimizer (matching paper: Adam)
        self._create_lagrange_optimizer()
    
    def _create_lagrange_optimizer(self):
        """Create optimizer for λ updates (matching AL-PINNs paper)."""
        if self._lagrange_optimizer_name == 'none':
            self._lagrange_optimizer = None
            return
        
        # Determine which λ to optimize
        lc = self._lagrange_constraints
        lagrange_params = []
        for name, lam in self.lagrange_multipliers.items():
            use_lagrange = (lc is None) or (name in lc)
            if use_lagrange:
                # Make λ a leaf tensor with gradients
                self.lagrange_multipliers[name] = lam.clone().detach().requires_grad_(True)
                lagrange_params.append(self.lagrange_multipliers[name])
        
        if not lagrange_params:
            self._lagrange_optimizer = None
            return
        
        if self._lagrange_optimizer_name == 'adam':
            self._lagrange_optimizer = torch.optim.Adam(lagrange_params, lr=self.lagrange_lr)
        elif self._lagrange_optimizer_name == 'sgd':
            self._lagrange_optimizer = torch.optim.SGD(lagrange_params, lr=self.lagrange_lr)
    
    def _reinitialize_lagrange_if_needed(self):
        """Reinitialize λ if data size changed (e.g., after resampling)."""
        needs_reinit = False
        for name, data in self._train_data.items():
            if name in self.lagrange_multipliers:
                if len(self.lagrange_multipliers[name]) != len(data):
                    self.lagrange_multipliers[name] = torch.zeros(len(data), device=self.device, dtype=self.dtype)
                    needs_reinit = True
        
        # Recreate optimizer if λ sizes changed
        if needs_reinit and self._lagrange_optimizer is not None:
            self._create_lagrange_optimizer()
    
    def _compute_constraint_residuals(self, data: Dict, params_dict: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute raw constraint residuals (not squared) for all constraints.
        
        Args:
            data: Dict with 'pde' and BC name keys mapping to input tensors
            params_dict: Parameters dict from _build_params()
            
        Returns:
            Dict mapping constraint names to residual tensors
        """
        residuals = {}
        
        # PDE residual
        if 'pde' in data:
            x_pde = data['pde']
            y = self.network.forward(x_pde, params_dict)
            pde_residual = self._get_pde_residual_tensor(x_pde, y, params_dict)
            
            # Handle multiple PDE equations
            if isinstance(pde_residual, (list, tuple)):
                residuals['pde'] = sum(r.flatten() for r in pde_residual)
            else:
                residuals['pde'] = pde_residual.flatten()
        
        # BC residuals
        bc_names = self._get_bc_names()
        for i, bc in enumerate(self.problem.boundary_conditions):
            name = bc_names[i]
            if name in data:
                x_bc = data[name]
                bc_residual = self._compute_bc_residual(bc, x_bc, params_dict)
                residuals[name] = bc_residual.flatten()
        
        return residuals
    
    def _compute_bc_residual(self, bc, x, params_dict) -> torch.Tensor:
        """
        Compute boundary condition residual (not squared loss).
        
        Args:
            bc: Boundary condition object
            x: Input points tensor
            params_dict: Parameters dict
            
        Returns:
            Residual tensor of shape (n_points,)
        """
        y = self.network.forward(x, params_dict)
        component = bc.component if bc.component is not None else 0
        u = y[:, component:component + 1]
        
        if isinstance(bc, DirichletBC):
            if callable(bc.value):
                target = bc.value(x)
                if target.dim() == 1:
                    target = target.unsqueeze(-1)
            else:
                target = bc.value
            return (u - target).flatten()
        
        elif isinstance(bc, NeumannBC):
            grads = torch.autograd.grad(
                u, x, grad_outputs=torch.ones_like(u),
                create_graph=True, retain_graph=True
            )[0]
            du_dn = grads[:, bc.normal_dim:bc.normal_dim + 1]
            
            if callable(bc.value):
                target = bc.value(x)
                if target.dim() == 1:
                    target = target.unsqueeze(-1)
            else:
                target = bc.value
            return (du_dn - target).flatten()
        
        elif isinstance(bc, RobinBC):
            grads = torch.autograd.grad(
                u, x, grad_outputs=torch.ones_like(u),
                create_graph=True, retain_graph=True
            )[0]
            du_dn = grads[:, bc.normal_dim:bc.normal_dim + 1]
            
            alpha = bc.alpha(x) if callable(bc.alpha) else bc.alpha
            beta = bc.beta(x) if callable(bc.beta) else bc.beta
            gamma = bc.gamma(x) if callable(bc.gamma) else bc.gamma
            
            return (alpha * u + beta * du_dn - gamma).flatten()
        
        elif isinstance(bc, PointsetBC):
            target = bc.values[:, bc.component:bc.component + 1]
            target = torch.as_tensor(target, device=x.device, dtype=x.dtype)
            return (u - target).flatten()
        
        else:
            raise ValueError(f"Unknown BC type: {type(bc)}")
    
    def _compute_al_loss(self, data: Dict, params_dict: Dict, weights_dict: Dict) -> tuple:
        """
        Compute augmented Lagrangian loss.
        
        L = sum_i [w_i ||g_i||^2 + λ_i · g_i]
        
        Where w_i are the weights (penalty parameters β).
        λ terms are only added for constraints in lagrange_constraints (or all if None).
        
        Args:
            data: Dict with constraint data
            params_dict: Parameters dict
            weights_dict: Penalty weights for each constraint (serves as β_i)
            
        Returns:
            Tuple of (total_loss, losses_dict)
        """
        total_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        losses = {'bcs': []}
        
        # Compute all constraint residuals
        residuals = self._compute_constraint_residuals(data, params_dict)
        lc = self._lagrange_constraints
        
        # PDE term: w_pde ||g_pde||^2 + λ_pde · g_pde (if pde in lagrange_constraints)
        if 'pde' in residuals:
            g_pde = residuals['pde']
            weight_pde = weights_dict.get('pde', 1.0)
            lagrange_pde = self.lagrange_multipliers.get('pde', torch.zeros_like(g_pde))
            
            # Ensure λ matches current data size
            if len(lagrange_pde) != len(g_pde):
                lagrange_pde = torch.zeros_like(g_pde)
                self.lagrange_multipliers['pde'] = lagrange_pde
            
            # Penalty term: w * (1/N) Σ_j ||g_j||^2
            penalty_pde = weight_pde * torch.mean(g_pde ** 2)
            
            # Lagrangian term: only if pde is in lagrange_constraints
            use_lagrange = (lc is None) or ('pde' in lc)
            if use_lagrange:
                lagrangian_pde = torch.mean(lagrange_pde.detach() * g_pde)
            else:
                lagrangian_pde = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            
            pde_loss = penalty_pde + lagrangian_pde
            losses['pde'] = pde_loss
            losses['pde_penalty'] = penalty_pde
            losses['pde_lagrangian'] = lagrangian_pde
            losses['pde_residual_mean'] = torch.mean(torch.abs(g_pde))
            total_loss = total_loss + pde_loss
        
        # BC terms: w_i ||g_i||^2 + λ_i · g_i
        bc_names = self._get_bc_names()
        for name in bc_names:
            if name in residuals:
                g_bc = residuals[name]
                weight_bc = weights_dict.get(name, 1.0)
                lagrange_bc = self.lagrange_multipliers.get(name, torch.zeros_like(g_bc))
                
                if len(lagrange_bc) != len(g_bc):
                    lagrange_bc = torch.zeros_like(g_bc)
                    self.lagrange_multipliers[name] = lagrange_bc
                
                # Penalty term: w_i * (1/N_i) Σ_j ||g_ij||^2
                penalty_bc = weight_bc * torch.mean(g_bc ** 2)
                
                # Lagrangian term: only if this constraint is in lagrange_constraints
                use_lagrange = (lc is None) or (name in lc)
                if use_lagrange:
                    lagrangian_bc = torch.mean(lagrange_bc.detach() * g_bc)
                else:
                    lagrangian_bc = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                
                bc_loss = penalty_bc + lagrangian_bc
                losses['bcs'].append(bc_loss)
                losses[name] = bc_loss
                losses[f'{name}_penalty'] = penalty_bc
                losses[f'{name}_lagrangian'] = lagrangian_bc
                losses[f'{name}_residual_mean'] = torch.mean(torch.abs(g_bc))
                total_loss = total_loss + bc_loss
        
        return total_loss, losses
    
    def _update_lagrange_multipliers(self, data: Dict, params_dict: Dict):
        """
        Update Lagrange multipliers (dual ascent step).
        
        When using optimizer (default 'adam', matching AL-PINNs paper):
            The gradient of mean(λ·g) w.r.t. λ is g/N.
            For gradient ascent: set λ.grad = -g/N, then optimizer.step()
            This matches paper's `lbd.grad *= -1` followed by optimizer.step().
        
        When optimizer='none' (simple gradient ascent):
            λ_{ij} = λ_{ij} + lagrange_lr * g_i(x_j) / N_i
        
        Only updates λ for constraints in lagrange_constraints (or all if None).
        """
        lc = self._lagrange_constraints
        
        if self._lagrange_optimizer is not None:
            # Optimizer-based update (paper's approach)
            self._lagrange_optimizer.zero_grad()
            
            # Compute residuals
            with torch.no_grad():
                residuals = self._compute_constraint_residuals(data, params_dict)
            
            # Set gradients for gradient ascent (negate for maximization)
            for name, g in residuals.items():
                use_lagrange = (lc is None) or (name in lc)
                if name in self.lagrange_multipliers and use_lagrange and self.lagrange_multipliers[name].requires_grad:
                    n_points = len(g)
                    # Gradient for ascent: -g/N (optimizer does descent, so negate)
                    self.lagrange_multipliers[name].grad = -g / n_points
            
            # Apply optimizer update
            self._lagrange_optimizer.step()
            
            # Clip λ for stability
            with torch.no_grad():
                for name in self.lagrange_multipliers:
                    use_lagrange = (lc is None) or (name in lc)
                    if use_lagrange:
                        self.lagrange_multipliers[name].data.clamp_(-self._lagrange_max, self._lagrange_max)
        else:
            # Simple gradient ascent (no optimizer)
            with torch.no_grad():
                residuals = self._compute_constraint_residuals(data, params_dict)
                
                for name, g in residuals.items():
                    use_lagrange = (lc is None) or (name in lc)
                    if name in self.lagrange_multipliers and use_lagrange:
                        n_points = len(g)
                        self.lagrange_multipliers[name] = self.lagrange_multipliers[name] + self.lagrange_lr * g / n_points
                        self.lagrange_multipliers[name] = torch.clamp(
                            self.lagrange_multipliers[name], 
                            -self._lagrange_max, 
                            self._lagrange_max
                        )
    
    def train(self):
        """
        Run augmented Lagrangian training loop.
        
        Each epoch: one primal optimization step + one dual update (λ).
        """
        epochs = self._epochs
        print_each = self._print_each
        show_plots = self._show_plots
        save_plots = self._save_plots
        
        self.network.train()
        
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
        
        start_epoch = self._global_epoch
        training_start_time = time.time()
        
        resample_each = getattr(self, '_resample_each', 0)
        adaptive_sampling = getattr(self, '_adaptive_sampling', False)
        adaptive_each = getattr(self, '_adaptive_each', 100)
        lr_scheduler = getattr(self, '_lr_scheduler', None)
        
        weights_dict = self._list_to_dict_weights(self.weights)
        
        print(f"Starting ALTrainer (PyTorch) for {epochs} epochs...")
        print(f"Weights (penalty parameters): {weights_dict}")
        print(f"Lagrange optimizer: {self._lagrange_optimizer_name}, lr: {self.lagrange_lr}")
        
        # Display initial plot before training starts
        if show_plots:
            internal_init = {'global_step': start_epoch, 'step': start_epoch, 'outer_epoch': 0}
            params_dict_init = self._build_params(internal_init)
            self.plot_progress(epoch=start_epoch, internal=internal_init,
                             weights_dict=weights_dict,
                             params_dict=params_dict_init)
        
        # Print epoch 0 (before any training)
        if print_each > 0:
            def to_float(val):
                if hasattr(val, 'item'):
                    return val.item()
                return float(val)
            
            internal_init = {'global_step': start_epoch, 'step': start_epoch, 'outer_epoch': 0}
            params_dict_init = self._build_params(internal_init)
            al_loss, mse_loss, losses, residuals = self._compute_al_loss(self._train_data, params_dict_init, weights_dict)
            
            al_loss_val = to_float(al_loss)
            mse_loss_val = to_float(mse_loss)
            pde_mse = to_float(torch.mean(residuals['pde'] ** 2)) if 'pde' in residuals else 0.0
            bc_names = self._get_bc_names()
            bc_mse_losses = [to_float(torch.mean(residuals[name] ** 2)) if name in residuals else 0.0 
                            for name in bc_names]
            bc_losses_str = ", ".join(
                f"{bc_names[i]}: {bc_mse_losses[i]:.2e}"
                for i in range(len(bc_names))
            )
            
            self.history['epoch'].append(start_epoch)
            self.history['train_loss'].append(mse_loss_val)
            self.history['loss'].append(al_loss_val)
            if 'al_loss' not in self.history:
                self.history['al_loss'] = []
            self.history['al_loss'].append(al_loss_val)
            if 'mse_loss' not in self.history:
                self.history['mse_loss'] = []
            self.history['mse_loss'].append(mse_loss_val)
            self.history['loss_pde'].append([pde_mse])
            self.history['loss_bcs'].append(bc_mse_losses)
            
            # Initialize AL-specific history lists for epoch 0
            if 'al_pde_penalty' not in self.history:
                self.history['al_pde_penalty'] = []
                self.history['al_pde_lagrangian'] = []
                self.history['al_bcs_penalty'] = []
                self.history['al_bcs_lagrangian'] = []
            
            # At epoch 0, record initial penalty and lagrangian terms
            pde_penalty_init = to_float(losses.get('pde_penalty', 0.0))
            pde_lagrangian_init = to_float(losses.get('pde_lagrangian', 0.0))
            self.history['al_pde_penalty'].append(pde_penalty_init)
            self.history['al_pde_lagrangian'].append(pde_lagrangian_init)
            bc_penalty_init = [to_float(losses.get(f'{name}_penalty', 0.0)) for name in bc_names]
            bc_lagrangian_init = [to_float(losses.get(f'{name}_lagrangian', 0.0)) for name in bc_names]
            self.history['al_bcs_penalty'].append(bc_penalty_init)
            self.history['al_bcs_lagrangian'].append(bc_lagrangian_init)
            
            if any(s > 0 for s in self.test_samples) and self._test_data:
                test_residuals = self._compute_constraint_residuals(self._test_data, params_dict_init)
                test_mse = sum(torch.mean(g ** 2) for g in test_residuals.values())
                self.history['test_loss'].append(to_float(test_mse))
            
            if self.problem.solution is not None:
                sol_error = self._compute_solution_error()
                self.history['solution_error'].append(sol_error)
            
            msg = (
                f"Epoch 0/{self._epochs + start_epoch} | "
                f"AL Loss: {al_loss_val:.2e} | "
                f"MSE Loss: {mse_loss_val:.2e} | "
                f"PDE: {pde_mse:.2e} | "
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
            
            # ==================== Primal Optimization ====================
            internal = {'global_step': epoch, 'step': epoch, 'outer_epoch': self._outer_epoch}
            weights_dict = self._list_to_dict_weights(self.weights)
            params_dict = self._build_params(internal)
            
            if lr_scheduler is not None and self.optimizer_name != "lbfgs":
                new_lr = lr_scheduler.lr(self.learning_rate, epoch)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
            
            if resample_each > 0 and epoch > start_epoch and epoch % resample_each == 0:
                self._sample_train_data()
                self._reinitialize_lagrange_if_needed()
            
            if adaptive_sampling and epoch > start_epoch and epoch % adaptive_each == 0:
                self._adaptive_resample(params_dict)
                self._reinitialize_lagrange_if_needed()
            
            if self.optimizer_name == "lbfgs":
                def closure():
                    self.optimizer.zero_grad()
                    loss, _ = self._compute_al_loss(self._train_data, params_dict, weights_dict)
                    loss.backward()
                    return loss
                self.optimizer.step(closure)
                loss, losses = self._compute_al_loss(self._train_data, params_dict, weights_dict)
            else:
                self.optimizer.zero_grad()
                loss, losses = self._compute_al_loss(self._train_data, params_dict, weights_dict)
                loss.backward()
                self.optimizer.step()
            
            if lr_scheduler is not None and hasattr(lr_scheduler, 'step'):
                lr_scheduler.step(loss.item(), epoch)
            
            # ==================== Dual Update (Lagrange Multipliers) ====================
            self._update_lagrange_multipliers(self._train_data, params_dict)
            
            # ==================== Record History and Print ====================
            def to_float(val):
                return val.item() if hasattr(val, 'item') else float(val)
            
            # Store both AL loss (for training dynamics) and MSE loss (for comparison)
            al_loss_val = to_float(loss)
            residuals = self._compute_constraint_residuals(self._train_data, params_dict)
            mse_loss = sum(torch.mean(g ** 2) for g in residuals.values())
            mse_loss_val = to_float(mse_loss)
            
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
            
            self.history['al_pde_penalty'].append(to_float(losses.get('pde_penalty', 0.0)))
            self.history['al_pde_lagrangian'].append(to_float(losses.get('pde_lagrangian', 0.0)))
            
            bc_names = self._get_bc_names()
            bc_penalty_list = [to_float(losses.get(f'{name}_penalty', 0.0)) for name in bc_names]
            bc_lagrangian_list = [to_float(losses.get(f'{name}_lagrangian', 0.0)) for name in bc_names]
            self.history['al_bcs_penalty'].append(bc_penalty_list)
            self.history['al_bcs_lagrangian'].append(bc_lagrangian_list)
            
            # Compute per-constraint MSE losses
            pde_mse = to_float(torch.mean(residuals['pde'] ** 2)) if 'pde' in residuals else 0.0
            self.history['loss_pde'].append([pde_mse])
            
            bc_mse_losses = [to_float(torch.mean(residuals[name] ** 2)) if name in residuals else 0.0 
                            for name in bc_names]
            self.history['loss_bcs'].append(bc_mse_losses)
            
            if print_each > 0 and ((epoch + 1) % print_each == 0 or self._outer_epoch == epochs - 1):
                if any(s > 0 for s in self.test_samples) and self._test_data:
                    # Compute test MSE loss (without λ terms)
                    test_residuals = self._compute_constraint_residuals(self._test_data, params_dict)
                    test_mse = sum(torch.mean(g ** 2) for g in test_residuals.values())
                    self.history['test_loss'].append(to_float(test_mse))
                
                if self.problem.solution is not None:
                    sol_error = self._compute_solution_error()
                    self.history['solution_error'].append(sol_error)
                
                elapsed_time = time.time() - training_start_time
                
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
        print(f"ALTrainer complete in {time.time() - training_start_time:.1f}s")
        
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
                'mean': lam.mean().item(),
                'std': lam.std().item(),
                'min': lam.min().item(),
                'max': lam.max().item(),
            }
        return stats
    
    def reset_lagrange_multipliers(self):
        """Reset all Lagrange multipliers to zero."""
        for name in self.lagrange_multipliers:
            self.lagrange_multipliers[name].zero_()
    
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
    