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
        
        for local_epoch in range(epochs):
            epoch = start_epoch + local_epoch
            internal = {'global_step': epoch, 'step': local_epoch}
            weights_dict = self._list_to_dict_weights(self.weights)
            params_dict = self._build_params(internal)
            
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
            if print_each > 0 and (local_epoch % print_each == 0 or local_epoch == epochs - 1):
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
                    f"Epoch {epoch}/{self._epochs + start_epoch} | "
                    f"Loss: {loss_val:.6f} | "
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
    