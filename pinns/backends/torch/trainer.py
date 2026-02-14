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
    
    def _init_optimizer_state(self):
        """PyTorch optimizers don't need separate state initialization."""
        pass
    
    # ==================== Tensor Conversion ====================
    
    def _to_tensor(self, np_array: np.ndarray):
        """Convert numpy array to PyTorch tensor."""
        return torch.tensor(np_array, device=self.device, dtype=self.dtype, requires_grad=True)
    
    def _to_numpy(self, tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy."""
        return tensor.detach().cpu().numpy()

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
        
        if show_plots and is_notebook():
            n_zoom_regions = len(getattr(self, '_plot_regions', []))
            needs_recreation = self._fig is None
            if not needs_recreation and self._axes is not None and n_zoom_regions > 0:
                if f'zoom_0_0' not in self._axes:
                    needs_recreation = True
            if needs_recreation:
                self._fig, self._axes = self._create_figure()
        
        start_epoch = self._global_epoch
        training_start_time = time.time()
        
        # Auto-save path for script mode
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
        
        print(f"Starting training for {epochs} epochs...")
        
        for local_epoch in range(epochs):
            epoch = start_epoch + local_epoch
            internal = {'global_step': epoch, 'step': local_epoch}
            weights_dict = self._list_to_dict_weights(self.weights)
            params_dict = self._build_params(internal)
            
            if self.optimizer_name == "lbfgs":
                def closure():
                    self.optimizer.zero_grad()
                    loss, _ = self._compute_total_loss(self._train_data, params_dict, weights_dict)
                    loss.backward()
                    return loss
                
                self.optimizer.step(closure)
                loss, losses = self._compute_total_loss(self._train_data, params_dict, weights_dict)
            else:
                self.optimizer.zero_grad()
                loss, losses = self._compute_total_loss(self._train_data, params_dict, weights_dict)
                loss.backward()
                self.optimizer.step()
            
            # Record history
            self.history['epoch'].append(epoch)
            self.history['loss'].append(loss.item())
            self.history['train_loss'].append(loss.item())
            pde_individual = losses.get('pde_individual', [losses['pde']])
            self.history['loss_pde'].append([l.item() for l in pde_individual])
            self.history['loss_bcs'].append([l.item() for l in losses['bcs']])
            
            # Print progress
            if print_each > 0 and (local_epoch % print_each == 0 or local_epoch == epochs - 1):
                if any(s > 0 for s in self.test_samples) and self._test_data:
                    # Enable gradients for test data derivative computation
                    for key in self._test_data:
                        if hasattr(self._test_data[key], 'requires_grad_'):
                            self._test_data[key].requires_grad_(True)
                    with torch.enable_grad():
                        test_weights = {k: 1.0 for k in self._test_data.keys()}
                        test_total, _ = self._compute_total_loss(self._test_data, params_dict, test_weights)
                    self.history['test_loss'].append(float(test_total.detach()))
                
                if self.problem.solution is not None:
                    sol_error = self._compute_solution_error()
                    self.history['solution_error'].append(sol_error)
                
                elapsed_time = time.time() - training_start_time
                
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
    