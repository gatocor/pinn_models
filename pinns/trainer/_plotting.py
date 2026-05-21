"""
Plotting mixin for Trainer.

All region-parsing, figure creation, and plot_progress logic lives here
so that trainer.py can focus on the training loop and loss computation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from typing import Any, Callable, Dict, List, Optional, Union

from .schedulers import is_notebook


class TrainPlotter:
    """Controls all plotting behaviour during :py:meth:`~pinns.Trainer.train`.

    Pass an instance to ``compile(show=TrainPlotter(...))`` to enable live plots.
    Pass ``show=None`` (the default) to suppress all plotting.

    Parameters
    ----------
    save : str, optional
        Path prefix for saving plots.  A file is written every ``print_each``
        epochs as ``<save>_epoch00500.png``.  When *None* (default) plots are
        shown live but not saved (in notebooks) or saved to an auto-generated
        path (in scripts outside a notebook).
    subdomains : bool or dict, optional
        Draw subdomain / partition boundaries in solution / residual panels.
        Pass a dict with keys ``"solution"``, ``"residuals"``, ``"zoom"`` to
        control per-panel visibility.  Default: ``False``.
    sampling_points : bool or dict, optional
        Overlay training collocation points in the panels.  Same bool-or-dict
        semantics as *subdomains*.  Default: ``False``.
    regions : list of tuple, optional
        List of zoom-region specifications.  Each tuple has one element per
        input dimension; entries can be ``None`` (full range), ``(lo, hi)``
        (zoom range) or a scalar (slice at that value).  Default: ``[]``.
    n_points : int, optional
        Resolution of the plot grid (per dimension).  Default: ``200``.
    panel_kwargs : dict of dict, optional
        Per-panel ``ax.set()`` keyword arguments.
        Supported panel keys: ``"losses"``, ``"mse_losses"``, ``"solution"``,
        ``"residuals"``, ``"error"``.  Default: ``{}``.
    style : dict, optional
        Figure-level style overrides.  Supported keys: ``"theme"``,
        ``"bg_color"``, ``"fig_color"``, ``"text_color"``, ``"grid_color"``.
        Default: ``{}``.
    callback : callable, optional
        Called as ``callback(epoch, trainer)`` every ``print_each`` steps.
        Default: ``None``.
    time_points : list of float, optional
        Snapshot time values for transient mesh solutions.  Default: ``None``.
    """

    def __init__(
        self,
        *,
        save: Optional[str] = None,
        subdomains: Union[bool, Dict[str, bool]] = False,
        sampling_points: Union[bool, Dict[str, bool]] = False,
        regions: Optional[List[tuple]] = None,
        n_points: int = 200,
        panel_kwargs: Optional[Dict[str, Any]] = None,
        style: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable] = None,
        time_points: Optional[List[float]] = None,
        show_prediction: bool = True,
        show_residuals: bool = True,
        show_loss: bool = True,
        show_mse_loss: bool = False,
    ) -> None:
        self.save = save
        self.subdomains = subdomains
        self.sampling_points = sampling_points
        self.regions = regions if regions is not None else []
        self.n_points = n_points
        self.panel_kwargs = panel_kwargs if panel_kwargs is not None else {}
        self.style = style if style is not None else {}
        self.callback = callback
        self.time_points = list(time_points) if time_points is not None else None
        self.show_prediction = show_prediction
        self.show_residuals = show_residuals
        self.show_loss = show_loss
        self.show_mse_loss = show_mse_loss
        self.show_parameters = True  # always enabled; draws param-history panel when params tracked

        # State (set by _activate / reset on each compile)
        self._trainer = None
        self._fig = None
        self._axes = None
        self._display_handle = None
        self._colorbars = []
        # Expanded config (set by _activate)
        self._show_subdomains = {'solution': False, 'residuals': False, 'zoom': False}
        self._show_sampling_points = {'solution': False, 'residuals': False, 'zoom': False}
        self._plot_time_points = None  # alias for self.time_points (set by _activate)

    def _activate(self, trainer):
        """Called by Trainer._compile_base to bind this plotter to the trainer."""
        self._trainer = trainer
        # Reset figure state
        self._fig = None
        self._axes = None
        self._display_handle = None
        self._colorbars = []
        # Expand dict config
        _sd = self.subdomains
        if isinstance(_sd, bool):
            self._show_subdomains = {'solution': _sd, 'residuals': _sd, 'zoom': _sd}
        else:
            self._show_subdomains = {'solution': False, 'residuals': False, 'zoom': False}
            self._show_subdomains.update(_sd)
        _sp = self.sampling_points
        if isinstance(_sp, bool):
            self._show_sampling_points = {'solution': _sp, 'residuals': _sp, 'zoom': _sp}
        else:
            self._show_sampling_points = {'solution': False, 'residuals': False, 'zoom': False}
            self._show_sampling_points.update(_sp)
        # When there is no problem (dataset-only / ModelSpectralSolver mode) disable residuals
        if self._trainer.problem is None:
            self.show_residuals = False
        if self.time_points is not None:
            self._plot_time_points = self.time_points
        elif self._is_mesh_domain():
            domain = self._get_domain()
            if domain._t_min is not None:
                self._plot_time_points = [domain._t_min, domain._t_max]
        elif not self._is_mesh_domain():
            domain = self._get_domain()
            if (domain is not None
                    and getattr(domain, '_t_min', None) is not None
                    and getattr(domain, '_spatial_dims', 0) >= 2):
                self._plot_time_points = [domain._t_min, domain._t_max]
            else:
                self._plot_time_points = None
        else:
            self._plot_time_points = None

    # ==================== Domain/Problem Helpers ====================

    def _get_domain(self):
        """Return the domain from problem (if set) or from the network."""
        if self._trainer.problem is not None:
            return self._trainer.problem.domain
        return getattr(self._trainer.model, 'domain', None)

    def _get_n_dims(self):
        """Return total input dimensions (spatial + time)."""
        if self._trainer.problem is not None:
            return self._trainer.problem.n_dims
        dom = self._get_domain()
        if dom is None:
            return 1
        return dom.n_dims

    def _get_xmin(self, i):
        """Return lower bound for input dimension i."""
        if self._trainer.problem is not None:
            return self._trainer.problem.xmin[i]
        dom = self._get_domain()
        if dom is None:
            return 0.0
        return float(dom.xmin[i])

    def _get_xmax(self, i):
        """Return upper bound for input dimension i."""
        if self._trainer.problem is not None:
            return self._trainer.problem.xmax[i]
        dom = self._get_domain()
        if dom is None:
            return 1.0
        return float(dom.xmax[i])

    def _get_n_outputs(self):
        """Return number of model outputs."""
        if self._trainer.problem is not None:
            return self._trainer.problem.n_outputs
        return getattr(self._trainer.model, 'output_dim', 1)

    def _get_fit_params(self) -> list:
        """Return list of fitted parameter names (from _fit_model_parameters)."""
        return list(getattr(self._trainer, '_fit_model_parameters', None) or [])

    def _get_n_param_rows(self) -> int:
        """Return number of parameter-history rows to show."""
        if not self.show_parameters:
            return 0
        return len(self._get_fit_params())

    def _get_has_solution(self):
        """Return True if an analytical solution is available (trainer or problem)."""
        return self._trainer._has_solution

    def __repr__(self) -> str:  # pragma: no cover
        parts = []
        if self.save:
            parts.append(f"save={self.save!r}")
        if self.subdomains:
            parts.append(f"subdomains={self.subdomains!r}")
        if self.sampling_points:
            parts.append(f"sampling_points={self.sampling_points!r}")
        if self.regions:
            parts.append(f"regions={self.regions!r}")
        if self.n_points != 200:
            parts.append(f"n_points={self.n_points!r}")
        if self.panel_kwargs:
            parts.append(f"panel_kwargs={self.panel_kwargs!r}")
        if self.style:
            parts.append(f"style={self.style!r}")
        if self.callback is not None:
            parts.append("callback=<callable>")
        if self.time_points is not None:
            parts.append(f"time_points={self.time_points!r}")
        if not self.show_prediction:
            parts.append("show_prediction=False")
        if not self.show_residuals:
            parts.append("show_residuals=False")
        return f"TrainPlotter({', '.join(parts)})"


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
        n_dims = self._get_n_dims()
        
        if region is None:
            region = [None] * n_dims
        
        free_dims = []
        free_ranges = []
        fixed_dims = []
        fixed_values = []
        
        for i, spec in enumerate(region):
            if spec is None:
                free_dims.append(i)
                free_ranges.append((self._get_xmin(i), self._get_xmax(i)))
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
        kwargs = {k: v for k, v in self.panel_kwargs.get(key, {}).items()
                  if k not in _IMSHOW_KEYS}
        if kwargs:
            ax.set(**kwargs)

    def _get_imshow_kwargs(self, key: str) -> dict:
        """Return imshow-specific kwargs (norm, cmap, vmin, vmax, …) from plot_kwargs[key]."""
        _IMSHOW_KEYS = {'norm', 'cmap', 'vmin', 'vmax', 'alpha', 'interpolation'}
        return {k: v for k, v in self.panel_kwargs.get(key, {}).items()
                if k in _IMSHOW_KEYS}

    def _apply_plot_style(self, fig, axes: dict):
        """Apply plot_style settings (theme, bg_color, fig_color, text_color) to fig/axes."""
        style = self.style
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
        for cbar in self._colorbars:
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
        n_dims = self._get_n_dims()
        n_outputs = self._get_n_outputs()
        has_solution = self._get_has_solution()
        n_regions = len(self.regions)
        obs_names = list(getattr(self._trainer.problem, 'obs_names', None) or []) if self._trainer.problem is not None else []
        obs_spatial = list(getattr(self._trainer.problem, 'obs_spatial', None) or []) if self._trainer.problem is not None else []
        # Regular observables: those not listed in obs_spatial
        obs_regular = [n for n in obs_names if n not in obs_spatial]
        has_spatial = len(obs_spatial) > 0
        n_obs = len(obs_regular) + (1 if has_spatial else 0)

        _show_pred = self.show_prediction
        _show_res  = self.show_residuals
        _show_loss = self.show_loss
        _show_mse  = self.show_mse_loss
        _n_loss_rows = int(_show_loss) + int(_show_mse)
        _n_param_rows = self._get_n_param_rows()
        _content_row = _n_loss_rows  # first row index after loss panels

        if n_dims == 1:
            n_cols = int(_show_pred) + int(_show_res) + int(has_solution)
            if n_cols == 0:
                n_cols = 1  # at minimum keep the loss rows

            n_rows = _n_loss_rows + n_outputs + n_regions + n_obs + _n_param_rows
            if n_rows == 0:
                n_rows = 1
            fig = plt.figure(figsize=(5 * n_cols, 3.5 * n_rows))
            gs = fig.add_gridspec(n_rows, n_cols)

            axes = {}
            _lr = 0
            if _show_loss:
                axes['losses'] = fig.add_subplot(gs[_lr, :]); _lr += 1
            if _show_mse:
                axes['mse_losses'] = fig.add_subplot(gs[_lr, :])

            for i in range(n_outputs):
                _col = 0
                if _show_pred:
                    axes[f'sol_{i}'] = fig.add_subplot(gs[_content_row + i, _col]); _col += 1
                if _show_res:
                    axes[f'res_{i}'] = fig.add_subplot(gs[_content_row + i, _col]); _col += 1
                if has_solution:
                    axes[f'err_{i}'] = fig.add_subplot(gs[_content_row + i, _col])

            for r in range(n_regions):
                axes[f'region_{r}'] = fig.add_subplot(gs[_content_row + n_outputs + r, :])

            for k, name in enumerate(obs_regular):
                axes[f'obs_{name}'] = fig.add_subplot(gs[_content_row + n_outputs + n_regions + k, :])
            if has_spatial:
                row_s = _content_row + n_outputs + n_regions + len(obs_regular)
                axes['obs__deformed_ref'] = fig.add_subplot(gs[row_s, 0])
                axes['obs__deformed_def'] = fig.add_subplot(gs[row_s, 1])

            _param_row_start = _content_row + n_outputs + n_regions + n_obs
            for _pi, _pname in enumerate(self._get_fit_params()):
                axes[f'param_{_pname}'] = fig.add_subplot(gs[_param_row_start + _pi, :])

        elif n_dims == 2:
            n_cols = int(_show_pred) + int(has_solution) + int(_show_res) + int(has_solution)
            if n_cols == 0:
                n_cols = 1

            n_rows = _n_loss_rows + n_outputs + n_regions + n_obs + _n_param_rows
            if n_rows == 0:
                n_rows = 1
            fig = plt.figure(figsize=(4 * n_cols, 3.5 * n_rows))
            gs = fig.add_gridspec(n_rows, n_cols)

            axes = {}
            _lr = 0
            if _show_loss:
                axes['losses'] = fig.add_subplot(gs[_lr, :]); _lr += 1
            if _show_mse:
                axes['mse_losses'] = fig.add_subplot(gs[_lr, :])

            for i in range(n_outputs):
                _col = 0
                if _show_pred:
                    axes[f'sol_{i}'] = fig.add_subplot(gs[_content_row + i, _col]); _col += 1
                if has_solution:
                    axes[f'true_{i}'] = fig.add_subplot(gs[_content_row + i, _col]); _col += 1
                if _show_res:
                    axes[f'res_{i}'] = fig.add_subplot(gs[_content_row + i, _col]); _col += 1
                if has_solution:
                    axes[f'err_{i}'] = fig.add_subplot(gs[_content_row + i, _col])

            for r in range(n_regions):
                axes[f'region_{r}'] = fig.add_subplot(gs[_content_row + n_outputs + r, :])

            for k, name in enumerate(obs_regular):
                axes[f'obs_{name}'] = fig.add_subplot(gs[_content_row + n_outputs + n_regions + k, :2])
            if has_spatial:
                row_s = _content_row + n_outputs + n_regions + len(obs_regular)
                axes['obs__deformed_ref'] = fig.add_subplot(gs[row_s, 0])
                axes['obs__deformed_def'] = fig.add_subplot(gs[row_s, 1])

            _param_row_start = _content_row + n_outputs + n_regions + n_obs
            for _pi, _pname in enumerate(self._get_fit_params()):
                axes[f'param_{_pname}'] = fig.add_subplot(gs[_param_row_start + _pi, :])

        elif self._is_mesh_domain() and self._plot_time_points:
            # Transient mesh domain with time snapshots.
            # Columns: predicted (opt) | true (opt) | residual (opt) | error (opt)
            ts = self._plot_time_points
            n_snap = len(ts) * n_outputs
            n_cols = int(_show_pred) + int(has_solution) + int(_show_res) + int(has_solution)
            if n_cols == 0:
                n_cols = 1

            n_rows = _n_loss_rows + n_snap
            if n_rows == 0:
                n_rows = 1
            fig = plt.figure(figsize=(4 * n_cols, 3.5 * n_rows))
            gs = fig.add_gridspec(n_rows, n_cols)

            axes = {}
            _lr = 0
            if _show_loss:
                axes['losses'] = fig.add_subplot(gs[_lr, :]); _lr += 1
            if _show_mse:
                axes['mse_losses'] = fig.add_subplot(gs[_lr, :])

            row = _content_row
            for t_val in ts:
                for i in range(n_outputs):
                    _col = 0
                    if _show_pred:
                        axes[f'snap_sol_{i}_t{t_val}'] = fig.add_subplot(gs[row, _col]); _col += 1
                    if has_solution:
                        axes[f'snap_true_{i}_t{t_val}'] = fig.add_subplot(gs[row, _col]); _col += 1
                    if _show_res:
                        axes[f'snap_res_{i}_t{t_val}'] = fig.add_subplot(gs[row, _col]); _col += 1
                    if has_solution:
                        axes[f'snap_err_{i}_t{t_val}'] = fig.add_subplot(gs[row, _col])
                    row += 1
        elif self._is_mesh_domain() and not self._plot_time_points:
            # Static 3D surface mesh.
            # Layout: for each content panel (pred/res) we allocate two gridspec
            # columns — a wide one for the 3D axes and a thin one for its colorbar.
            # This means the colorbar never steals space from the 3D axes.
            n_panels = int(_show_pred) + int(_show_res)
            if n_panels == 0:
                n_panels = 1
            # Two gridspec columns per panel per output: [wide, thin, wide, thin, ...]
            n_gs_cols = 2 * n_panels * n_outputs
            width_ratios = [5, 0.35] * (n_panels * n_outputs)
            n_rows = _n_loss_rows + n_outputs
            if n_rows == 0:
                n_rows = 1
            fig = plt.figure(figsize=(5.5 * n_panels * n_outputs, 4.5 * n_rows))
            gs = fig.add_gridspec(n_rows, n_gs_cols, width_ratios=width_ratios, wspace=0.05)

            axes = {}
            _lr = 0
            if _show_loss:
                axes['losses'] = fig.add_subplot(gs[_lr, :]); _lr += 1
            if _show_mse:
                axes['mse_losses'] = fig.add_subplot(gs[_lr, :])

            for i in range(n_outputs):
                _gs_col = 0  # steps by 2 (wide + thin) per panel
                if _show_pred:
                    ax3 = fig.add_subplot(gs[_content_row + i, _gs_col], projection='3d')
                    axes[f'mesh3d_sol_{i}'] = ax3
                    axes[f'cax_mesh3d_sol_{i}'] = fig.add_subplot(gs[_content_row + i, _gs_col + 1])
                    _gs_col += 2
                if _show_res:
                    ax3 = fig.add_subplot(gs[_content_row + i, _gs_col], projection='3d')
                    axes[f'mesh3d_res_{i}'] = ax3
                    axes[f'cax_mesh3d_res_{i}'] = fig.add_subplot(gs[_content_row + i, _gs_col + 1])

        elif self._plot_time_points and not self._is_mesh_domain():
            # Cubic 2+1D (e.g. Grey-Scott x,y,t): one row per (time × output)
            ts = self._plot_time_points
            n_snap = len(ts) * n_outputs
            n_cols = int(_show_pred) + int(has_solution) + int(_show_res) + int(has_solution)
            if n_cols == 0:
                n_cols = 1

            n_rows = _n_loss_rows + n_snap + _n_param_rows
            if n_rows == 0:
                n_rows = 1
            fig = plt.figure(figsize=(4 * n_cols, 3.5 * n_rows))
            gs = fig.add_gridspec(n_rows, n_cols)

            axes = {}
            _lr = 0
            if _show_loss:
                axes['losses'] = fig.add_subplot(gs[_lr, :]); _lr += 1
            if _show_mse:
                axes['mse_losses'] = fig.add_subplot(gs[_lr, :])

            row = _content_row
            for t_val in ts:
                for i in range(n_outputs):
                    _col = 0
                    if _show_pred:
                        axes[f'snap_sol_{i}_t{t_val}'] = fig.add_subplot(gs[row, _col]); _col += 1
                    if has_solution:
                        axes[f'snap_true_{i}_t{t_val}'] = fig.add_subplot(gs[row, _col]); _col += 1
                    if _show_res:
                        axes[f'snap_res_{i}_t{t_val}'] = fig.add_subplot(gs[row, _col]); _col += 1
                    if has_solution:
                        axes[f'snap_err_{i}_t{t_val}'] = fig.add_subplot(gs[row, _col])
                    row += 1

            for _pi, _pname in enumerate(self._get_fit_params()):
                axes[f'param_{_pname}'] = fig.add_subplot(gs[_content_row + n_snap + _pi, :])

        else:
            # For 3D+: loss plot + region slices for all outputs with residuals
            n_cols = max(2 * n_outputs, 1)  # Two columns per output (solution + residual)
            n_rows = _n_loss_rows + n_regions + _n_param_rows  # loss rows + one row per region
            if n_rows == 0:
                n_rows = 1
            fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
            gs = fig.add_gridspec(n_rows, n_cols)

            axes = {}
            _lr = 0
            if _show_loss:
                axes['losses'] = fig.add_subplot(gs[_lr, :]); _lr += 1
            if _show_mse:
                axes['mse_losses'] = fig.add_subplot(gs[_lr, :])

            for r in range(n_regions):
                for i in range(n_outputs):
                    axes[f'region_{r}_{i}'] = fig.add_subplot(gs[_content_row + r, 2*i])
                    axes[f'region_res_{r}_{i}'] = fig.add_subplot(gs[_content_row + r, 2*i + 1])

            for _pi, _pname in enumerate(self._get_fit_params()):
                axes[f'param_{_pname}'] = fig.add_subplot(gs[_content_row + n_regions + _pi, :])
        
        self._colorbars = []
        self._apply_plot_style(fig, axes)
        return fig, axes
    
    def _plot_parameters(self, axes: dict):
        """Plot inferred parameter values over epochs, with optional true-value reference lines."""
        epochs = self._trainer.history['epoch']
        params_history = self._trainer.history.get('params', {})
        param_solutions = getattr(self._trainer, '_parameter_solutions', {})

        for pname in self._get_fit_params():
            ax_key = f'param_{pname}'
            if ax_key not in axes:
                continue
            ax = axes[ax_key]
            vals = params_history.get(pname, [])
            if vals:
                ep = epochs[:len(vals)]
                ax.plot(ep, vals, 'b-', linewidth=2, label=f'{pname} (inferred)')
            if pname in param_solutions:
                true_val = param_solutions[pname]
                ax.axhline(true_val, color='r', linestyle='--', linewidth=1.5,
                           label=f'{pname} = {true_val:.6g} (true)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(pname)
            ax.set_title(f'Inferred parameter: {pname}')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

    def _plot_losses(self, ax):
        """Plot training-objective curves (weighted + Lagrange) on given axes."""
        epochs = self._trainer.history['epoch']
        if not epochs:
            return

        # Total training loss (includes weights and Lagrange multiplier terms)
        loss_data = self._trainer.history.get('loss', self._trainer.history.get('train_loss', []))
        if loss_data:
            ax.semilogy(epochs, loss_data, 'k-', label='Total (train obj)', linewidth=2)

        # Per-term contributions using actual term names
        mse_terms = self._trainer.history.get('mse_terms', {})
        if mse_terms:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            weights = self._trainer.weights if hasattr(self._trainer, 'weights') else {}
            for i, (name, values) in enumerate(mse_terms.items()):
                if not values:
                    continue
                ep = epochs[:len(values)]
                w = float(weights.get(name, 1.0))
                weighted_vals = [v * w for v in values]
                ax.semilogy(ep, weighted_vals, '--', color=colors[i % len(colors)],
                            label=name, linewidth=1.5)
        else:
            # Fallback for old history format
            pde_losses = self._trainer.history.get('loss_pde', [])
            if len(pde_losses) > 0:
                if isinstance(pde_losses[0], (list, tuple)):
                    pde_array = np.array(pde_losses)
                    for i in range(pde_array.shape[1]):
                        ax.semilogy(epochs, pde_array[:, i], '--', label=f'PDE eq{i+1} (w)')
                else:
                    ax.semilogy(epochs, pde_losses, '--', label='PDE (weighted)')
            bc_names = self._trainer._get_bc_plot_names()
            bc_losses = self._trainer.history.get('loss_bcs', [])
            if bc_losses and len(bc_losses) > 0:
                bc_losses_array = np.array(bc_losses)
                if bc_losses_array.ndim == 2:
                    for i in range(bc_losses_array.shape[1]):
                        bc_label = (bc_names[i] if i < len(bc_names) else f'BC {i+1}') + ' (w)'
                        ax.semilogy(epochs, bc_losses_array[:, i], '--', label=bc_label)

        # Test loss if available
        test_loss = self._trainer.history.get('test_loss', [])
        if len(test_loss) > 0:
            n_test = len(test_loss)
            test_epochs = np.linspace(epochs[0], epochs[-1], n_test).astype(int) if n_test > 1 else [epochs[-1]]
            ax.semilogy(test_epochs, test_loss, 'r:', marker='o', markersize=4, label='Test', linewidth=2)

        # Solution error if available
        sol_error = self._trainer.history.get('solution_error', [])
        if len(sol_error) > 0:
            n_err = len(sol_error)
            err_epochs = np.linspace(epochs[0], epochs[-1], n_err).astype(int) if n_err > 1 else [epochs[-1]]
            ax.semilogy(err_epochs, sol_error, 'm-', marker='s', markersize=4, label='Solution Error', linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss (weighted + Lagrange)')
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_mse_losses(self, ax):
        """Plot unweighted per-term residual MSE (no weights, no Lagrange)."""
        epochs = self._trainer.history['epoch']
        if not epochs:
            return

        # Total unweighted MSE
        mse_total = self._trainer.history.get('mse_loss', [])
        if mse_total:
            ax.semilogy(epochs[:len(mse_total)], mse_total, 'k-', label='Total MSE', linewidth=2)
        else:
            # Fallback: use 'loss' if mse_loss not yet recorded (older history)
            loss_data = self._trainer.history.get('loss', self._trainer.history.get('train_loss', []))
            if loss_data:
                ax.semilogy(epochs, loss_data, 'k-', label='Total', linewidth=2)

        # Per-term unweighted MSE
        mse_terms = self._trainer.history.get('mse_terms', {})
        if mse_terms:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            for i, (name, values) in enumerate(mse_terms.items()):
                if not values:
                    continue
                ep = epochs[:len(values)]
                ax.semilogy(ep, values, '--', color=colors[i % len(colors)],
                            label=name, linewidth=1.5)
        else:
            # Fallback for old history format
            pde_losses = self._trainer.history.get('loss_pde', [])
            if len(pde_losses) > 0:
                ax.semilogy(epochs, pde_losses, 'b--', label='PDE')
            bc_names = self._trainer._get_bc_plot_names()
            bc_losses = self._trainer.history.get('loss_bcs', [])
            if bc_losses and len(bc_losses) > 0:
                bc_losses_array = np.array(bc_losses)
                if bc_losses_array.ndim == 2:
                    for i in range(bc_losses_array.shape[1]):
                        bc_label = bc_names[i] if i < len(bc_names) else f'BC {i+1}'
                        ax.semilogy(epochs, bc_losses_array[:, i], '--', label=bc_label)

        # Solution error
        sol_error = self._trainer.history.get('solution_error', [])
        if len(sol_error) > 0:
            n_err = len(sol_error)
            err_epochs = np.linspace(epochs[0], epochs[-1], n_err).astype(int) if n_err > 1 else [epochs[-1]]
            ax.semilogy(err_epochs, sol_error, 'm-', marker='s', markersize=4, label='Solution Error', linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('mean(r²)')
        ax.set_title('Residual MSE per term (unweighted)')
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_solution_1d(self, ax, output_idx, n_points=200):
        """Plot 1D solution on given axes."""
        x = np.linspace(self._get_xmin(0), self._get_xmax(0), n_points).reshape(-1, 1)
        y = self._trainer.eval(x)
        
        # Plot true solution if available
        if self._get_has_solution():
            y_true = self._trainer._call_solution(x)
            if isinstance(y_true, (list, tuple)):
                y_true = np.concatenate([np.atleast_2d(yt).T if yt.ndim == 1 else yt for yt in y_true], axis=1)
            elif y_true.ndim == 1:
                y_true = y_true.reshape(-1, 1)
            ax.plot(x, y_true[:, output_idx], 'r-', linewidth=2, label='True')
            ax.plot(x, y[:, output_idx], 'b--', linewidth=2, label='Predicted')
            ax.legend(loc='best', fontsize=8)
        else:
            ax.plot(x, y[:, output_idx], 'b-', linewidth=2)
        
        output_name = self._trainer._get_output_name(output_idx)
        input_name = self._trainer._get_input_name(0)
        ax.set_xlabel(input_name)
        ax.set_ylabel(output_name)
        ax.set_title(f'Solution ({output_name})')
        ax.grid(True, alpha=0.3)
    
    def _is_mesh_domain(self):
        """Return True when the problem domain is a DomainMesh."""
        try:
            from pinns.domain import DomainMesh as _DomainMesh
            dom = self._get_domain()
            return dom is not None and isinstance(dom, _DomainMesh)
        except ImportError:
            return False

    def _is_3d_mesh_domain(self):
        """Return True when the domain is a DomainMesh with 3 spatial dimensions (no time)."""
        if not self._is_mesh_domain():
            return False
        dom = self._get_domain()
        # spatial_dims == 3 and no time axis
        return getattr(dom, '_spatial_dims', dom._vertices.shape[1]) == 3

    def _plot_mesh_snapshot_base(self, ax, output_idx, t_val, kind='sol'):
        """Plot a spatial snapshot of a transient mesh solution at time t_val.

        kind : 'sol'  – predicted u(x,y,t)
               'true' – reference solution u(x,y,t)
               'res'  – absolute weak residual |R_j| (t ignored, uses stored residual)
               'err'  – absolute pointwise error |pred - true|
        """
        import matplotlib.tri as _mtri

        dom      = self._get_domain()
        verts_xy = dom._vertices          # (N, 2)
        faces    = dom._faces             # (F, 3)
        n_verts  = len(verts_xy)
        t_dim    = getattr(dom, '_t_dim', self._get_n_dims() - 1)

        # Build space-time input: (N, n_inputs) with t injected at t_dim
        n_inputs = self._get_n_dims()
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
                if not isinstance(self._trainer.problem, _PW) or _u_and_grad is None:
                    raise ValueError('weak residual not available')
                import jax as _jax
                _res_fn = _jax.jit(
                    self._trainer.problem.make_residual_vector_fn_at_t(_u_and_grad, float(t_val))
                )
                R_full = np.array(_res_fn(self._trainer.model.params))   # (n_dofs * n_comp,)
                _n_dofs = self._trainer.problem.n_dofs
                # Take the component for output_idx (row block)
                R_comp = R_full[output_idx * _n_dofs:(output_idx + 1) * _n_dofs]
                # Place values on nodes; non-free nodes get NaN (Dirichlet BCs)
                vals = np.full(n_verts, np.nan, dtype=float)
                for _nj in self._trainer.problem.free_nodes:
                    if int(_nj) < n_verts:
                        vals[int(_nj)] = float(np.abs(R_comp[int(_nj)]))
                # Normalise by node support area so the colour scale is O(residual)
                _node_norm = getattr(self._trainer.problem, 'node_norm', None)
                if _node_norm is not None:
                    for _nj in self._trainer.problem.free_nodes:
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
            y_pred = self._trainer.eval(x_st)[:, output_idx]
            if kind == 'true':
                if not self._get_has_solution():
                    return
                y_true_raw = self._trainer._call_solution(x_st)
                if isinstance(y_true_raw, (list, tuple)):
                    y_true_raw = np.concatenate(
                        [np.atleast_2d(yt).T if yt.ndim == 1 else yt for yt in y_true_raw], axis=1)
                elif y_true_raw.ndim == 1:
                    y_true_raw = y_true_raw.reshape(-1, 1)
                vals = y_true_raw[:, output_idx].astype(float)
                label = f'True (t={t_val:.3g})'
                cmap = self._trainer._get_colormap(output_idx)
            elif kind == 'err':
                if not self._get_has_solution():
                    return
                y_true_raw = self._trainer._call_solution(x_st)
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
                cmap = self._trainer._get_colormap(output_idx)
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
        out_name = self._trainer._get_output_name(output_idx)
        ax.set_title(f'{label} ({out_name})')
        ax.set_xlabel(self._trainer._get_input_name(spatial_dims[0]))
        ax.set_ylabel(self._trainer._get_input_name(spatial_dims[1]) if len(spatial_dims) > 1 else '')
        ax.set_aspect('equal')

    # ------------------------------------------------------------------
    # Snapshot helpers: 2D spatial heatmap at a fixed time value
    # Used for DomainCubic with n_spatial=2, has_time=True (e.g. Grey-Scott)
    # ------------------------------------------------------------------

    def _build_2d_spatial_grid_at_t(self, t_val, n_points):
        """Return (x0, x1, x_flat) for a spatial meshgrid with t=t_val appended."""
        x0 = np.linspace(self._get_xmin(0), self._get_xmax(0), n_points)
        x1 = np.linspace(self._get_xmin(1), self._get_xmax(1), n_points)
        X0, X1 = np.meshgrid(x0, x1)
        x_flat = np.column_stack([X0.ravel(), X1.ravel(), np.full(X0.size, float(t_val))])
        return x0, x1, X0, x_flat

    def _plot_solution_2d_at_time(self, ax, output_idx, t_val, n_points=50, plot_key='solution'):
        """Plot predicted 2D spatial snapshot at fixed time t_val as a heatmap."""
        cmap = self._trainer._get_colormap(output_idx)
        ikw = {'cmap': cmap}
        ikw.update(self._get_imshow_kwargs(plot_key))

        x0, x1, X0, x_flat = self._build_2d_spatial_grid_at_t(t_val, n_points)
        y = self._trainer.eval(x_flat)
        Y = y[:, output_idx].reshape(X0.shape)
        extent = [x0.min(), x0.max(), x1.min(), x1.max()]
        im = ax.imshow(Y, extent=extent, origin='lower', aspect='equal', **ikw)
        cbar = self._fig.colorbar(im, ax=ax)
        self._colorbars.append(cbar)
        output_name = self._trainer._get_output_name(output_idx)
        ax.set_title(f'Predicted ({output_name}, t={t_val:.3g})')
        ax.set_xlabel(self._trainer._get_input_name(0))
        ax.set_ylabel(self._trainer._get_input_name(1))

    def _plot_true_solution_2d_at_time(self, ax, output_idx, t_val, n_points=50, plot_key='solution'):
        """Plot true 2D spatial snapshot at fixed time t_val as a heatmap."""
        if not self._get_has_solution():
            return
        cmap = self._trainer._get_colormap(output_idx)
        ikw = {'cmap': cmap}
        ikw.update(self._get_imshow_kwargs(plot_key))

        x0, x1, X0, x_flat = self._build_2d_spatial_grid_at_t(t_val, n_points)
        y_true = self._trainer._call_solution(x_flat)
        if isinstance(y_true, (list, tuple)):
            y_true = np.concatenate([np.atleast_2d(yt).T if yt.ndim == 1 else yt for yt in y_true], axis=1)
        elif y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        Y_true = y_true[:, output_idx].reshape(X0.shape)
        extent = [x0.min(), x0.max(), x1.min(), x1.max()]
        im = ax.imshow(Y_true, extent=extent, origin='lower', aspect='equal', **ikw)
        cbar = self._fig.colorbar(im, ax=ax)
        self._colorbars.append(cbar)
        output_name = self._trainer._get_output_name(output_idx)
        ax.set_title(f'True ({output_name}, t={t_val:.3g})')
        ax.set_xlabel(self._trainer._get_input_name(0))
        ax.set_ylabel(self._trainer._get_input_name(1))

    def _plot_residuals_2d_at_time(self, ax, output_idx, t_val, n_points=50, plot_key='residuals'):
        """Plot |PDE residual| 2D spatial snapshot at fixed time t_val as a heatmap."""
        ikw = {'cmap': 'inferno'}
        ikw.update(self._get_imshow_kwargs(plot_key))

        x0, x1, X0, x_flat = self._build_2d_spatial_grid_at_t(t_val, n_points)
        residuals = self._trainer._compute_residuals(x_flat)
        if output_idx < len(residuals):
            Res = np.abs(residuals[output_idx]).reshape(X0.shape)
        else:
            Res = np.zeros(X0.shape)
        extent = [x0.min(), x0.max(), x1.min(), x1.max()]
        im = ax.imshow(Res, extent=extent, origin='lower', aspect='equal', **ikw)
        cbar = self._fig.colorbar(im, ax=ax, label='|Residual|')
        self._colorbars.append(cbar)
        output_name = self._trainer._get_output_name(output_idx)
        ax.set_title(f'PDE Residual ({output_name}, t={t_val:.3g})')
        ax.set_xlabel(self._trainer._get_input_name(0))
        ax.set_ylabel(self._trainer._get_input_name(1))

    def _plot_error_2d_at_time(self, ax, output_idx, t_val, n_points=50, plot_key='error'):
        """Plot |error| 2D spatial snapshot at fixed time t_val as a heatmap."""
        if not self._get_has_solution():
            return
        ikw = {'cmap': 'Reds'}
        ikw.update(self._get_imshow_kwargs(plot_key))

        x0, x1, X0, x_flat = self._build_2d_spatial_grid_at_t(t_val, n_points)
        y = self._trainer.eval(x_flat)
        y_true = self._trainer._call_solution(x_flat)
        if isinstance(y_true, (list, tuple)):
            y_true = np.concatenate([np.atleast_2d(yt).T if yt.ndim == 1 else yt for yt in y_true], axis=1)
        elif y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        Err = np.abs(y[:, output_idx] - y_true[:, output_idx]).reshape(X0.shape)
        extent = [x0.min(), x0.max(), x1.min(), x1.max()]
        im = ax.imshow(Err, extent=extent, origin='lower', aspect='equal', **ikw)
        cbar = self._fig.colorbar(im, ax=ax)
        self._colorbars.append(cbar)
        output_name = self._trainer._get_output_name(output_idx)
        ax.set_title(f'|Error| ({output_name}, t={t_val:.3g})')
        ax.set_xlabel(self._trainer._get_input_name(0))
        ax.set_ylabel(self._trainer._get_input_name(1))

    # ------------------------------------------------------------------
    # 1D snapshot helpers: overlaid lines at multiple t values
    # Used for DomainCubic with n_spatial=1, has_time=True, time_points=[...]
    # ------------------------------------------------------------------

    def _plot_solution_1d_at_times(self, ax, output_idx, t_vals, n_points=200):
        """Overlay predicted 1D snapshots u(x) at multiple time values."""
        x0 = np.linspace(self._get_xmin(0), self._get_xmax(0), n_points)
        colors = plt.cm.viridis(np.linspace(0, 1, len(t_vals)))
        for t_val, color in zip(t_vals, colors):
            x_full = np.column_stack([x0, np.full(n_points, float(t_val))])
            y = self._trainer.eval(x_full)
            ax.plot(x0, y[:, output_idx], color=color, linewidth=1.5, label=f't={t_val:.3g}')
        output_name = self._trainer._get_output_name(output_idx)
        ax.set_xlabel(self._trainer._get_input_name(0))
        ax.set_ylabel(output_name)
        ax.set_title(f'Solution ({output_name}) snapshots')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_residuals_1d_at_times(self, ax, output_idx, t_vals, n_points=200):
        """Overlay |PDE residual| 1D snapshots at multiple time values."""
        x0 = np.linspace(self._get_xmin(0), self._get_xmax(0), n_points)
        colors = plt.cm.inferno(np.linspace(0, 0.9, len(t_vals)))
        for t_val, color in zip(t_vals, colors):
            x_full = np.column_stack([x0, np.full(n_points, float(t_val))])
            residuals = self._trainer._compute_residuals(x_full)
            if output_idx < len(residuals):
                res = np.abs(residuals[output_idx]).flatten()
            else:
                res = np.zeros(n_points)
            ax.plot(x0, res, color=color, linewidth=1.5, label=f't={t_val:.3g}')
        output_name = self._trainer._get_output_name(output_idx)
        ax.set_xlabel(self._trainer._get_input_name(0))
        ax.set_ylabel(f'|Residual| ({output_name})')
        ax.set_title(f'PDE Residual ({output_name}) snapshots')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_true_solution_1d_at_times(self, ax, output_idx, t_vals, n_points=200):
        """Overlay true/reference 1D snapshots u(x) at multiple time values."""
        if not self._get_has_solution():
            return
        x0 = np.linspace(self._get_xmin(0), self._get_xmax(0), n_points)
        colors = plt.cm.viridis(np.linspace(0, 1, len(t_vals)))
        for t_val, color in zip(t_vals, colors):
            x_full = np.column_stack([x0, np.full(n_points, float(t_val))])
            y_true_raw = self._trainer._call_solution(x_full)
            if isinstance(y_true_raw, (list, tuple)):
                y_true_raw = np.concatenate(
                    [np.atleast_2d(yt).T if yt.ndim == 1 else yt for yt in y_true_raw], axis=1)
            elif y_true_raw.ndim == 1:
                y_true_raw = y_true_raw.reshape(-1, 1)
            ax.plot(x0, y_true_raw[:, output_idx], color=color, linewidth=1.5, label=f't={t_val:.3g}')
        output_name = self._trainer._get_output_name(output_idx)
        ax.set_xlabel(self._trainer._get_input_name(0))
        ax.set_ylabel(output_name)
        ax.set_title(f'True ({output_name}) snapshots')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_error_1d_at_times(self, ax, output_idx, t_vals, n_points=200):
        """Overlay |error| 1D snapshots at multiple time values."""
        if not self._get_has_solution():
            return
        x0 = np.linspace(self._get_xmin(0), self._get_xmax(0), n_points)
        colors = plt.cm.Reds(np.linspace(0.3, 1.0, len(t_vals)))
        for t_val, color in zip(t_vals, colors):
            x_full = np.column_stack([x0, np.full(n_points, float(t_val))])
            y_pred = self._trainer.eval(x_full)[:, output_idx]
            y_true_raw = self._trainer._call_solution(x_full)
            if isinstance(y_true_raw, (list, tuple)):
                y_true_raw = np.concatenate(
                    [np.atleast_2d(yt).T if yt.ndim == 1 else yt for yt in y_true_raw], axis=1)
            elif y_true_raw.ndim == 1:
                y_true_raw = y_true_raw.reshape(-1, 1)
            err = np.abs(y_pred - y_true_raw[:, output_idx])
            ax.plot(x0, err, color=color, linewidth=1.5, label=f't={t_val:.3g}')
        output_name = self._trainer._get_output_name(output_idx)
        ax.set_xlabel(self._trainer._get_input_name(0))
        ax.set_ylabel(f'|Error| ({output_name})')
        ax.set_title(f'Error ({output_name}) snapshots')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_solution_2d(self, ax, output_idx, n_points=50, plot_key='solution'):
        """Plot 2D solution as heatmap on given axes."""
        cmap = self._trainer._get_colormap(output_idx)
        ikw = {'cmap': cmap}
        ikw.update(self._get_imshow_kwargs(plot_key))

        if self._is_mesh_domain():
            dom = self._get_domain()
            tri = mtri.Triangulation(dom._vertices[:, 0], dom._vertices[:, 1], dom._faces)
            y = self._trainer.eval(dom._vertices)
            vals = y[:, output_idx]
            im = ax.tricontourf(tri, vals, levels=50, **ikw)
            ax.triplot(tri, color='gray', lw=0.3, alpha=0.3)
        else:
            x0 = np.linspace(self._get_xmin(0), self._get_xmax(0), n_points)
            x1 = np.linspace(self._get_xmin(1), self._get_xmax(1), n_points)
            X0, X1 = np.meshgrid(x0, x1)
            x_flat = np.column_stack([X0.ravel(), X1.ravel()])
            y = self._trainer.eval(x_flat)
            Y = y[:, output_idx].reshape(X0.shape)
            extent = [x0.min(), x0.max(), x1.min(), x1.max()]
            im = ax.imshow(Y, extent=extent, origin='lower', aspect='equal', **ikw)

        cbar = self._fig.colorbar(im, ax=ax)
        self._colorbars.append(cbar)

        output_name = self._trainer._get_output_name(output_idx)
        ax.set_title(f'Predicted ({output_name})')
        ax.set_xlabel(self._trainer._get_input_name(0))
        ax.set_ylabel(self._trainer._get_input_name(1))
    
    def _plot_solution_3d(self, ax, output_idx):
        """Plot predicted solution on a 3D surface mesh using Poly3DCollection."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import matplotlib.cm as _cm
        import matplotlib.colors as _mcolors

        dom = self._trainer.problem.domain
        verts = dom._vertices   # (N, 3)
        faces = dom._faces      # (F, 3)

        y = self._trainer.eval(verts)
        vals = np.array(y[:, output_idx], dtype=float)
        face_vals = vals[faces].mean(axis=1)
        v_min, v_max = face_vals.min(), face_vals.max()
        if v_min == v_max:
            v_max = v_min + 1.0

        cmap = _cm.get_cmap(self._trainer._get_colormap(output_idx))
        norm = _mcolors.Normalize(vmin=v_min, vmax=v_max)
        facecolors = cmap(norm(face_vals))

        # Remove only Poly3DCollection objects — never call ax.cla() on a 3D axis
        # (cla() detaches internal pane objects, breaking get_figure() → dpi_scale_trans)
        for col in list(ax.collections):
            col.remove()

        polys = verts[faces]
        coll = Poly3DCollection(polys, facecolors=facecolors, edgecolors='none', linewidths=0)
        ax.add_collection3d(coll)
        for dim, setter in enumerate([ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
            setter(verts[:, dim].min(), verts[:, dim].max())

        output_name = self._trainer._get_output_name(output_idx)
        ax.set_title(f'Predicted ({output_name})')
        ax.set_xlabel(self._trainer._get_input_name(0))
        ax.set_ylabel(self._trainer._get_input_name(1))
        ax.set_zlabel(self._trainer._get_input_name(2))

    def _plot_residuals_3d(self, ax, output_idx):
        """Plot PDE residual magnitudes on a 3D surface mesh using Poly3DCollection."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import matplotlib.cm as _cm
        import matplotlib.colors as _mcolors

        dom = self._trainer.problem.domain
        verts = dom._vertices   # (N, 3)
        faces = dom._faces      # (F, 3)

        try:
            residuals = self._trainer._compute_residuals(verts)
            if output_idx < len(residuals):
                vals = np.abs(np.array(residuals[output_idx], dtype=float)).flatten()
            else:
                vals = np.zeros(len(verts))
        except Exception:
            vals = np.zeros(len(verts))

        face_vals = vals[faces].mean(axis=1)
        v_min, v_max = face_vals.min(), face_vals.max()
        if v_min == v_max:
            v_max = v_min + 1.0

        cmap = _cm.get_cmap('inferno')
        norm = _mcolors.Normalize(vmin=v_min, vmax=v_max)
        facecolors = cmap(norm(face_vals))

        # Remove only Poly3DCollection objects — never call ax.cla() on a 3D axis
        for col in list(ax.collections):
            col.remove()

        polys = verts[faces]
        coll = Poly3DCollection(polys, facecolors=facecolors, edgecolors='none', linewidths=0)
        ax.add_collection3d(coll)
        for dim, setter in enumerate([ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
            setter(verts[:, dim].min(), verts[:, dim].max())

        output_name = self._trainer._get_output_name(output_idx)
        ax.set_title(f'PDE Residual ({output_name})')
        ax.set_xlabel(self._trainer._get_input_name(0))
        ax.set_ylabel(self._trainer._get_input_name(1))
        ax.set_zlabel(self._trainer._get_input_name(2))

    def _plot_true_solution_2d(self, ax, output_idx, n_points=50, plot_key='solution'):
        """Plot 2D true solution as heatmap on given axes."""
        if not self._get_has_solution():
            return

        cmap = self._trainer._get_colormap(output_idx)
        ikw = {'cmap': cmap}
        ikw.update(self._get_imshow_kwargs(plot_key))

        def _normalise(y_true):
            if isinstance(y_true, (list, tuple)):
                y_true = np.concatenate([np.atleast_2d(yt).T if yt.ndim == 1 else yt for yt in y_true], axis=1)
            elif y_true.ndim == 1:
                y_true = y_true.reshape(-1, 1)
            return y_true

        if self._is_mesh_domain():
            dom = self._get_domain()
            tri = mtri.Triangulation(dom._vertices[:, 0], dom._vertices[:, 1], dom._faces)
            y_true = _normalise(self._trainer._call_solution(dom._vertices))
            vals = y_true[:, output_idx]
            im = ax.tricontourf(tri, vals, levels=50, **ikw)
            ax.triplot(tri, color='gray', lw=0.3, alpha=0.3)
        else:
            x0 = np.linspace(self._get_xmin(0), self._get_xmax(0), n_points)
            x1 = np.linspace(self._get_xmin(1), self._get_xmax(1), n_points)
            X0, X1 = np.meshgrid(x0, x1)
            x_flat = np.column_stack([X0.ravel(), X1.ravel()])
            y_true = _normalise(self._trainer._call_solution(x_flat))
            Y_true = y_true[:, output_idx].reshape(X0.shape)
            extent = [x0.min(), x0.max(), x1.min(), x1.max()]
            im = ax.imshow(Y_true, extent=extent, origin='lower', aspect='equal', **ikw)

        cbar = self._fig.colorbar(im, ax=ax)
        self._colorbars.append(cbar)

        output_name = self._trainer._get_output_name(output_idx)
        ax.set_title(f'True Solution ({output_name})')
        ax.set_xlabel(self._trainer._get_input_name(0))
        ax.set_ylabel(self._trainer._get_input_name(1))
    
    def _plot_error_1d(self, ax, output_idx, n_points=200):
        """Plot 1D absolute error."""
        if not self._get_has_solution():
            return
        
        x = np.linspace(self._get_xmin(0), self._get_xmax(0), n_points).reshape(-1, 1)
        y = self._trainer.eval(x)
        
        y_true = self._trainer._call_solution(x)
        if isinstance(y_true, (list, tuple)):
            y_true = np.concatenate([np.atleast_2d(yt).T if yt.ndim == 1 else yt for yt in y_true], axis=1)
        elif y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        
        error = np.abs(y[:, output_idx] - y_true[:, output_idx])
        ax.plot(x, error, 'r-', linewidth=2)
        
        output_name = self._trainer._get_output_name(output_idx)
        input_name = self._trainer._get_input_name(0)
        ax.set_xlabel(input_name)
        ax.set_ylabel(f'|Error| ({output_name})')
        ax.set_title(f'Absolute Error ({output_name})')
        ax.grid(True, alpha=0.3)
    
    def _plot_error_2d(self, ax, output_idx, n_points=50, plot_key='error'):
        """Plot 2D absolute error as heatmap."""
        if not self._get_has_solution():
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
            dom = self._get_domain()
            tri = mtri.Triangulation(dom._vertices[:, 0], dom._vertices[:, 1], dom._faces)
            y = self._trainer.eval(dom._vertices)
            y_true = _normalise(self._trainer._call_solution(dom._vertices))
            error = np.abs(y[:, output_idx] - y_true[:, output_idx])
            im = ax.tricontourf(tri, error, levels=50, **ikw)
            ax.triplot(tri, color='gray', lw=0.3, alpha=0.3)
        else:
            x0 = np.linspace(self._get_xmin(0), self._get_xmax(0), n_points)
            x1 = np.linspace(self._get_xmin(1), self._get_xmax(1), n_points)
            X0, X1 = np.meshgrid(x0, x1)
            x_flat = np.column_stack([X0.ravel(), X1.ravel()])
            y = self._trainer.eval(x_flat)
            y_true = _normalise(self._trainer._call_solution(x_flat))
            error = np.abs(y[:, output_idx] - y_true[:, output_idx]).reshape(X0.shape)
            extent = [x0.min(), x0.max(), x1.min(), x1.max()]
            im = ax.imshow(error, extent=extent, origin='lower', aspect='equal', **ikw)

        cbar = self._fig.colorbar(im, ax=ax)
        self._colorbars.append(cbar)

        output_name = self._trainer._get_output_name(output_idx)
        ax.set_title(f'Absolute Error ({output_name})')
        ax.set_xlabel(self._trainer._get_input_name(0))
        ax.set_ylabel(self._trainer._get_input_name(1))
    
    def _plot_region_nd(self, ax, output_idx, region, n_points=50):
        """Plot a region of an N-dimensional solution as 1D or 2D."""
        free_dims, free_ranges, fixed_dims, fixed_values = self._parse_region_nd(region)
        n_free = len(free_dims)
        
        if n_free == 0:
            # No free dimensions - just show the single point value
            x_point = np.zeros((1, self._get_n_dims()))
            for i, val in zip(fixed_dims, fixed_values):
                x_point[0, i] = val
            y = self._trainer.eval(x_point)
            ax.text(0.5, 0.5, f'u={y[0, output_idx]:.4f}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(self._trainer._get_output_name(output_idx))
            return
        
        elif n_free == 1:
            # 1D plot
            dim = free_dims[0]
            x_range = free_ranges[0]
            x_vals = np.linspace(x_range[0], x_range[1], n_points)
            
            x_full = np.zeros((n_points, self._get_n_dims()))
            x_full[:, dim] = x_vals
            for i, val in zip(fixed_dims, fixed_values):
                x_full[:, i] = val
            
            y = self._trainer.eval(x_full)
            
            ax.plot(x_vals, y[:, output_idx], linewidth=2)
            ax.set_xlabel(self._trainer._get_input_name(dim))
            ax.set_ylabel(self._trainer._get_output_name(output_idx))
            
            title_parts = [self._trainer._get_output_name(output_idx)]
            if fixed_dims:
                fixed_str = ', '.join([f'{self._trainer._get_input_name(d)}={v:.3g}' 
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
            x_full = np.zeros((n_total, self._get_n_dims()))
            x_full[:, dim0] = X0.ravel()
            x_full[:, dim1] = X1.ravel()
            for i, val in zip(fixed_dims, fixed_values):
                x_full[:, i] = val
            
            y = self._trainer.eval(x_full)
            Y = y[:, output_idx].reshape(X0.shape)
            
            extent = [x0.min(), x0.max(), x1.min(), x1.max()]
            cmap = self._trainer._get_colormap(output_idx)
            im = ax.imshow(Y, extent=extent, origin='lower', aspect='equal', cmap=cmap)
            cbar = self._fig.colorbar(im, ax=ax)
            self._colorbars.append(cbar)
            
            ax.set_xlabel(self._trainer._get_input_name(dim0))
            ax.set_ylabel(self._trainer._get_input_name(dim1))
            
            output_name = self._trainer._get_output_name(output_idx)
            if fixed_dims:
                fixed_str = ', '.join([f'{self._trainer._get_input_name(d)}={v:.3g}' 
                                       for d, v in zip(fixed_dims, fixed_values)])
                ax.set_title(f'{output_name} at {fixed_str}')
            else:
                ax.set_title(output_name)
        else:
            ax.text(0.5, 0.5, f'Cannot plot {n_free}D (max 2D)',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(self._trainer._get_output_name(output_idx))
    
    def _plot_region_residuals_nd(self, ax, residual_idx, region, n_points=50):
        """Plot residuals for a region of an N-dimensional problem as 1D or 2D."""
        free_dims, free_ranges, fixed_dims, fixed_values = self._parse_region_nd(region)
        n_free = len(free_dims)
        
        if n_free == 0:
            # No free dimensions - just show the single point residual value
            x_point = np.zeros((1, self._get_n_dims()))
            for i, val in zip(fixed_dims, fixed_values):
                x_point[0, i] = val
            residuals = self._trainer._compute_residuals(x_point)
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
            
            x_full = np.zeros((n_points, self._get_n_dims()))
            x_full[:, dim] = x_vals
            for i, val in zip(fixed_dims, fixed_values):
                x_full[:, i] = val
            
            residuals = self._trainer._compute_residuals(x_full)
            if residual_idx < len(residuals):
                res = np.abs(residuals[residual_idx])
            else:
                res = np.zeros(n_points)
            
            ax.plot(x_vals, res, 'm-', linewidth=2)
            ax.set_xlabel(self._trainer._get_input_name(dim))
            ax.set_ylabel(f'|Residual eq{residual_idx+1}|')
            
            title_parts = [f'Residual eq{residual_idx+1}']
            if fixed_dims:
                fixed_str = ', '.join([f'{self._trainer._get_input_name(d)}={v:.3g}' 
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
            x_full = np.zeros((n_total, self._get_n_dims()))
            x_full[:, dim0] = X0.ravel()
            x_full[:, dim1] = X1.ravel()
            for i, val in zip(fixed_dims, fixed_values):
                x_full[:, i] = val
            
            residuals = self._trainer._compute_residuals(x_full)
            if residual_idx < len(residuals):
                Res = np.abs(residuals[residual_idx]).reshape(X0.shape)
            else:
                Res = np.zeros(X0.shape)
            
            extent = [x0.min(), x0.max(), x1.min(), x1.max()]
            im = ax.imshow(Res, extent=extent, origin='lower', aspect='equal', cmap='viridis')
            cbar = self._fig.colorbar(im, ax=ax, label='|Residual|')
            self._colorbars.append(cbar)
            
            ax.set_xlabel(self._trainer._get_input_name(dim0))
            ax.set_ylabel(self._trainer._get_input_name(dim1))
            
            if fixed_dims:
                fixed_str = ', '.join([f'{self._trainer._get_input_name(d)}={v:.3g}' 
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
        self._axes = axes   # make axes accessible to plot helpers (e.g. cax lookup)
        n_dims = self._get_n_dims()
        n_outputs = self._get_n_outputs()
        has_solution = self._get_has_solution()
        
        self._clear_colorbars()
        
        for key, ax in axes.items():
            if key.startswith('cax_'):
                continue  # colorbar axes — never clear (inset_axes would lose figure ref)
            if key.startswith('mesh3d_'):
                continue  # 3-D mesh axes — collection & colorbar updated in-place
            if hasattr(ax, 'clear'):
                ax.clear()
        
        if 'losses' in axes:
            self._plot_losses(axes['losses'])
            self._apply_plot_kwargs(axes['losses'], 'losses')

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
            _dom2 = self._get_domain()
            _has_time2 = getattr(_dom2, 'has_time', False)
            _pts2 = self._plot_time_points
            if _has_time2 and _pts2 is not None:
                # 1+1D: overlaid 1D snapshots at the requested time values
                for i in range(n_outputs):
                    if f'sol_{i}' in axes:
                        self._plot_solution_1d_at_times(axes[f'sol_{i}'], i, _pts2, n_points)
                        self._apply_plot_kwargs(axes[f'sol_{i}'], 'solution')
                    if f'true_{i}' in axes and has_solution:
                        self._plot_true_solution_1d_at_times(axes[f'true_{i}'], i, _pts2, n_points)
                        self._apply_plot_kwargs(axes[f'true_{i}'], 'solution')
                    if f'res_{i}' in axes:
                        self._plot_residuals_1d_at_times(axes[f'res_{i}'], i, _pts2, n_points)
                        self._apply_plot_kwargs(axes[f'res_{i}'], 'residuals')
                    if f'err_{i}' in axes and has_solution:
                        self._plot_error_1d_at_times(axes[f'err_{i}'], i, _pts2, n_points)
                        self._apply_plot_kwargs(axes[f'err_{i}'], 'error')
            else:
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

        elif self._is_3d_mesh_domain() and not self._plot_time_points:
            for i in range(n_outputs):
                if f'mesh3d_sol_{i}' in axes:
                    self._plot_solution_3d(axes[f'mesh3d_sol_{i}'], i)
                    self._apply_plot_kwargs(axes[f'mesh3d_sol_{i}'], 'solution')
                if f'mesh3d_res_{i}' in axes:
                    self._plot_residuals_3d(axes[f'mesh3d_res_{i}'], i)
                    self._apply_plot_kwargs(axes[f'mesh3d_res_{i}'], 'residuals')

        # ── Cubic 2+1D spatial snapshots at fixed times (e.g. Grey-Scott) ──────
        _pts = self._plot_time_points
        if _pts and not self._is_mesh_domain():
            _dom_c = self._get_domain()
            if (_dom_c is not None
                    and getattr(_dom_c, 'has_time', False)
                    and getattr(_dom_c, '_spatial_dims', 0) >= 2):
                _np_snap = max(n_points // 4, 30)
                for _t_val in _pts:
                    for _i in range(n_outputs):
                        _sol_key  = f'snap_sol_{_i}_t{_t_val}'
                        _true_key = f'snap_true_{_i}_t{_t_val}'
                        _res_key  = f'snap_res_{_i}_t{_t_val}'
                        _err_key  = f'snap_err_{_i}_t{_t_val}'
                        if _sol_key in axes:
                            self._plot_solution_2d_at_time(axes[_sol_key], _i, _t_val, _np_snap)
                            self._apply_plot_kwargs(axes[_sol_key], 'solution')
                        if _true_key in axes and has_solution:
                            self._plot_true_solution_2d_at_time(axes[_true_key], _i, _t_val, _np_snap)
                            self._apply_plot_kwargs(axes[_true_key], 'solution')
                        if _res_key in axes:
                            self._plot_residuals_2d_at_time(axes[_res_key], _i, _t_val, _np_snap)
                            self._apply_plot_kwargs(axes[_res_key], 'residuals')
                        if _err_key in axes and has_solution:
                            self._plot_error_2d_at_time(axes[_err_key], _i, _t_val, _np_snap)
                            self._apply_plot_kwargs(axes[_err_key], 'error')

        # ── Transient mesh snapshots (works regardless of n_dims) ──────────────
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
        obs_spatial = list(getattr(self._trainer.problem, 'obs_spatial', None) or [])
        for obs_name in (getattr(self._trainer.problem, 'obs_names', None) or []):
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
        regions = self.regions
        n_outputs = self._get_n_outputs()
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
        
        # Plot inferred parameter histories
        _fit_params = self._get_fit_params()
        if _fit_params:
            self._plot_parameters(axes)

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
        if isinstance(self._trainer.problem, _ProblemWeak):
            self._plot_weak_residuals_on_mesh(ax, output_idx)
            return
        x_np = np.linspace(self._get_xmin(0), self._get_xmax(0), n_points).reshape(-1, 1)
        
        residuals = self._trainer._compute_residuals(x_np)
        
        if output_idx < len(residuals):
            res = np.abs(residuals[output_idx]).flatten()
        else:
            res = np.zeros(n_points)
        
        ax.plot(x_np.flatten(), res, 'm-', linewidth=2)
        
        output_name = self._trainer._get_output_name(output_idx)
        input_name = self._trainer._get_input_name(0)
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

        R_raw = weak_res_fn(self._trainer.model.params)

        dom        = self._trainer.problem.domain
        verts_xy   = dom._vertices                        # (n_verts, 2)
        faces      = dom._faces                           # (n_faces, 3)
        n_verts    = len(verts_xy)

        # make_residual_fn returns dict {term_name: array of shape (n_free * n_comp,)}
        # where n_free = number of truly-free nodes (Dirichlet nodes excluded).
        # Reconstruct a full-vertex array with NaN at constrained nodes.
        R_verts = np.full(n_verts, np.nan, dtype=float)

        free_nodes = getattr(self._trainer.problem, 'free_nodes', None)
        if isinstance(R_raw, dict) and free_nodes is not None:
            import jax.numpy as _jnp
            # Reproduce the same true-free list used in make_residual_fn
            _dset: set = set()
            _domain = self._trainer.problem.domain
            for _bc in self._trainer.problem.boundary_conditions:
                if _bc.kind == 'dirichlet':
                    _region = getattr(_bc, 'region', None)
                    if _region and _region in _domain._boundary_regions:
                        _ni = _domain._boundary_regions[_region].get('node_indices')
                        if _ni is not None:
                            _dset.update(int(i) for i in _ni)
                    elif hasattr(_bc, 'node_indices') and _bc.node_indices is not None:
                        _dset.update(int(i) for i in _bc.node_indices)
            _all_free = set(int(i) for i in free_nodes if int(i) < n_verts)
            true_free = sorted(_all_free - _dset)
            n_free = len(true_free)
            if n_free > 0:
                # Average absolute residuals across all terms and components
                R_flat = np.array(_jnp.concatenate(list(R_raw.values())))
                n_comp = len(R_flat) // n_free
                R_per_node = np.zeros(n_free, dtype=float)
                for _k in range(max(n_comp, 1)):
                    R_per_node += np.abs(R_flat[_k * n_free: (_k + 1) * n_free])
                R_per_node /= max(n_comp, 1)
                for _idx, _node in enumerate(true_free):
                    R_verts[_node] = R_per_node[_idx]
        elif not isinstance(R_raw, dict):
            # Legacy: flat (n_dofs,) vector
            R_flat = np.abs(np.array(R_raw)[:n_verts]).astype(float)
            if free_nodes is not None:
                free_set = set(int(i) for i in free_nodes if int(i) < n_verts)
                for _i, _v in enumerate(R_flat):
                    if _i in free_set:
                        R_verts[_i] = _v
            else:
                R_verts = R_flat

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
        ax.set_xlabel(self._trainer._get_input_name(0))
        ax.set_ylabel(self._trainer._get_input_name(1))

    def _plot_residuals_2d(self, ax, output_idx, n_points=50, plot_key='residuals'):
        """Plot 2D PDE residuals as heatmap."""
        from pinns.problems.problem_weak import ProblemWeak as _ProblemWeak
        if isinstance(self._trainer.problem, _ProblemWeak):
            self._plot_weak_residuals_on_mesh(ax, output_idx)
            return
        ikw = {'cmap': 'viridis'}
        ikw.update(self._get_imshow_kwargs(plot_key))

        if self._is_mesh_domain():
            dom = self._trainer.problem.domain
            tri = mtri.Triangulation(dom._vertices[:, 0], dom._vertices[:, 1], dom._faces)
            residuals = self._trainer._compute_residuals(dom._vertices)
            if output_idx < len(residuals):
                res = np.abs(residuals[output_idx]).flatten()
            else:
                res = np.zeros(dom._vertices.shape[0])
            im = ax.tricontourf(tri, res, levels=50, **ikw)
            ax.triplot(tri, color='gray', lw=0.3, alpha=0.3)
        else:
            x0 = np.linspace(self._get_xmin(0), self._get_xmax(0), n_points)
            x1 = np.linspace(self._get_xmin(1), self._get_xmax(1), n_points)
            X0, X1 = np.meshgrid(x0, x1)
            x_np = np.column_stack([X0.ravel(), X1.ravel()])
            try:
                residuals = self._trainer._compute_residuals(x_np)
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

        output_name = self._trainer._get_output_name(output_idx)
        ax.set_title(f'PDE Residual ({output_name})')
        ax.set_xlabel(self._trainer._get_input_name(0))
        ax.set_ylabel(self._trainer._get_input_name(1))

    # ==================== Observable Plotting ====================

    def _plot_observable_1d(self, ax, obs_name: str, n_points: int = 200):
        """Plot a 1D observable field on the given axes."""
        x_np = np.linspace(self._trainer.problem.xmin[0], self._trainer.problem.xmax[0], n_points).reshape(-1, 1)
        obs = self._trainer._evaluate_observables(x_np)
        if obs_name not in obs:
            ax.text(0.5, 0.5, f'Observable\n{obs_name!r}\nnot available',
                    ha='center', va='center', transform=ax.transAxes)
            return
        vals = obs[obs_name]   # (n, 1) or (n,)
        ax.plot(x_np.flatten(), vals.flatten(), linewidth=2)
        ax.set_xlabel(self._trainer._get_input_name(0))
        ax.set_ylabel(obs_name)
        ax.set_title(f'Observable: {obs_name}')
        ax.grid(True, alpha=0.3)

    def _plot_observable_2d(self, ax, obs_name: str, n_points: int = 50):
        """Plot a scalar 2D observable as a filled contour / heatmap."""
        ikw = {'cmap': 'viridis'}
        if self._is_mesh_domain():
            dom = self._trainer.problem.domain
            x_np = dom._vertices
            obs = self._trainer._evaluate_observables(x_np)
        else:
            x0 = np.linspace(self._trainer.problem.xmin[0], self._trainer.problem.xmax[0], n_points)
            x1 = np.linspace(self._trainer.problem.xmin[1], self._trainer.problem.xmax[1], n_points)
            X0, X1 = np.meshgrid(x0, x1)
            x_np = np.column_stack([X0.ravel(), X1.ravel()])
            obs = self._trainer._evaluate_observables(x_np)

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
        ax.set_xlabel(self._trainer._get_input_name(0))
        ax.set_ylabel(self._trainer._get_input_name(1))

    def _plot_deformed_mesh_spatial(self, ax_ref, ax_def, obs_spatial: list, n_points: int = 50):
        """Side-by-side deformed-mesh plot.

        ``obs_spatial`` is an ordered list of observable names whose scalar
        values are the **absolute new positions** of each node (e.g. ``x + u1``,
        ``y + u2``).

        * Left panel (``ax_ref``): original (undeformed) mesh
        * Right panel (``ax_def``): deformed mesh coloured by ``‖displacement‖``
        """
        if self._is_mesh_domain():
            dom = self._trainer.problem.domain
            x_np = dom._vertices   # (n, 2)
        else:
            x0 = np.linspace(self._trainer.problem.xmin[0], self._trainer.problem.xmax[0], n_points)
            x1 = np.linspace(self._trainer.problem.xmin[1], self._trainer.problem.xmax[1], n_points)
            X0, X1 = np.meshgrid(x0, x1)
            x_np = np.column_stack([X0.ravel(), X1.ravel()])

        obs = self._trainer._evaluate_observables(x_np)

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

        xlabel = self._trainer._get_input_name(0)
        ylabel = self._trainer._get_input_name(1)

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
        x_np = np.linspace(self._trainer.problem.xmin[0], self._trainer.problem.xmax[0], n_points).reshape(-1, 1)
        
        predictions, windows = self._get_subdomain_predictions_np(x_np)
        
        if predictions is None:
            return  # Not an FBPINN network
        
        if hasattr(self._trainer.model, 'domain'):
            lower_bounds, upper_bounds = self._trainer.model.domain.get_subdomain_bounds()
            n_subdomains = self._trainer.model.n_subdomains
            colors = plt.cm.tab10(np.linspace(0, 1, min(n_subdomains, 10)))
            
            for i in range(n_subdomains):
                pred_i = predictions[:, i, output_idx]
                
                # Unnormalize if needed
                if hasattr(self._trainer.model, 'output_range_min') and self._trainer.model.output_range_min is not None:
                    if hasattr(self._trainer.model, 'unnormalize_output') and self._trainer.model.unnormalize_output:
                        y_min = np.array(self._trainer.model.output_range_min[output_idx])
                        y_max = np.array(self._trainer.model.output_range_max[output_idx])
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
        
        if hasattr(self._trainer.model, 'domain') and hasattr(self._trainer.model.domain, 'get_subdomain_bounds'):
            lower_bounds, upper_bounds = self._trainer.model.domain.get_subdomain_bounds()
            n_subdomains = self._trainer.model.n_subdomains
            
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
        
        if self._trainer.train_samples[0] > 0:
            x_train = self._trainer.problem.domain.sample_interior(
                self._trainer.train_samples[0], rng=self._trainer.rng)
            n_train = len(x_train)
            train_size = max(5, min(50, 1000 / n_train))
            y_train = np.full(n_train, y_min + 0.02 * y_range)
            ax.scatter(x_train[:, 0], y_train, s=train_size, c=train_color,
                      alpha=1, marker='|', label=f'Train ({n_train})', zorder=5)

    def _plot_sampling_points_2d(self, ax, cmap='viridis'):
        """Plot training sampling points on 2D axes."""
        train_color = '#15B01A'
        bc_color = '#15B01A'
        
        if self._trainer.train_samples[0] > 0:
            x_train = self._trainer.problem.domain.sample_interior(
                self._trainer.train_samples[0], rng=self._trainer.rng)
            n_train = len(x_train)
            train_size = max(1, min(20, 500 / n_train))
            ax.scatter(x_train[:, 0], x_train[:, 1], s=train_size, c=train_color,
                      alpha=1, marker='.', label=f'Train ({n_train})', zorder=5)
        
        for i, bc in enumerate(self._trainer.problem.boundary_conditions):
            if self._trainer.train_samples[i + 1] > 0:
                x_bc = self._trainer._sample_bc_np(bc, self._trainer.train_samples[i + 1])
                n_bc = len(x_bc)
                bc_size = max(2, min(30, 300 / n_bc))
                ax.scatter(x_bc[:, 0], x_bc[:, 1], s=bc_size, c=bc_color,
                          alpha=1, marker='x', zorder=6)

    # ==================== Plot snapshot (rollout override) ====================

    def _plot_mesh_snapshot(self, ax, output_idx, t_val, kind='sol'):
        """Override: for BPTT rollout, fetch the correct time-step from predict_rollout."""
        from pinns.problems.problem_weak import ProblemWeak as _PW
        import numpy as _np

        _is_rollout = (
            isinstance(self._trainer.problem, _PW)
            and getattr(self._trainer.problem.domain, '_time_mode', None) == 'discrete'
            and getattr(self._trainer.problem, 'n_time_steps', None) is not None
            and hasattr(self._trainer.model, 'predict_rollout')
        )
        if not _is_rollout or kind not in ('sol', 'err', 'true', 'res'):
            return self._plot_mesh_snapshot_base(ax, output_idx, t_val, kind)

        domain = self._trainer.problem.domain
        # Always roll out the full domain horizon for plotting
        n_steps = domain.n_steps
        dt = float(domain.dt)
        t_points = _np.array(domain._time_points)  # (n_steps+1,)

        # run rollout over the full domain
        try:
            u_all = self._trainer.model.predict_rollout(n_steps=n_steps, dt=dt)  # (n_steps+1, n_nodes)
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

        elif kind == 'err' and self._get_has_solution():
            x_ref = _np.hstack([verts_xy, _np.full((len(verts_xy), 1), t_actual, dtype=_np.float32)])
            y_true = self._trainer._call_solution(x_ref)
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

        elif kind == 'true' and self._get_has_solution():
            x_ref = _np.hstack([verts_xy, _np.full((len(verts_xy), 1), t_actual, dtype=_np.float32)])
            y_true = self._trainer._call_solution(x_ref)
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
        cd   = self._trainer.problem.cubature_data
        phi  = _np.array(cd['phi'],      dtype=float)   # (F, Q, L)
        gph  = _np.array(cd['grad_phi'], dtype=float)   # (F, Q, L, 2)
        wts  = _np.array(cd['weights'],  dtype=float)   # (F, Q)
        nid  = _np.array(cd['node_ids'], dtype=int)     # (F, L)
        dt   = float(self._trainer.problem.domain.dt)
        kappa = float(self._trainer.problem.params.get('kappa', 1.0))

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
