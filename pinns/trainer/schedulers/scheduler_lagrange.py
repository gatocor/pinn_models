"""
SchedulerLagrange – Augmented Lagrangian (AL) training for PINNs.

Instead of minimising a weighted sum of squared residuals the trainer
maximises constraint satisfaction through Lagrange multipliers:

    L(θ, λ) = Σ_k  w_k ‖g_k(θ)‖²  +  Σ_k  λ_k · g_k(θ)

where ``g_k`` are the residual vectors (PDE + BCs) and ``λ_k`` are the
per-sample multipliers updated by dual ascent after every gradient step.

Usage
-----
::

    from pinns.schedulers import SchedulerLagrange

    trainer.compile(
        ...
        schedulers=[SchedulerLagrange(lr=1.0)],
    )

Parameters
----------
constraints : None | list[str] | callable
    Which terms to apply multipliers to.
    * ``None``            – all terms (default)
    * list of names       – only the named terms
    * ``callable(name)``  – predicate returning ``True`` to include
lr : float
    Step size for dual-ascent multiplier update.
max_val : float
    Clip multipliers to ``[-max_val, max_val]`` after each update.
optimizer : str
    Multiplier update rule: ``'adam'`` (default) or ``'sgd'``.
    ``'sgd'`` is plain gradient ascent: ``λ ← λ + lr * g``.
"""

import time
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .scheduler_base import Scheduler
from ..schedulers.base import is_notebook


class SchedulerLagrange(Scheduler):
    """
    Augmented Lagrangian training scheduler.

    See module docstring for full description.
    """

    def __init__(
        self,
        constraints=None,
        lr: float = 1.0,
        max_val: float = 1e6,
        optimizer: str = 'adam',
    ):
        self.constraints = constraints          # None | list | callable
        self.lr = lr
        self.max_val = max_val
        self.optimizer_name = optimizer

        # Per-term Lagrange multiplier arrays (populated in on_compile)
        self.lagrange_multipliers: dict = {}
        self._lagrange_optimizer = None
        self._lagrange_opt_states: dict = {}
        self._resolved_constraints = None  # final list[str] | None

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_compile(self, trainer) -> None:
        """Resolve which terms to constrain and initialise λ vectors."""
        self._resolved_constraints = self._resolve_constraints(trainer)
        self._init_lambdas(trainer)

    def on_epoch_start(self, trainer, epoch: int) -> None:
        """Resize λ vectors if data shape changed (e.g. after SchedulerResample)."""
        self.reinitialize_if_needed(trainer)

    # ------------------------------------------------------------------
    # Public helpers (called by SchedulerCurriculum and the Trainer)
    # ------------------------------------------------------------------

    def reinitialize_if_needed(self, trainer) -> None:
        """Resize λ if train data changed size."""
        for name, data in trainer._train_data.items():
            if name not in self.lagrange_multipliers:
                continue
            if len(self.lagrange_multipliers[name]) != len(data):
                self.lagrange_multipliers[name] = jnp.zeros(len(data))
                if self._lagrange_optimizer is not None:
                    self._lagrange_opt_states[name] = self._lagrange_optimizer.init(
                        self.lagrange_multipliers[name])

    def get_statistics(self) -> dict:
        """Return mean/std/min/max of each λ vector."""
        return {
            name: {
                'mean': float(jnp.mean(lam)),
                'std':  float(jnp.std(lam)),
                'min':  float(jnp.min(lam)),
                'max':  float(jnp.max(lam)),
            }
            for name, lam in self.lagrange_multipliers.items()
        }

    def reset(self, trainer) -> None:
        """Reset all λ to zero."""
        for name in self.lagrange_multipliers:
            self.lagrange_multipliers[name] = jnp.zeros_like(
                self.lagrange_multipliers[name])

    # ------------------------------------------------------------------
    # Full AL training loop (called by Trainer.train())
    # ------------------------------------------------------------------

    def run_training(self, trainer) -> None:
        """Run the complete Augmented-Lagrangian training loop."""
        from ..schedulers.base import is_notebook as _is_nb
        from pinns.functional import make_derivative_fn

        epochs     = trainer._epochs
        print_each = trainer._print_each
        show_plots = trainer._show_plots
        save_plots = trainer._save_plots

        params_dict  = trainer._build_params()
        weights_dict = trainer._list_to_dict_weights(trainer.weights)

        compute_al_loss, _ = self._build_al_loss_fn(trainer, params_dict)
        lag_lr_ratio = self.lr / max(float(
            trainer.learning_rate(0) if callable(trainer.learning_rate)
            else trainer.learning_rate), 1e-12)

        @jax.jit
        def train_step(params, opt_state, train_data, lagrange_dict,
                       weights_dict, targets_dict):
            (loss, (losses, residuals)), grads = jax.value_and_grad(
                compute_al_loss, has_aux=True)(
                params, train_data, lagrange_dict, weights_dict, targets_dict)
            updates, new_opt_state = trainer.optimizer.update(
                grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss, losses, residuals

        start_time  = time.time()
        start_epoch = trainer._global_epoch
        lr_scheduler = getattr(trainer, '_lr_scheduler', None)
        train_targets = getattr(trainer, '_train_targets', {})

        if show_plots:
            if trainer._fig is None:
                trainer._fig, trainer._axes = trainer._create_figure()
            _, _, trainer._display_handle = trainer.plot_progress(
                save_path=None, n_points=trainer._plot_n_points,
                fig=trainer._fig, axes=trainer._axes,
                display_handle=trainer._display_handle,
            )

        print(f"Starting Trainer (Lagrangian mode) for {epochs} epochs …")

        # Epoch 0 metrics
        if print_each > 0:
            _, compute_res = self._build_al_loss_fn(trainer, params_dict)
            res0 = compute_res(trainer.network.params, trainer._train_data,
                               train_targets)
            bc_names = trainer._get_soft_bc_names() if hasattr(trainer, '_get_soft_bc_names') else trainer._get_bc_names()
            pde_mse0 = float(jnp.mean(res0['pde'] ** 2)) if 'pde' in res0 else 0.0
            bc_mse0  = [float(jnp.mean(res0[n] ** 2)) if n in res0 else 0.0
                        for n in bc_names]
            mse0 = float(sum(jnp.mean(g ** 2) for g in res0.values()))
            trainer.history['epoch'].append(start_epoch)
            trainer.history['loss'].append(mse0)
            trainer.history['train_loss'].append(mse0)
            trainer.history['loss_pde'].append([pde_mse0])
            trainer.history['loss_bcs'].append(bc_mse0)
            bc_str0 = ', '.join(f'{bc_names[i]}: {bc_mse0[i]:.2e}'
                                 for i in range(len(bc_names)))
            print(f"Epoch 0/{epochs} | MSE: {mse0:.2e} | PDE: {pde_mse0:.2e}"
                  f" | BCs: [{bc_str0}]")

        for epoch in range(start_epoch, start_epoch + epochs):
            # Non-Lagrange schedulers epoch_start (curriculum, resample…)
            for s in getattr(trainer, '_schedulers', []):
                if s is not self:
                    s.on_epoch_start(trainer, epoch - start_epoch)

            self.reinitialize_if_needed(trainer)

            # Learning rate schedule
            if (lr_scheduler is not None
                    and trainer.optimizer_name not in ('lbfgs', 'soap')
                    and hasattr(trainer.opt_state, 'hyperparams')):
                new_lr = lr_scheduler.lr(trainer.learning_rate, epoch)
                hp = dict(trainer.opt_state.hyperparams)
                hp['learning_rate'] = new_lr
                trainer.opt_state = trainer.opt_state._replace(hyperparams=hp)
                self.lr = lag_lr_ratio * new_lr
                for k in self._lagrange_opt_states:
                    if hasattr(self._lagrange_opt_states[k], 'hyperparams'):
                        lhp = dict(self._lagrange_opt_states[k].hyperparams)
                        lhp['learning_rate'] = self.lr
                        self._lagrange_opt_states[k] = \
                            self._lagrange_opt_states[k]._replace(hyperparams=lhp)

            (trainer.network.params, trainer.opt_state,
             loss, losses, residuals) = train_step(
                trainer.network.params, trainer.opt_state,
                trainer._train_data, self.lagrange_multipliers,
                weights_dict, train_targets,
            )
            self._update_lambdas(residuals)

            if (print_each > 0
                    and ((epoch + 1) % print_each == 0
                         or epoch == start_epoch + epochs - 1)):
                al_loss  = float(loss)
                mse_loss = float(sum(jnp.mean(g ** 2) for g in residuals.values()))
                bc_names = (trainer._get_soft_bc_names()
                             if hasattr(trainer, '_get_soft_bc_names')
                             else trainer._get_bc_names())
                pde_mse  = float(jnp.mean(residuals['pde'] ** 2)) if 'pde' in residuals else 0.0
                bc_mse   = [float(jnp.mean(residuals[n] ** 2)) if n in residuals else 0.0
                             for n in bc_names]

                trainer.history['epoch'].append(epoch)
                trainer.history['loss'].append(mse_loss)
                trainer.history['train_loss'].append(mse_loss)
                trainer.history['loss_pde'].append([pde_mse])
                trainer.history['loss_bcs'].append(bc_mse)
                trainer.history.setdefault('al_loss', []).append(al_loss)

                elapsed = time.time() - start_time
                if trainer.problem.solution is not None:
                    sol_err = trainer._compute_solution_error()
                    trainer.history['solution_error'].append(sol_err)

                bc_str = ', '.join(f'{bc_names[i]}: {bc_mse[i]:.2e}'
                                    for i in range(len(bc_names)))
                msg = (f"Epoch {epoch + 1}/{trainer._epochs + start_epoch}"
                       f" | AL: {al_loss:.2e} | MSE: {mse_loss:.2e}"
                       f" | PDE: {pde_mse:.2e} | BCs: [{bc_str}]"
                       f" | Time: {elapsed:.1f}s")
                if trainer.problem.solution is not None:
                    msg += f" | Error: {trainer.history['solution_error'][-1]:.2e}"
                print(msg)

                if show_plots:
                    _, _, trainer._display_handle = trainer.plot_progress(
                        save_path=None, n_points=trainer._plot_n_points,
                        fig=trainer._fig, axes=trainer._axes,
                        display_handle=trainer._display_handle,
                    )

        trainer._global_epoch += epochs
        print(f"Trainer (Lagrangian mode) done in {time.time() - start_time:.1f}s")

        # on_training_end for all other schedulers
        for s in getattr(trainer, '_schedulers', []):
            if s is not self:
                s.on_training_end(trainer)

        if _is_nb() and show_plots and trainer._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(trainer._fig)

    # ------------------------------------------------------------------
    # Internal: resolve constraints
    # ------------------------------------------------------------------

    def _resolve_constraints(self, trainer) -> 'list | None':
        """Return the final list of term names to constrain, or None (= all)."""
        c = self.constraints
        if c is None:
            return None
        if callable(c):
            # Predicate: apply to all known term names
            all_names = ['pde'] + trainer._get_bc_names()
            return [n for n in all_names if c(n)]
        return list(c)

    # ------------------------------------------------------------------
    # Internal: initialise λ vectors
    # ------------------------------------------------------------------

    def _init_lambdas(self, trainer) -> None:
        import optax as _optax
        self.lagrange_multipliers = {}
        self._lagrange_opt_states = {}

        if self.optimizer_name == 'adam':
            self._lagrange_optimizer = _optax.inject_hyperparams(
                _optax.adam)(learning_rate=self.lr)
        elif self.optimizer_name == 'sgd':
            self._lagrange_optimizer = _optax.inject_hyperparams(
                _optax.sgd)(learning_rate=self.lr)
        else:
            self._lagrange_optimizer = None

        rc = self._resolved_constraints  # None = all

        def _add(name, size):
            self.lagrange_multipliers[name] = jnp.zeros(size)
            if self._lagrange_optimizer is not None:
                self._lagrange_opt_states[name] = self._lagrange_optimizer.init(
                    self.lagrange_multipliers[name])

        # PDE collocation term
        if ('pde' in trainer._train_data
                and (rc is None or 'pde' in rc)):
            _add('pde', len(trainer._train_data['pde']))

        # ProblemWeak: one λ vector per inner term
        from pinns.problems.problem_weak import ProblemWeak as _PW
        if isinstance(trainer.problem, _PW):
            _n_free_comp = trainer.problem.n_free_nodes * trainer.problem.n_outputs
            _inner = (getattr(trainer.problem, '_inner_terms', None)
                      or [{'fn': None, 'name': 'pde'}])
            for _wt in _inner:
                wname = _wt['name']
                if rc is None or wname in rc:
                    _add(wname, _n_free_comp)

        # BC terms
        for name in trainer._get_bc_names():
            if name in trainer._train_data and (rc is None or name in rc):
                _add(name, len(trainer._train_data[name]))

        # TermPeriodicBC pairs
        from pinns.terms import TermPeriodicBC as _PBC
        _n_out = (len(trainer.problem.output_names)
                  if (hasattr(trainer.problem, 'output_names')
                      and trainer.problem.output_names)
                  else getattr(trainer.problem, 'n_outputs', 1))
        for bc in getattr(getattr(trainer.problem, 'domain', None),
                          'boundary_conditions', []):
            if not isinstance(bc, _PBC):
                continue
            comps = ([bc.component] if bc.component is not None
                     else list(range(_n_out)))
            for i in comps:
                sub = bc.name if bc.component is not None else f'{bc.name}_{i}'
                if rc is None or sub in rc:
                    n_pairs = (bc.n_pairs if hasattr(bc, 'n_pairs')
                               else len(bc.node_positions_a))
                    n_res = (n_pairs * 2
                             if getattr(bc, 'match_x_derivative', False)
                             else n_pairs)
                    _add(sub, n_res)

    # ------------------------------------------------------------------
    # Internal: dual-ascent update
    # ------------------------------------------------------------------

    def _update_lambdas(self, residuals: dict) -> None:
        rc = self._resolved_constraints
        for name, g in residuals.items():
            if name not in self.lagrange_multipliers:
                continue
            if rc is not None and name not in rc:
                continue
            n = len(g)
            if self._lagrange_optimizer is not None:
                grad = -g / n
                updates, new_state = self._lagrange_optimizer.update(
                    grad, self._lagrange_opt_states[name],
                    self.lagrange_multipliers[name])
                self.lagrange_multipliers[name] = optax.apply_updates(
                    self.lagrange_multipliers[name], updates)
                self._lagrange_opt_states[name] = new_state
            else:
                self.lagrange_multipliers[name] = (
                    self.lagrange_multipliers[name] + self.lr * g / n)
            self.lagrange_multipliers[name] = jnp.clip(
                self.lagrange_multipliers[name], -self.max_val, self.max_val)

    # ------------------------------------------------------------------
    # Internal: build the AL loss closure
    # ------------------------------------------------------------------

    def _build_al_loss_fn(self, trainer, params_dict):
        """Return ``(compute_al_loss, compute_residuals)`` closures."""
        from pinns.functional import make_derivative_fn, set_context, clear_context
        from pinns.terms import TermNeumannBC, TermRobinBC, TermPeriodicBC

        rc   = self._resolved_constraints
        sched = self  # alias for inner closures

        def _constraint_uses_quad(name):
            return trainer._constraint_uses_quadratic(name)

        # ── ProblemWeak ───────────────────────────────────────────────
        from pinns.problems.problem_weak import ProblemWeak as _PW
        if isinstance(trainer.problem, _PW):
            return self._build_al_loss_fn_weak(trainer, params_dict)

        # ── ProblemStrong ─────────────────────────────────────────────
        from pinns.problems.problem_strong import ProblemStrong as _PS
        if isinstance(trainer.problem, _PS):
            return self._build_al_loss_fn_strong(trainer, params_dict)

        # ── Generic (old Problem API) ─────────────────────────────────
        network  = trainer.network
        pde_fn   = trainer.problem.pde_fn
        pde4args = len(inspect.signature(pde_fn).parameters) >= 4

        def _model(params, x):
            return network.apply(params, x, params_dict)

        bc_names = trainer._get_bc_names()
        bc_info  = {}
        _periodic_entries = []
        _n_out = (len(trainer.problem.output_names)
                  if (hasattr(trainer.problem, 'output_names')
                      and trainer.problem.output_names)
                  else getattr(trainer.problem, 'n_outputs', 1))

        # Periodic AL entries
        import numpy as _np
        for bc in getattr(getattr(trainer.problem, 'domain', None),
                          'boundary_conditions', []):
            if not isinstance(bc, TermPeriodicBC):
                continue
            if hasattr(bc, 'node_positions_a'):
                _xa = jnp.asarray(bc.node_positions_a, dtype=jnp.float32)
                _xb = jnp.asarray(bc.node_positions_b, dtype=jnp.float32)
            else:
                _rng = _np.random.default_rng()
                _n   = bc.n_pairs or 200
                _xa  = jnp.asarray(
                    trainer.problem.domain.sample_boundary(_n, region=bc.region_a, rng=_rng),
                    dtype=jnp.float32)
                _xb  = jnp.asarray(
                    trainer.problem.domain.sample_boundary(_n, region=bc.region_b, rng=_rng),
                    dtype=jnp.float32)
            comps = ([bc.component] if bc.component is not None
                     else list(range(_n_out)))
            for i in comps:
                sub = bc.name if bc.component is not None else f'{bc.name}_{i}'
                _periodic_entries.append({
                    'name': sub, 'x_a': _xa, 'x_b': _xb,
                    'component': i, 'dim': 1,
                    'match_deriv': getattr(bc, 'match_x_derivative', False),
                })

        name_idx = 0
        for bc in getattr(trainer.problem, 'boundary_conditions', []):
            if isinstance(bc, TermPeriodicBC):
                continue
            name = bc_names[name_idx]; name_idx += 1
            is_nm = isinstance(bc, (TermNeumannBC, TermRobinBC))
            bc_info[name] = {
                'component':      bc.component,
                'is_neumann':     is_nm,
                'const_value':    bc.value if not callable(bc.value) else None,
                'normal_dim':     0,
                'normal_sign':    1,
            }
            if is_nm:
                _ni = trainer.problem.domain.get_face_normal_direction(
                    getattr(bc, 'region', '')) or (0, 1)
                bc_info[name]['normal_dim']  = _ni[0]
                bc_info[name]['normal_sign'] = _ni[1]

        def compute_residuals(params, train_data, targets_dict=None):
            td = {} if targets_dict is None else targets_dict
            res = {}
            if 'pde' in train_data:
                xp = train_data['pde']
                yp = _model(params, xp)
                df = make_derivative_fn(_model, params)
                if pde4args:
                    r = pde_fn(xp, yp, params_dict, df)
                else:
                    set_context(network.apply, params)
                    try:
                        r = pde_fn(xp, yp, params_dict)
                    finally:
                        clear_context()
                res['pde'] = (sum(ri.flatten() for ri in r)
                              if isinstance(r, (list, tuple)) else r.flatten())
            for name, info in bc_info.items():
                if name not in train_data:
                    continue
                xb = train_data[name]
                yb = _model(params, xb)
                comp   = info['component']
                target = (info['const_value'] if info['const_value'] is not None
                          else td.get(name, 0.0))
                if info['is_neumann']:
                    def _fwd(x): return _model(params, x)[:, comp]
                    tang = jnp.zeros_like(xb).at[:, info['normal_dim']].set(1.0)
                    _, du = jax.jvp(_fwd, (xb,), (tang,))
                    res[name] = (info['normal_sign'] * du - target).flatten()
                else:
                    res[name] = (yb[:, comp] - target).flatten()
            for pe in _periodic_entries:
                ya = _model(params, pe['x_a'])
                yb2 = _model(params, pe['x_b'])
                ru = ya[:, pe['component']] - yb2[:, pe['component']]
                if pe['match_deriv']:
                    d = pe['dim']
                    ta = jnp.zeros_like(pe['x_a']).at[:, d].set(1.0)
                    tb = jnp.zeros_like(pe['x_b']).at[:, d].set(1.0)
                    def _fa(x): return _model(params, x)[:, pe['component']]
                    def _fb(x): return _model(params, x)[:, pe['component']]
                    _, ua = jax.jvp(_fa, (pe['x_a'],), (ta,))
                    _, ub = jax.jvp(_fb, (pe['x_b'],), (tb,))
                    res[pe['name']] = jnp.concatenate([ru, ua - ub])
                else:
                    res[pe['name']] = ru
            return res

        def compute_al_loss(params, train_data, lagrange_dict,
                            weights_dict, targets_dict=None):
            residuals = compute_residuals(params, train_data, targets_dict)
            total = 0.0
            losses = {'bcs': []}
            for name, g in residuals.items():
                lam = lagrange_dict.get(name, jnp.zeros_like(g))
                if len(lam) != len(g):
                    lam = jnp.zeros_like(g)
                quad = _constraint_uses_quad(name)
                use_lam = rc is None or name in rc
                penalty    = weights_dict.get(name, 1.0) * jnp.mean(g ** 2) if quad else 0.0
                lagrangian = jnp.mean(jax.lax.stop_gradient(lam) * g) if use_lam else 0.0
                cl = penalty + lagrangian
                losses[name] = cl
                losses[f'{name}_penalty']       = penalty
                losses[f'{name}_lagrangian']    = lagrangian
                losses[f'{name}_residual_mean'] = jnp.mean(jnp.abs(g))
                if name != 'pde':
                    losses['bcs'].append(cl)
                total = total + cl
            return total, (losses, residuals)

        return compute_al_loss, compute_residuals

    # ------------------------------------------------------------------
    # ProblemWeak AL
    # ------------------------------------------------------------------

    def _build_al_loss_fn_weak(self, trainer, params_dict):
        from pinns.problems.problem_weak import ProblemWeak as _PW
        from pinns.terms import TermPeriodicBC

        network = trainer.network
        _n_out  = trainer.problem.n_outputs
        rc      = self._resolved_constraints

        if _n_out == 1:
            def _u_and_grad(p, xy):
                def _u(z): return network.apply(p, z[None])[0, 0]
                return jax.value_and_grad(_u)(xy)
        else:
            def _u_and_grad(p, xy):
                def _u_vec(z): return network.apply(p, z[None])[0]
                u   = _u_vec(xy)
                jac = jax.jacobian(_u_vec)(xy)
                return u, jac

        _weak_res_fn = jax.jit(trainer.problem.make_residual_vectors_fn(_u_and_grad))
        trainer._weak_residual_fn = _weak_res_fn
        trainer._u_and_grad_fn    = _u_and_grad

        _n_dofs  = trainer.problem.n_dofs
        _free    = jnp.array(trainer.problem.free_nodes, dtype=jnp.int32)
        _free_jax = (jnp.concatenate([_free + k * _n_dofs for k in range(_n_out)])
                     if _n_out > 1 else _free)
        _inner_names = set(
            t['name'] for t in (getattr(trainer.problem, '_inner_terms', []) or
                                 [{'name': 'pde'}])
        )

        bc_names    = trainer._get_bc_names()
        _bc_info    = []
        _periodic_bc_idx = 0
        for i, bc in enumerate(trainer.problem.boundary_conditions):
            if isinstance(bc, TermPeriodicBC):
                continue
            name = bc_names[_periodic_bc_idx]; _periodic_bc_idx += 1
            _bc_info.append({
                'name': name, 'component': bc.component,
                'const_value': bc.value if not callable(bc.value) else None,
            })

        def _model(p, x): return network.apply(p, x)

        def compute_residuals(params, train_data, targets_dict=None):
            td  = {} if targets_dict is None else targets_dict
            res = {}
            R_dict = _weak_res_fn(params)
            for tname, R_full in R_dict.items():
                res[tname] = R_full[_free_jax]
            for info in _bc_info:
                bname = info['name']
                if bname not in train_data:
                    continue
                xb = train_data[bname]
                yb = _model(params, xb)
                comp   = info['component']
                target = (info['const_value'] if info['const_value'] is not None
                          else td.get(bname, 0.0))
                res[bname] = (yb[:, comp] - target).flatten()
            return res

        def compute_al_loss(params, train_data, lagrange_dict,
                            weights_dict, targets_dict=None):
            residuals = compute_residuals(params, train_data, targets_dict)
            total = 0.0
            losses = {'bcs': []}
            for name, g in residuals.items():
                lam = lagrange_dict.get(name, jnp.zeros_like(g))
                if len(lam) != len(g):
                    lam = jnp.zeros_like(g)
                use_quad = trainer._constraint_uses_quadratic(name)
                use_lam  = rc is None or name in rc
                penalty    = weights_dict.get(name, 1.0) * jnp.mean(g ** 2) if use_quad else 0.0
                lagrangian = jnp.mean(jax.lax.stop_gradient(lam) * g) if use_lam else 0.0
                cl = penalty + lagrangian
                losses[name] = cl
                losses[f'{name}_penalty']      = penalty
                losses[f'{name}_lagrangian']   = lagrangian
                losses[f'{name}_residual_mean']= jnp.mean(jnp.abs(g))
                if name not in _inner_names:
                    losses['bcs'].append(cl)
                total = total + cl
            return total, (losses, residuals)

        return compute_al_loss, compute_residuals

    # ------------------------------------------------------------------
    # ProblemStrong AL
    # ------------------------------------------------------------------

    def _build_al_loss_fn_strong(self, trainer, params_dict):
        from pinns.functional import make_derivative_fn

        network = trainer.network
        _terms  = list(trainer.problem._terms)
        rc      = self._resolved_constraints

        def _model(params, x):
            return network.apply(params, x, params_dict)

        def compute_residuals(params, train_data, targets_dict=None):
            df  = make_derivative_fn(_model, params)
            res = {}
            for term in _terms:
                if term.name not in train_data:
                    continue
                x = train_data[term.name]
                u = _model(params, x)
                if term.kind == 'points':
                    col = term.output_idx if term.output_idx is not None else 0
                    tgt = jnp.array(term.u_data, dtype=jnp.float32).flatten()
                    r   = (u[:, col] - tgt).flatten()
                elif term.fn is not None and callable(term.fn):
                    r = term.fn(x, u, params_dict, df)
                    if (term.eq_idx is not None
                            and hasattr(r, 'ndim') and r.ndim == 2):
                        r = r[:, term.eq_idx:term.eq_idx + 1]
                    r = r.flatten()
                elif term.fn is not None:
                    col = term.output_idx if term.output_idx is not None else 0
                    r   = (u[:, col:col + 1] - float(term.fn)).flatten()
                else:
                    continue
                res[term.name] = r
            return res

        def compute_al_loss(params, train_data, lagrange_dict,
                            weights_dict, targets_dict=None):
            residuals = compute_residuals(params, train_data, targets_dict)
            total = jnp.array(0.0)
            losses = {'bcs': []}
            for name, g in residuals.items():
                lam = lagrange_dict.get(name, jnp.zeros_like(g))
                if len(lam) != len(g):
                    lam = jnp.zeros_like(g)
                use_quad = trainer._constraint_uses_quadratic(name)
                use_lam  = rc is None or name in rc
                penalty    = weights_dict.get(name, 1.0) * jnp.mean(g ** 2) if use_quad else 0.0
                lagrangian = jnp.mean(jax.lax.stop_gradient(lam) * g) if use_lam else 0.0
                cl = penalty + lagrangian
                losses[name] = cl
                losses[f'{name}_penalty']      = penalty
                losses[f'{name}_lagrangian']   = lagrangian
                losses[f'{name}_residual_mean']= jnp.mean(jnp.abs(g))
                total = total + cl
            return total, (losses, residuals)

        return compute_al_loss, compute_residuals


__all__ = ["SchedulerLagrange"]
