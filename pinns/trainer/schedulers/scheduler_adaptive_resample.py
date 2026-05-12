"""
SchedulerAdaptiveResample – residual-driven adaptive collocation resampling.

Three modes
-----------
``'replace'``
    Replace the ``ratio`` fraction of lowest-residual PDE points with new
    points sampled near the highest-residual locations.

``'add'``
    Like ``'replace'`` but *append* rather than replace (PDE dataset grows
    until ``max_samples`` is reached).

``'rar'``
    Residual-based Adaptive Resampling (RAR) via importance sampling.
    Sample ``factor × n`` candidates, weight by ``r^k`` and draw ``n``
    without replacement.
"""

import numpy as np
from .scheduler_base import Scheduler


class SchedulerAdaptiveResample(Scheduler):
    """
    Adaptive collocation resampling driven by PDE residuals.

    Parameters
    ----------
    mode : str
        One of ``'replace'``, ``'add'`` or ``'rar'``.
    every_n : int
        Perform resampling every this many epochs.
    ratio : float
        Fraction of points to add / replace per resampling step (modes
        ``'replace'`` and ``'add'``).
    std : float
        Stddev of the Gaussian kernel used to perturb high-residual points,
        expressed as a fraction of the domain size per dimension.
    max_samples : int or None
        Upper bound on the PDE dataset size (mode ``'add'`` only).
    k : float
        Residual power exponent for RAR importance weights.
    c : float
        Uniform-sampling offset for RAR mode (higher → more uniform).
    factor : int
        Oversampling factor for RAR: sample ``factor × n`` candidates.
    """

    def __init__(
        self,
        mode: str = 'replace',
        every_n: int = 100,
        ratio: float = 0.5,
        std: float = 0.1,
        max_samples: int = None,
        k: float = 1.0,
        c: float = 1.0,
        factor: int = 2,
    ):
        if mode not in ('replace', 'add', 'rar'):
            raise ValueError(f"mode must be 'replace', 'add' or 'rar', got {mode!r}")
        self.mode = mode
        self.every_n = every_n
        self.ratio = ratio
        self.std = std
        self.max_samples = max_samples
        self.k = k
        self.c = c
        self.factor = factor

    # ------------------------------------------------------------------
    def on_epoch_start(self, trainer, epoch: int) -> None:
        if epoch <= 0 or epoch % self.every_n != 0:
            return
        if self.mode == 'rar':
            self._rar_resample(trainer)
        else:
            self._replace_or_add_resample(trainer)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _replace_or_add_resample(self, trainer) -> None:
        if 'pde' not in trainer._train_data:
            return

        x_pde = trainer._to_numpy(trainer._train_data['pde'])
        n_points = len(x_pde)

        if self.mode == 'add' and self.max_samples is not None:
            if n_points >= self.max_samples:
                return

        n_new = int(n_points * self.ratio)
        if self.mode == 'add' and self.max_samples is not None:
            n_new = min(n_new, self.max_samples - n_points)
        if n_new < 1:
            return

        residuals = trainer._compute_residuals(x_pde, batch_size=trainer._batch_size)
        total_res = np.zeros(n_points)
        for res in residuals:
            total_res += res.flatten() ** 2
        total_res = np.sqrt(total_res)

        high_idx = np.argsort(total_res)[-n_new:]
        high_pts = x_pde[high_idx]

        domain = trainer.problem.domain
        scale = np.array([domain.xmax[d] - domain.xmin[d]
                          for d in range(len(domain.xmin))])
        new_pts = high_pts + trainer.rng.normal(0, self.std * scale,
                                                 size=high_pts.shape)
        for d in range(new_pts.shape[1]):
            new_pts[:, d] = np.clip(new_pts[:, d], domain.xmin[d], domain.xmax[d])

        if self.mode == 'add':
            x_pde = np.concatenate([x_pde, new_pts], axis=0)
        else:
            low_idx = np.argsort(total_res)[:n_new]
            x_pde[low_idx] = new_pts

        trainer._train_data['pde'] = trainer._to_tensor(x_pde)

    def _rar_resample(self, trainer) -> None:
        ts = trainer.train_samples
        n_target = (ts['pde'] if isinstance(ts, dict) else ts[0])
        n_cands = self.factor * n_target

        params_dict = trainer._build_params()
        x_cands = trainer.problem.domain.sample_interior(
            n_cands, rng=trainer.rng, params=params_dict)

        residuals = trainer._compute_residuals(x_cands, batch_size=trainer._batch_size)
        total_res = np.zeros(n_cands)
        for res in residuals:
            total_res += res.flatten() ** 2
        total_res = np.sqrt(total_res)

        r_pow_k = np.power(np.abs(total_res) + 1e-10, self.k)
        weights = (r_pow_k / (np.mean(r_pow_k) + 1e-10)) + self.c
        weights /= weights.sum()

        sel = trainer.rng.choice(n_cands, size=n_target, replace=False, p=weights)
        trainer._train_data['pde'] = trainer._to_tensor(x_cands[sel])


__all__ = ["SchedulerAdaptiveResample"]
