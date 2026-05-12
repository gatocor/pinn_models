"""Fixed time-step strategy for :class:`~pinns.models.model_stepper.ModelStepper`."""

from __future__ import annotations

import numpy as np

__all__ = ["StepperDt"]


class StepperDt:
    """Fixed time-step strategy for :class:`~pinns.models.model_stepper.ModelStepper`.

    Generates the sequence of time coordinates fed to each rollout step::

        t[i] = t0 + (i + 1) * dt    for i = 0, …, n_steps - 1

    This is the **default** stepping strategy for
    :class:`~pinns.models.model_stepper.ModelStepper`.

    Parameters
    ----------
    dt : float
        Fixed step size.  Must be positive.  Default ``1.0``.

    Examples
    --------
    Default stepper with ``dt=0.01``::

        from pinns import create_model
        from pinns.models.stepping import StepperDt

        m = create_model(domain, output_dim=1,
                         context_range=[(0.0, 1.0)],
                         stepper=StepperDt(dt=0.01))

        # Rollout for 50 steps starting at t=0:
        traj = m.rollout(params, x_spatial, initial_output, n_steps=50, t0=0.0)
        # traj.shape == (50, B, 1)
    """

    def __init__(self, dt: float = 1.0):
        if dt <= 0:
            raise ValueError("StepperDt: dt must be positive.")
        self.dt = float(dt)

    def get_times(self, t0: float, n_steps: int) -> np.ndarray:
        """Return the time values for an autoregressive rollout.

        Parameters
        ----------
        t0 : float
            Start time (time of the initial condition / context).
        n_steps : int
            Number of steps to generate.

        Returns
        -------
        np.ndarray  shape ``(n_steps,)``
            ``[t0 + dt, t0 + 2*dt, …, t0 + n_steps * dt]``
        """
        return np.arange(1, n_steps + 1, dtype=float) * self.dt + t0

    def __repr__(self) -> str:
        return f"StepperDt(dt={self.dt})"
