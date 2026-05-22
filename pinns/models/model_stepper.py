"""
Autoregressive time-stepper wrapping a :class:`~pinns.models.model_base.ModelBase`
or :class:`~pinns.models.model_partitioned.ModelPartitioned`.

The wrapped model treats the **previous step's outputs** as trailing *context*
columns in its input tensor.  At each step the stepper concatenates the spatial
(and optional time) coordinates with the output from the last step, feeds the
result through the underlying model, and returns the new state.

Quick start::

    from pinns import Model, ModelStepper
    import jax, jax.numpy as jnp, numpy as np

    domain = DomainCubic(space=[0.0, 1.0], time=np.linspace(0, 1, 11))

    # Build a time-dependent model with one output.
    model = Model(domain, output_dim=1, n_context=1,
                  context_range=[(0.0, 1.0)])
    model.add(Normalize())
    model.add(FNN([64, 64]))
    model.add(Denormalize())

    stepper = ModelStepper(model)
    params  = stepper.init(jax.random.PRNGKey(0))

    x_spatial  = jnp.ones((100, 1))          # spatial coords, shape (B, d)
    t_values   = jnp.linspace(0, 1, 11)[1:]  # 10 time values  (n_steps,)
    u0         = jnp.zeros((100, 1))          # initial condition

    trajectory = stepper.rollout(params, x_spatial, t_values, u0)
    # trajectory: shape (n_steps, B, output_dim)

Alternatively, when the model has no time axis, call
:meth:`~ModelStepper.rollout` without ``t_values`` (or ``t_values=None``).
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from .model_base import ModelBase
from .model_partitioned import ModelPartitioned
from .stepping import StepperDt


class ModelStepper:
    """
    Autoregressive time-stepper built on top of a ``ModelBase`` / ``ModelPartitioned``.

    The wrapped model is used **as-is** — no deep copy is made.  Only the
    ``set_context`` call may mutate the wrapped model (if ``context_range`` is
    provided and the model's current ``n_context`` needs updating).

    Parameters
    ----------
    model : ModelBase | ModelPartitioned
        The network to step.  Must expose ``output_dim``, ``n_context``,
        ``init(rng) -> dict``, and ``apply(params, x) -> jnp.ndarray``.
        The model's ``n_context`` **must equal** ``output_dim`` (i.e. the
        previous outputs fill the context slots exactly).

        Use :meth:`~pinns.models.model_base.ModelBase.set_context` on *model* **before**
        constructing the stepper if that is not yet the case, or pass
        ``context_range`` here to let the stepper call ``set_context``
        automatically.
    context_range : list of (min, max) pairs, optional
        If given, :meth:`~pinns.models.model_base.ModelBase.set_context` is called on
        the wrapped model with ``n_context = output_dim`` and these ranges.
        Useful when the model was constructed without context and you want the
        stepper to configure it.

    Raises
    ------
    TypeError
        If *model* is not a ``ModelBase`` or ``ModelPartitioned``.
    ValueError
        If *model*'s ``n_context`` does not equal its ``output_dim`` (and
        ``context_range`` was not provided to fix it automatically).

    Attributes
    ----------
    model : ModelBase | ModelPartitioned
        The wrapped model.
    output_dim : int
        Output dimension.
    n_context : int
        Number of context columns (always equals ``output_dim``).
    """

    def __init__(
        self,
        model: Union[ModelBase, ModelPartitioned],
        context_range: Optional[List[Tuple[float, float]]] = None,
        strategy: Optional[StepperDt] = None,
    ):
        # ModelStepper handles time via discrete stepping — no spatial tiling needed.
        self.n_time_collocation = None
        if not isinstance(model, (ModelBase, ModelPartitioned)):
            raise TypeError(
                f"ModelStepper: model must be a ModelBase or ModelPartitioned, "
                f"got {type(model).__name__!r}."
            )

        out_dim = model.output_dim

        if context_range is not None:
            # Configure context on the model automatically.
            if len(context_range) != out_dim:
                raise ValueError(
                    f"ModelStepper: context_range must have {out_dim} pairs "
                    f"(one per output dimension), got {len(context_range)}."
                )
            model.set_context(out_dim, context_range)
        else:
            # Validate that the model is already configured with the right n_context.
            if model.n_context != out_dim:
                raise ValueError(
                    f"ModelStepper: model.n_context ({model.n_context}) must "
                    f"equal model.output_dim ({out_dim}).  Either build the model "
                    f"with n_context=output_dim or pass context_range= here."
                )

        self.model = model
        self.output_dim = out_dim
        self.n_context = out_dim
        self.strategy: StepperDt = strategy if strategy is not None else StepperDt()

    # ── delegation ──────────────────────────────────────────────────── #

    @property
    def domain(self):
        """Domain of the wrapped model."""
        return self.model.domain

    def init(self, rng: "jax.random.PRNGKey") -> dict:
        """Initialise the wrapped model's parameters."""
        return self.model.init(rng)

    def add_constraint(
        self,
        value,
        *,
        region: str = "all",
        output_idx=None,
        sigma=None,
    ) -> "ModelStepper":
        """Delegate :meth:`~pinns.models.model_base.ModelBase.add_constraint` to the wrapped model.

        Returns
        -------
        ModelStepper
            ``self`` for method chaining.
        """
        self.model.add_constraint(value, region=region,
                                  output_idx=output_idx, sigma=sigma)
        return self

    # ── single-step interface ────────────────────────────────────────── #

    def apply(
        self,
        x: "jnp.ndarray",
        prev_output: "jnp.ndarray",
        params: dict = None,
        params_dict: Optional[dict] = None,
    ) -> "jnp.ndarray":
        """Perform a **single** time step.

        Parameters
        ----------
        x :
            Input coordinates, shape ``(B, d)`` where ``d`` is
            ``spatial_dims`` [+ 1 if time is included].  Do **not** include
            the context columns — they are appended from ``prev_output``.
        prev_output :
            Previous step outputs, shape ``(B, output_dim)``.  Concatenated as
            trailing context columns before calling the model.
        params :
            Parameter dict returned by :meth:`init`.
        params_dict :
            Optional auxiliary dict forwarded to every layer.

        Returns
        -------
        jnp.ndarray  shape ``(B, output_dim)``
        """
        x_ctx = jnp.concatenate([x, prev_output], axis=-1)
        return self.model.apply(x_ctx, params, params_dict)

    # ── autoregressive rollout ───────────────────────────────────────── #

    def rollout(
        self,
        params: dict,
        x_spatial: "jnp.ndarray",
        t_values: Optional["jnp.ndarray"],
        initial_output: "jnp.ndarray",
        params_dict: Optional[dict] = None,
        *,
        n_steps: Optional[int] = None,
        t0: float = 0.0,
    ) -> "jnp.ndarray":
        """Step through time autoregressively.

        Time values can be supplied in three ways (in order of precedence):

        1. **Explicit** ``t_values`` array — used as-is.
        2. **Strategy** — when ``t_values=None`` and ``n_steps`` is given,
           ``self.strategy.get_times(t0, n_steps)`` generates the values
           (default strategy: :class:`~pinns.models.stepping.StepperDt`).
        3. **No time axis** — pass ``t_values=None`` and omit ``n_steps``
           for purely spatial models (raises ``ValueError`` if the model
           requires a time column).

        Parameters
        ----------
        params :
            Parameter dict.
        x_spatial :
            Spatial coordinates, shape ``(B, d_s)``.  Time is **not**
            included here — pass it via ``t_values`` / ``n_steps``.
            If the wrapped model has no time axis, the full coordinate array
            can be passed here and ``t_values=None``.
        t_values :
            1-D array of time values, shape ``(n_steps,)``.
            ``None`` to use the strategy or to skip time entirely.
        initial_output :
            Initial condition / context for step 0, shape ``(B, output_dim)``.
        params_dict :
            Optional auxiliary dict forwarded to every layer at every step.
        n_steps : int or None
            Number of steps to generate via the stepping strategy when
            ``t_values`` is ``None``.  Ignored when ``t_values`` is given.
        t0 : float
            Start time passed to ``self.strategy.get_times``.
            Default ``0.0``.  Ignored when ``t_values`` is given.

        Returns
        -------
        jnp.ndarray  shape ``(n_steps, B, output_dim)``
            Stacked outputs for all steps (does **not** include the initial
            condition).
        """
        if t_values is not None:
            t_values = jnp.asarray(t_values, dtype=jnp.float32)
            _n_steps = t_values.shape[0]
        elif n_steps is not None:
            # Generate time values using the stepping strategy.
            _t_np = self.strategy.get_times(float(t0), int(n_steps))
            t_values = jnp.asarray(_t_np, dtype=jnp.float32)
            _n_steps = n_steps
        else:
            # No time axis — purely spatial rollout.
            _n_steps = None

        if _n_steps is None:
            raise ValueError(
                "ModelStepper.rollout: provide either t_values or n_steps (+ optional t0). "
                "Pass t_values=None and omit n_steps only for purely spatial models."
            )
        n_steps = _n_steps

        outputs = []
        prev = initial_output
        for i in range(n_steps):
            if t_values is not None:
                t_i = jnp.full((x_spatial.shape[0], 1), t_values[i], dtype=jnp.float32)
                x_i = jnp.concatenate([x_spatial, t_i], axis=-1)
            else:
                x_i = x_spatial
            prev = self.apply(x_i, prev, params, params_dict)
            outputs.append(prev)

        return jnp.stack(outputs, axis=0)  # (n_steps, B, output_dim)

    # ── repr ─────────────────────────────────────────────────────────── #

    def __repr__(self) -> str:
        model_name = type(self.model).__name__
        return (
            f"ModelStepper(output_dim={self.output_dim}, "
            f"n_context={self.n_context}, "
            f"strategy={self.strategy!r}, "
            f"model={model_name})"
        )


__all__ = ["ModelStepper"]
