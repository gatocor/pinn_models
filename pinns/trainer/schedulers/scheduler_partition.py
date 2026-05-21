"""
SchedulerPartition — temporal curriculum for ``ModelPartitioned`` networks.

Architecture
------------
The gradient mask lives **inside** ``opt_state`` as a ``MaskedState(inner, mask)``
JAX pytree.  Because ``opt_state`` is already a JIT argument in ``train_step``,
updating the mask is a pure data mutation — **no JIT recompilation ever occurs**.

At compile time ``SchedulerPartition`` replaces ``trainer.optimizer`` with a
wrapped ``GradientTransformation`` that stores ``(inner_state, mask)`` together.
Leaves whose mask value is 0.0 receive a zero gradient, so their parameters and
optimizer moments are never touched.

Curriculum logic
----------------
Subnets are ordered by their lower time bound (``network._all_xmin[i, t_col]``).
Every *epochs_per_partition* epochs the active window slides forward by one.
The ``window`` parameter controls how many consecutive subnets are simultaneously
active:

    window=1 (default) — focus on one partition at a time:
        stage 0  →  {sub_0}
        stage 1  →  {sub_1}
        stage 2  →  {sub_2}
        …

    window=2 — sliding window of 2:
        stage 0  →  {sub_0}
        stage 1  →  {sub_0, sub_1}
        stage 2  →  {sub_1, sub_2}
        stage 3  →  {sub_2, sub_3}
        …

    window=3 — sliding window of 3:
        stage 0  →  {sub_0}
        stage 1  →  {sub_0, sub_1}
        stage 2  →  {sub_0, sub_1, sub_2}
        stage 3  →  {sub_1, sub_2, sub_3}
        …

    window=None — cumulative (all subnets up to current stage stay active):
        stage 0  →  {sub_0}
        stage 1  →  {sub_0, sub_1}
        stage 2  →  {sub_0, sub_1, sub_2}
        …

When ``group_by_time=True`` subnets that share the same lower time bound are
treated as a single stage, so all spatial subnets of a time row activate
together:

    group_by_time=True, window=None — cumulative, one row per stage:
        stage 0  →  {sub_0, sub_1, sub_2, sub_3}          (t_row 0)
        stage 1  →  {sub_0..7}                             (+ t_row 1)
        stage 2  →  {sub_0..11}                            (+ t_row 2)
        …

When ``reset_optimizer=True`` the optimizer moments of newly activated subnets
are zeroed at the moment they are unmasked, giving them a fresh gradient history.

Usage
-----

    sched = SchedulerPartition(
        epochs_per_partition=10_000,
        t_col=-1,               # time is the last input dimension
        window=1,               # only the current subnet is active (default)
        reset_optimizer=False,  # keep moments across stages (default)
    )

    trainer.compile(
        problem={...},
        optimizer=pinns.AdamOptimizer(1e-3),
        schedulers=[
            pinns.SchedulerResample(1),
            pinns.SchedulerWarmupDecay(warmup_steps=1000),
            pinns.SchedulerCausal(term="pde", t_col=2),
            sched,
        ],
        epochs=500_000,
    )

Public helpers (call from a notebook cell)
------------------------------------------

    sched.active_subnets(trainer)        # → {'sub_0', 'sub_1'}
    sched.freeze(trainer, ['sub_2'])     # manually freeze
    sched.unfreeze(trainer, ['sub_2'])   # manually unfreeze
"""

from __future__ import annotations

from typing import List, Optional, Set

import jax
import jax.numpy as jnp
import optax

from .scheduler_base import Scheduler


# ---------------------------------------------------------------------------
# MaskedState — JAX pytree that pairs (inner_opt_state, mask)
# ---------------------------------------------------------------------------

class MaskedState:
    """Optimizer state container that holds ``(inner, mask)`` together.

    Registered as a JAX pytree so JIT treats it transparently.  Proxies
    ``.hyperparams`` and ``._replace(hyperparams=...)`` so that the trainer's
    ``set_learning_rate`` helper works without any modifications.
    """

    __slots__ = ("inner", "mask")

    def __init__(self, inner, mask):
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "mask", mask)

    # Proxy for trainer.set_learning_rate compatibility
    @property
    def hyperparams(self):
        hp = getattr(self.inner, "hyperparams", None)
        return hp if hp is not None else {}

    def _replace(self, **kwargs):
        if "hyperparams" in kwargs:
            new_inner = self.inner._replace(hyperparams=kwargs["hyperparams"])
            return MaskedState(inner=new_inner, mask=self.mask)
        raise ValueError(f"MaskedState._replace: unsupported keys {list(kwargs)!r}")

    def __repr__(self):
        return f"MaskedState(inner={self.inner!r}, mask=<pytree>)"


jax.tree_util.register_pytree_node(
    MaskedState,
    lambda s: ((s.inner, s.mask), None),
    lambda _, children: MaskedState(inner=children[0], mask=children[1]),
)


# ---------------------------------------------------------------------------
# Masked gradient optimizer
# ---------------------------------------------------------------------------

def make_masked_optimizer(inner: optax.GradientTransformation) -> optax.GradientTransformation:
    """Wrap *inner* so that leaves with mask == 0.0 get zero gradient updates.

    The mask is stored in the optimizer state (``MaskedState``), not hard-coded
    at construction time, so it can be changed between training steps without
    any JIT recompilation.
    """

    def init(params):
        inner_state = inner.init(params)
        mask = jax.tree_util.tree_map(lambda _: jnp.ones((), dtype=jnp.float32), params)
        return MaskedState(inner=inner_state, mask=mask)

    def update(grads, state: MaskedState, params=None):
        masked_grads = jax.tree_util.tree_map(
            lambda g, m: g * m.astype(g.dtype),
            grads,
            state.mask,
        )
        updates, new_inner = inner.update(masked_grads, state.inner, params)
        return updates, MaskedState(inner=new_inner, mask=state.mask)

    return optax.GradientTransformation(init=init, update=update)


# ---------------------------------------------------------------------------
# Pytree mask helpers
# ---------------------------------------------------------------------------

def _make_mask(params, active_subnets: Set[str]):
    """Build a float32 pytree mask: 1.0 for active leaves, 0.0 for frozen ones."""

    def _leaf(path, _leaf):
        if path:
            key = path[0]
            subnet_key = key.key if hasattr(key, "key") else str(key)
            return jnp.array(float(subnet_key in active_subnets), dtype=jnp.float32)
        return jnp.ones((), dtype=jnp.float32)                 # root-level leaf

    return jax.tree_util.tree_map_with_path(_leaf, params)


def _active_from_mask(params, mask) -> Set[str]:
    """Recover the set of active subnet keys from a mask pytree."""
    active: Set[str] = set()
    for (path, _), m in zip(
        jax.tree_util.tree_leaves_with_path(params),
        jax.tree_util.tree_leaves(mask),
    ):
        if path and float(m) > 0.5:
            key = path[0]
            active.add(key.key if hasattr(key, "key") else str(key))
    return active


def _zero_subnet_moments(opt_state: MaskedState, params, subnet_keys: Set[str]) -> MaskedState:
    """Zero the inner optimizer moments for *subnet_keys* leaves only.

    Works generically by zeroing any leaf whose top-level pytree key is in
    *subnet_keys*.  Leaves for other subnets are left untouched.
    """
    def _maybe_zero(path, leaf):
        if not path:
            return leaf
        key = path[0]
        skey = key.key if hasattr(key, "key") else str(key)
        return jnp.zeros_like(leaf) if skey in subnet_keys else leaf

    new_inner = jax.tree_util.tree_map_with_path(_maybe_zero, opt_state.inner)
    return MaskedState(inner=new_inner, mask=opt_state.mask)


# ---------------------------------------------------------------------------
# SchedulerPartition
# ---------------------------------------------------------------------------

class SchedulerPartition(Scheduler):
    """Temporal curriculum scheduler for ``ModelPartitioned`` networks.

    Advances a sliding window of active subnets every *epochs_per_partition*
    epochs.

    Parameters
    ----------
    epochs_per_partition : int or None
        Number of training epochs per stage.  ``None`` / ``0`` disables
        automatic advancement (manual control only).
    t_col : int
        Column index in ``network._all_xmin`` / ``_all_xmax`` for the time
        dimension.  ``-1`` (default) means the last column.
    window : int or None
        Number of consecutive *stages* that are simultaneously active.
        ``1`` (default) — only the current stage trains at a time.
        ``2`` — current and one predecessor.  ``None`` — cumulative
        (all stages up to the current one stay active).
    reset_optimizer : bool
        When ``True`` (default), zero the optimizer moments of newly activated
        subnets at the moment they are unmasked.
    group_by_time : bool
        When ``True`` (default), subnets that share the same lower
        time bound are grouped into a single stage and activated together.
    sync_sampling : bool
        When ``True`` (default), automatically advance ``trainer.t_max`` to
        the upper time bound of the currently active stages and resample
        collocation points.  This makes ``SchedulerCurriculum`` redundant
        when used alongside ``SchedulerPartition``.
        This means one call to ``epochs_per_partition`` advances the full
        spatial row of subnets for that time slice.  When ``False`` (original
        behaviour), every individual subnet is its own stage.
    """

    def __init__(
        self,
        epochs_per_partition: Optional[int] = 10_000,
        t_col: int = -1,
        window: Optional[int] = 1,
        reset_optimizer: bool = True,
        group_by_time: bool = True,
        sync_sampling: bool = True,
    ):
        if epochs_per_partition is not None and int(epochs_per_partition) < 0:
            raise ValueError("epochs_per_partition must be >= 0")
        if window is not None and int(window) < 1:
            raise ValueError("window must be >= 1 or None")
        self.epochs_per_partition = int(epochs_per_partition) if epochs_per_partition else None
        self.t_col = int(t_col)
        self.window = int(window) if window is not None else None
        self.reset_optimizer = bool(reset_optimizer)
        self.group_by_time = bool(group_by_time)
        self.sync_sampling = bool(sync_sampling)

        # Built at compile time from ModelPartitioned metadata.
        # When group_by_time=True this is a list-of-lists (one inner list per
        # time row); otherwise a flat list of individual subnet keys.
        self._sorted_subnet_keys: List = []  # List[str] or List[List[str]]
        # Outer (with overlap) sampling bounds per stage.
        # t_min: outer lower bound of the first active stage in the window.
        # t_max: outer upper bound of the last active stage in the window.
        self._t_min_per_stage: List[float] = []
        self._t_max_per_stage: List[float] = []
        self._original_t_min: Optional[float] = None
        self._original_t_max: Optional[float] = None
        self._current_stage: int = -1        # last activated stage index

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_compile(self, trainer) -> None:
        """Wrap the optimizer; activate stage 0 (first time partition only)."""
        if trainer.optimizer_name == "lbfgs":
            return  # L-BFGS is managed by jaxopt — masking not applicable

        trainer.optimizer = make_masked_optimizer(trainer.optimizer)
        trainer.opt_state = trainer.optimizer.init(trainer.model.params)

        self._sorted_subnet_keys = self._get_sorted_subnet_keys(trainer)
        self._t_min_per_stage, self._t_max_per_stage = self._get_sampling_bounds_per_stage(trainer)
        self._original_t_min = trainer.t_min
        self._original_t_max = trainer.t_max
        self._current_stage = -1  # force initial activation in _apply_stage

        # Activate the first stage immediately
        self._apply_stage(trainer, stage=0)

    def on_epoch_start(self, trainer, epoch: int) -> None:
        """Unlock the next time partition when enough epochs have elapsed."""
        if not self._sorted_subnet_keys:
            return
        if not isinstance(trainer.opt_state, MaskedState):
            return

        if self.epochs_per_partition is None or self.epochs_per_partition == 0:
            return  # manual-only mode

        global_epoch = trainer.get_global_epoch() + epoch
        stage = min(
            global_epoch // self.epochs_per_partition,
            len(self._sorted_subnet_keys) - 1,
        )
        if stage != self._current_stage:
            self._apply_stage(trainer, stage)

    def needs_epoch_end_at(self, epoch: int) -> bool:
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def active_subnets(self, trainer) -> Set[str]:
        """Return the set of currently active (trainable) subnet keys."""
        if not isinstance(trainer.opt_state, MaskedState):
            return set()
        return _active_from_mask(trainer.model.params, trainer.opt_state.mask)

    def freeze(self, trainer, subnet_names) -> None:
        """Manually freeze *subnet_names*.

        Parameters
        ----------
        subnet_names : sequence of str
            Subnet keys to freeze, e.g. ``["sub_2", "sub_3"]``.
        """
        if not isinstance(trainer.opt_state, MaskedState):
            raise RuntimeError(
                "SchedulerPartition has not been compiled yet. "
                "Add it to the schedulers list in trainer.compile()."
            )
        active = _active_from_mask(trainer.model.params, trainer.opt_state.mask)
        active -= set(subnet_names)
        new_mask = _make_mask(trainer.model.params, active)
        trainer.opt_state = MaskedState(inner=trainer.opt_state.inner, mask=new_mask)

    def unfreeze(self, trainer, subnet_names) -> None:
        """Manually unfreeze *subnet_names*.

        Parameters
        ----------
        subnet_names : sequence of str
            Subnet keys to unfreeze, e.g. ``["sub_2"]``.
        """
        if not isinstance(trainer.opt_state, MaskedState):
            raise RuntimeError(
                "SchedulerPartition has not been compiled yet. "
                "Add it to the schedulers list in trainer.compile()."
            )
        active = _active_from_mask(trainer.model.params, trainer.opt_state.mask)
        active |= set(subnet_names)
        new_mask = _make_mask(trainer.model.params, active)
        trainer.opt_state = MaskedState(inner=trainer.opt_state.inner, mask=new_mask)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_stage(self, trainer, stage: int) -> None:
        """Activate the window of stages centred on *stage*."""
        keys = self._sorted_subnet_keys   # List[str] or List[List[str]]
        n = len(keys)
        if self.window is None:
            lo = 0
        else:
            lo = max(0, stage - self.window + 1)

        # Flatten the selected stages into a set of subnet key strings.
        def _flatten(entries):
            out: Set[str] = set()
            for e in entries:
                if isinstance(e, list):
                    out.update(e)
                else:
                    out.add(e)
            return out

        active = _flatten(keys[lo : stage + 1])
        prev_lo = max(0, self._current_stage - (self.window or n) + 1) if self._current_stage >= 0 else 0
        prev_active = _flatten(keys[prev_lo : self._current_stage + 1]) if self._current_stage >= 0 else set()
        newly_active = active - prev_active
        new_mask = _make_mask(trainer.model.params, active)

        new_state = MaskedState(inner=trainer.opt_state.inner, mask=new_mask)
        if self.reset_optimizer and newly_active:
            new_state = _zero_subnet_moments(new_state, trainer.model.params, newly_active)

        trainer.opt_state = new_state
        self._current_stage = stage

        # Advance t_max to the hard breakpoint of the current active stage
        # (no future overlap — untrained subnets are noisy).
        # t_min extends to the outer edge including past overlap so points in
        # the transition region of the first active stage are sampled.
        if self.sync_sampling and self._t_max_per_stage:
            s = min(stage, len(self._t_max_per_stage) - 1)
            trainer.t_max = self._t_max_per_stage[s]
            if self._t_min_per_stage:
                trainer.t_min = self._t_min_per_stage[s]
            trainer._sample_train_data()
            if trainer._test_data:
                trainer._sample_test_data()

    def on_training_end(self, trainer) -> None:
        """Restore the original t_min/t_max."""
        if self._original_t_max is not None:
            trainer.t_max = self._original_t_max
        if self._original_t_min is not None:
            trainer.t_min = self._original_t_min

    def _get_sorted_subnet_keys(self, trainer) -> List:
        """Return subnet keys sorted by their lower time bound.

        When ``group_by_time=True`` returns a list of lists — each inner list
        contains all subnet keys that share the same ``t_min`` value.
        When ``group_by_time=False`` (default) returns a plain flat list.
        """
        if not self._is_partitioned(trainer):
            return []
        net = trainer.model
        t_col = self.t_col
        order = sorted(
            range(net.n_models),
            key=lambda i: float(net._all_xmin[i, t_col]),
        )
        if not self.group_by_time:
            return [f"sub_{i}" for i in order]

        # Group consecutive subnets by identical t_min value.
        from itertools import groupby
        groups = []
        for _t, grp in groupby(order, key=lambda i: round(float(net._all_xmin[i, t_col]), 10)):
            groups.append([f"sub_{i}" for i in grp])
        return groups

    def _get_sampling_bounds_per_stage(self, trainer):
        """Return (t_min_list, t_max_list) per stage.

        - ``t_min`` = outer lower bound of the first active stage in the
          window (``_all_outer_xmin = _all_xmin - wmin/2``), clamped to the
          domain.  Extending into the past overlap ensures points in the
          sigmoid transition region of the first active stage are sampled.
        - ``t_max`` = hard center breakpoint of the last active stage
          (``_all_xmax``), clamped to the domain.  Future overlap is
          deliberately excluded because those future subnets are still
          frozen/untrained and would add noise.
        """
        if not self._is_partitioned(trainer) or not self.sync_sampling:
            return [], []
        net = trainer.model
        t_col = self.t_col
        keys = self._sorted_subnet_keys
        n = len(keys)

        def _indices(group):
            g = group if isinstance(group, list) else [group]
            return [int(k.split('_')[1]) for k in g]

        def _outer_tmin(group):
            return min(float(net._all_outer_xmin[i, t_col]) for i in _indices(group))

        def _center_tmax(group):
            """Hard breakpoint — no future overlap."""
            return max(float(net._all_xmax[i, t_col]) for i in _indices(group))

        # Clamp to the global domain bounds so sampling never requests an
        # interval outside [trainer.t_min, original_t_max].
        domain_lo = trainer.t_min if trainer.t_min is not None else -float('inf')
        domain_hi = trainer.t_max if trainer.t_max is not None else float('inf')

        t_min_list, t_max_list = [], []
        for s in range(n):
            lo = 0 if self.window is None else max(0, s - self.window + 1)
            t_min_list.append(max(_outer_tmin(keys[lo]), domain_lo))
            t_max_list.append(min(_center_tmax(keys[s]), domain_hi))
        return t_min_list, t_max_list

    @staticmethod
    def _is_partitioned(trainer) -> bool:
        """Duck-type check: does the network expose the partitioned-model API?"""
        net = trainer.model
        return (
            hasattr(net, "n_models")
            and hasattr(net, "_all_xmin")
            and hasattr(net, "_all_xmax")
            and hasattr(net, "params")
        )


__all__ = ["SchedulerPartition", "MaskedState", "make_masked_optimizer"]
