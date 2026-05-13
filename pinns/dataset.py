"""
Dataset — fixed observation data for Physics-Informed Neural Networks.

Unlike interior or boundary terms (whose collocation points are *sampled*
from the domain each epoch), a :class:`Dataset` holds **fixed** ``(x, u)``
pairs — sensor readings, synthetic measurements, or any supervised data.

The full dataset is evaluated every training step without resampling.  Pass
a :class:`Dataset` instance as the optional ``dataset`` argument of
:class:`~pinns.trainer.Trainer` alongside the problem and network.

Example::

    dataset = Dataset()
    dataset.add_points(x_obs, u_obs, name='sensors', component=0)
    dataset.add_points(x_wall, u_wall, name='wall_data', component=0)

    trainer = Trainer(problem, network, dataset=dataset)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# TermPoints — single observation set (data container)
# ---------------------------------------------------------------------------

@dataclass
class TermPoints:
    """Data-assimilation term at arbitrary fixed coordinates.

    Points may lie anywhere — on the boundary, in the interior, or scattered
    measurement data.  The trainer always uses **all** points in a single
    batch (no domain-based sampling).

    When multiple output components are supervised, provide an ``(N, K)``
    target array and a matching ``components`` list.  Each column produces
    one independent sub-loss.

    Args:
        inputs: ``(N, n_dims)`` observation coordinates.
        outputs: ``(N, K)`` target values.  A 1-D array of length ``N`` is
                 treated as ``(N, 1)``.
        components: Network output index for each column of *outputs*.
        name: Base label used in weight dicts and plots.
        output_names: Optional per-column labels.

    Example::

        obs = TermPoints(inputs=x_meas, outputs=u_meas, components=0, name='obs')
    """

    inputs:       np.ndarray
    outputs:      np.ndarray
    components:   Union[int, List[int]] = 0
    name:         Optional[str]         = None
    output_names: Optional[List[str]]   = None

    kind:      ClassVar[str]  = 'points'
    has_value: ClassVar[bool] = False

    def __post_init__(self):
        self.inputs  = np.asarray(self.inputs,  dtype=np.float32)
        self.outputs = np.asarray(self.outputs, dtype=np.float32)
        if self.inputs.ndim  == 1:
            self.inputs  = self.inputs[:, None]
        if self.outputs.ndim == 1:
            self.outputs = self.outputs[:, None]
        if isinstance(self.components, int):
            self.components = [self.components]
        n_cols = self.outputs.shape[1]
        if n_cols != len(self.components):
            raise ValueError(
                f"TermPoints '{self.name}': outputs has {n_cols} column(s) "
                f"but components has {len(self.components)} element(s)."
            )

    def get_input_names(self) -> List[str]:
        base = self.name or 'pts'
        if self.output_names is not None:
            return list(self.output_names)
        return [base] if len(self.components) == 1 else \
               [f'{base}_{k}' for k in range(len(self.components))]


# ---------------------------------------------------------------------------
# Dataset — registry of one or more TermPoints
# ---------------------------------------------------------------------------

class Dataset:
    """Registry of fixed observation / measurement datasets.

    Each call to :meth:`add_points` registers one labelled point set.
    During training the trainer evaluates all registered sets every step
    (no resampling), computing a supervised MSE loss for each.

    Args:
        None — construct empty, then call :meth:`add_points` one or more
        times.

    Example::

        from pinns import Dataset

        ds = Dataset()
        ds.add_points(x_sensors, u_sensors, name='sensors', component=0)
        ds.add_points(x_wall,    u_wall,    name='wall',    component=0)

        trainer = Trainer(problem, network, dataset=ds)
        trainer.compile(weights={'sensors': 2.0, 'wall': 1.0})
        trainer.train()
    """

    def __init__(self):
        self._data_terms: List[TermPoints] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_points(
        self,
        x,
        u,
        name: str,
        component: int = 0,
        output_names: Optional[List[str]] = None,
    ) -> 'Dataset':
        """Register a labelled set of observation points.

        Args:
            x: Array-like of shape ``(N, n_dims)`` — observation coordinates.
            u: Array-like of shape ``(N,)`` or ``(N, K)`` — target values.
            name: Unique label used in the loss dict and weight dict.
            component: Network output index this data supervises.  For
                multi-output supervision call :meth:`add_points` once per
                output, or pass a 2-D *u* together with a list of
                *component* values.
            output_names: Optional list of per-column loss keys (length
                must match number of columns in *u*).

        Returns:
            ``self`` for method chaining.

        Raises:
            ValueError: If *x* and *u* have incompatible leading dimensions.
        """
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        u_arr = np.asarray(u, dtype=np.float32)
        if u_arr.ndim == 1:
            u_arr = u_arr[:, None]
        if x_arr.shape[0] != u_arr.shape[0]:
            raise ValueError(
                f"Dataset.add_points('{name}'): x has {x_arr.shape[0]} row(s) "
                f"but u has {u_arr.shape[0]} row(s)."
            )
        n_cols = u_arr.shape[1]
        components: List[int] = (
            [component] * n_cols if isinstance(component, int) else list(component)
        )
        if len(components) != n_cols:
            raise ValueError(
                f"Dataset.add_points('{name}'): u has {n_cols} column(s) "
                f"but component list has {len(components)} element(s)."
            )
        term = TermPoints(
            inputs=x_arr,
            outputs=u_arr,
            components=components,
            name=name,
            output_names=output_names,
        )
        self._data_terms.append(term)
        return self

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def data_terms(self) -> List[TermPoints]:
        """All registered :class:`TermPoints` in registration order."""
        return list(self._data_terms)

    @property
    def names(self) -> List[str]:
        """Names of all registered point sets."""
        return [t.name for t in self._data_terms]

    def __len__(self) -> int:
        return len(self._data_terms)

    def __repr__(self) -> str:
        parts = [
            f"  '{t.name}': {t.inputs.shape[0]} pts, comp={t.components}"
            for t in self._data_terms
        ]
        body = ("\n" + "\n".join(parts) + "\n") if parts else ""
        return f"Dataset({len(self._data_terms)} term(s)){{{body}}}"
