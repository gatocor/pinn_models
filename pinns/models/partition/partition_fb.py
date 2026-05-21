"""Finite-Basis PINN (FB-PINN) spatial strategy."""

from __future__ import annotations

from typing import Optional
import numpy as np
import jax.nn
import jax.numpy as jnp

__all__ = ["PartitionFB"]


class PartitionFB:
    """Finite-Basis PINN (FB-PINN) strategy.

    Each network covers one subdomain and its output is weighted by a smooth
    **window function**.  The subdomain decomposition is defined by the domain
    partition (``domain.set_partition(...)``); ``PartitionFB`` only controls
    *how wide* the windows are relative to each subdomain's width.

    The global solution is assembled as a **dynamic partition of unity**:

        u(x) = Σᵢ wᵢ(x)·uᵢ(x) / Σᵢ wᵢ(x)

    which is exact regardless of window shape, matching the original FBPINNs
    implementation.

    Parameters
    ----------
    overlap : float
        Window width as a multiple of the subdomain spacing.
        For ``window='hann'`` (default): each bell spans
        ``overlap × spacing`` centred on the subdomain midpoint.
        The original FBPINNs paper uses ``subdomain_ws = 0.15`` with
        ``spacing = 1/14``, giving ``overlap ≈ 2.1``.
        For ``window='cosine'``: the flat-top region equals the home interval
        and each transition extends ``overlap × home_width`` on each side.
        For ``window='sigmoid'``: same as cosine but with a logistic ramp.
        Must be > 0.
    window : str
        Window function:

        * ``'hann'`` (default) — pure cosine bell ``((1+cos)/2)²``,
          matching the original FBPINNs paper.  Dynamic POU normalization
          makes the precise width less critical.
        * ``'cosine'`` — flat-top cosine with zero-derivative plateau inside
          each subdomain.  Useful when ODE residuals must be free of
          window-derivative cross-terms.
        * ``'sigmoid'`` — logistic sigmoid ramp on each edge.

    Example::

        domain = DomainCubic(time=(0, 1))
        domain.set_partition(time=15)   # 14 subdomains
        model  = create_model(domain, output_dim=1, hidden_dims=(32,))
        net_fb = ModelPartitioned(model, PartitionFB(overlap=2.1))
        # window width ≈ 2.1 × (1/14) ≈ 0.15, matching the original paper
    """

    def __init__(
        self,
        overlap: float = 2.1,
        window: str = "hann",
        xmin=None,
        xmax=None,
        wmin=None,
        wmax=None,
    ):
        if overlap <= 0.0:
            raise ValueError("PartitionFB: overlap must be > 0.")
        if window not in ("cosine", "sigmoid", "hann"):
            raise ValueError("PartitionFB: window must be 'cosine', 'sigmoid', or 'hann'.")
        self.overlap = float(overlap)
        self.window = window
        self._xmin: Optional[np.ndarray] = (
            np.asarray(xmin, dtype=np.float64) if xmin is not None else None
        )
        self._xmax: Optional[np.ndarray] = (
            np.asarray(xmax, dtype=np.float64) if xmax is not None else None
        )
        # wmin/wmax: total window width on the left/right edge.
        # Set internally by ModelPartitioned based on overlap × spacing.
        self._wmin: Optional[np.ndarray] = (
            np.asarray(wmin, dtype=np.float64) if wmin is not None else None
        )
        self._wmax: Optional[np.ndarray] = (
            np.asarray(wmax, dtype=np.float64) if wmax is not None else None
        )

    def setup(self, domain) -> None:
        """Read spatial bounds from *domain* (if not already set explicitly)."""
        if self._xmin is not None:
            return  # user provided explicit bounds
        n_s = domain._spatial_dims
        self._xmin = np.asarray(domain.xmin[:n_s], dtype=np.float64)
        self._xmax = np.asarray(domain.xmax[:n_s], dtype=np.float64)

    # ------------------------------------------------------------------
    # Window function
    # ------------------------------------------------------------------

    def _window(self, x_spatial):
        """Window function for FB-PINN.

        Two variants are supported, chosen by ``self.window``:

        **'cosine'** (default):
            Flat-top window with cosine transitions at each edge.
            Inside the home interval ``[a, b]``, ``w = 1`` and ``dw/dt = 0``
            exactly, so auto-differentiated ODE residuals contain no spurious
            window-derivative cross-terms.  Outside ``[a - ov·h, b + ov·h]``,
            ``w = 0`` exactly.  Transition is a raised-cosine ramp.

        **'hann'** (default, matches the original FBPINNs paper exactly):
            Hann bell (raised cosine squared) centered at the subdomain midpoint
            with total width ``overlap × spacing`` (set by ModelPartitioned).
            The window spans ``[center - width/2, center + width/2]``.
            Sub-networks use global domain normalization, matching
            ``unnorm=(0., 1.)`` from the original code.

        **'sigmoid'** (matches eq. 14 of the paper text):
            Logistic sigmoid window.  Has no flat top, so window derivatives
            are non-zero everywhere and can pollute high-frequency ODE
            residuals at subdomain boundaries.

        In 'cosine'/'sigmoid' cases, ``xmin``/``xmax`` store the home
        breakpoints ``a``/``b`` and ``wmin``/``wmax`` store the transition
        widths.  In 'hann' case, ``xmin = xmax = center`` and
        ``wmin = wmax = width`` (total window width).

        Shape: ``(batch, 1)``.
        """
        a = jnp.array(self._xmin, dtype=x_spatial.dtype)   # home left  breakpoints (= center for hann)
        b = jnp.array(self._xmax, dtype=x_spatial.dtype)   # home right breakpoints (= center for hann)

        if self._wmin is not None and self._wmax is not None:
            wmin = jnp.array(self._wmin, dtype=x_spatial.dtype)
            wmax = jnp.array(self._wmax, dtype=x_spatial.dtype)
        else:
            # Fallback: transition width = half the home width
            wmin = wmax = (b - a) / 2.0

        if self.window in ("cosine", "hann"):  # hann = cosine with a=b=center
            # ── Cosine flat-top window ─────────────────────────────────── #
            # Transition half-widths on left (wl) and right (wr).
            # Since wmin = 2*ov*h, the transition width on each side = wmin/2 = ov*h.
            wl = wmin / 2.0   # left transition width  (a - wl  ..  a)
            wr = wmax / 2.0   # right transition width (b  ..  b + wr)

            # Per-dimension cosine window factors:
            #   left of outer_left (a-wl):  0
            #   left ramp [a-wl, a]:        0.5*(1 - cos(π*(x-(a-wl))/wl))
            #   flat top  [a, b]:           1
            #   right ramp [b, b+wr]:       0.5*(1 + cos(π*(x-b)/wr))
            #   right of b+wr:              0
            def _cos_window_1d(x_d, a_d, b_d, wl_d, wr_d):
                left_outer  = a_d - wl_d
                right_outer = b_d + wr_d
                left_ramp  = 0.5 * (1.0 - jnp.cos(jnp.pi * (x_d - left_outer) / wl_d))
                right_ramp = 0.5 * (1.0 + jnp.cos(jnp.pi * (x_d - b_d) / wr_d))
                return jnp.where(
                    x_d < left_outer, 0.0,
                    jnp.where(x_d < a_d, left_ramp,
                    jnp.where(x_d <= b_d, 1.0,
                    jnp.where(x_d <= right_outer, right_ramp,
                    0.0))))

            # Apply per-dimension and multiply (for multi-D domains)
            w = jnp.ones((x_spatial.shape[0], 1), dtype=x_spatial.dtype)
            for d in range(x_spatial.shape[1]):
                w_d = _cos_window_1d(
                    x_spatial[:, d:d+1], a[d], b[d], wl[d], wr[d]
                )
                w = w * w_d
            # Original FBPINNs squares the Hann bell: w = ((1+cos)/2)^2
            if self.window == 'hann':
                w = w ** 2

        else:
            # ── Sigmoid window (eq. 14 of the paper text) ─────────────── #
            tol = 1e-8
            t = jnp.log((1.0 - tol) / tol)  # ≈ 18.4
            sd_min = wmin / (2.0 * t)
            sd_max = wmax / (2.0 * t)
            ws = (jax.nn.sigmoid((x_spatial - a) / sd_min)
                  * jax.nn.sigmoid((b - x_spatial) / sd_max))    # (batch, n_s)
            w = jnp.prod(ws, axis=-1, keepdims=True)              # (batch, 1)

        return w

    def predict(self, apply_fn, params, x, params_dict=None):
        """Forward pass: sequential apply weighted by the subdomain window.

        Parameters
        ----------
        apply_fn :
            The network's sequential forward pass
            (``params, x, params_dict → jnp.ndarray``).
        params : dict
            ModelBase parameter dict.
        x : jnp.ndarray  shape ``(batch, n_dims)``
            Collocation points (spatial + optional time).
        params_dict : dict or None
            Optional auxiliary dict forwarded to every layer.

        Returns
        -------
        jnp.ndarray  shape ``(batch, output_dim)``
            ModelBase output multiplied element-wise by the window function
            evaluated on the spatial coordinates.
        """
        if self._xmin is None:
            raise RuntimeError(
                "PartitionFB.predict() called before setup(). "
                "Either call network._setup(domain) or use problem.set_model(network)."
            )
        n_s = self._xmin.shape[0]
        x_spatial = x[:, :n_s]
        y = apply_fn(params, x, params_dict)
        w = self._window(x_spatial)
        return w * y

    def __repr__(self) -> str:
        parts = [f"window={self.window!r}", f"overlap={self.overlap}"]
        if self._xmin is not None:
            parts.append(f"center={self._xmin.tolist()}" if self.window == "hann"
                         else f"a={self._xmin.tolist()}, b={self._xmax.tolist()}")
        if self._wmin is not None:
            parts.append(f"wmin={self._wmin.tolist()}, wmax={self._wmax.tolist()}")
        return f"PartitionFB({', '.join(parts)})"
