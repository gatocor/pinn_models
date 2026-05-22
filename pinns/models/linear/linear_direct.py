"""LinearSolverDirect — dense direct solve via jnp.linalg.solve."""
from __future__ import annotations
from .linear_base import LinearSolverBase


class LinearSolverDirect(LinearSolverBase):
    """Direct linear solver using ``jnp.linalg.solve``.

    Builds the full matrix by probing ``matvec`` with unit vectors, then
    calls ``jnp.linalg.solve``.  Differentiable.

    **Cost**: $O(N^{2D})$ memory, $O(N^{3D})$ solve — only suitable for
    small systems (1D Chebyshev with moderate N, or 2D with small N).

    For large systems use :class:`LinearSolverGMRES` instead.

    Args:
        None — no configuration needed.

    Example::

        model = pinns.ModelSpectralSolver(
            domain, ["u"],
            linear=LinearSolverDirect(),
            shape=32, bc="chebyshev",
        )
    """

    def solve(self, matvec, b):
        import jax
        import jax.numpy as jnp

        n = b.shape[0]
        # Build matrix column by column via vmap over unit vectors
        eye = jnp.eye(n, dtype=b.dtype)
        A = jax.vmap(matvec)(eye).T   # (n, n)
        return jnp.linalg.solve(A, b)
