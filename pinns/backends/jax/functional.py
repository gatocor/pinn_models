"""
JAX implementation of differential operators for PINNS.

Provides derivative, gradient, laplacian, and divergence functions
using JAX's automatic differentiation.

API is compatible with PyTorch backend - same function signatures.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Tuple, Dict, Any, Optional
import threading

# Thread-local storage for current computation context
_context = threading.local()


def set_context(apply_fn: Callable, params: Dict):
    """Set the current computation context for derivative calculations."""
    _context.apply_fn = apply_fn
    _context.params = params


def get_context():
    """Get current computation context."""
    if not hasattr(_context, 'apply_fn'):
        return None, None
    return _context.apply_fn, _context.params


def clear_context():
    """Clear computation context."""
    if hasattr(_context, 'apply_fn'):
        del _context.apply_fn
    if hasattr(_context, 'params'):
        del _context.params


def make_derivative_fn(model_apply, params, use_forward_mode=True):
    """
    Create a JIT-compatible derivative function for a given model and params.
    
    This returns a pure function that can be traced by JAX JIT.
    Uses batched forward-mode AD (jvp) for efficiency, inspired by FBPINNs.
    
    Features:
    - Batched JVP: Computes derivatives for entire batch in one pass (not per-point vmap)
    - Caching: Reuses intermediate derivatives (e.g., x_t reused when computing x_tt)
    
    Args:
        model_apply: Model apply function (params, x) -> y
        params: Current model parameters
        use_forward_mode: If True, use forward-mode AD (jvp) which is faster
                         for low-dimensional inputs (typical in PDEs)
    
    Returns:
        A derivative function with signature (Y, X, component, order) -> derivatives
    """
    if use_forward_mode:
        return _make_derivative_fn_batched(model_apply, params)
    else:
        return _make_derivative_fn_reverse(model_apply, params)


def _make_derivative_fn_batched(model_apply, params):
    """
    Batched forward-mode AD derivative using jvp (Jacobian-vector product).
    
    Key optimizations:
    1. Batched JVP: Instead of vmap over single points, we use tangent vectors 
       that span the whole batch (shape: batch_size x n_dims)
    2. Caching: Store intermediate functions to avoid recomputation
    
    This is significantly faster than per-point vmap for typical PINN batch sizes.
    """
    # Cache for storing wrapped functions for each (component, partial_order) combination
    # This allows x_tt to reuse the derivative function built for x_t
    _fn_cache = {}
    
    def deriv_fn(Y, X, component, order, create_graph=True):
        if len(order) == 0:
            raise ValueError("order tuple must have at least one element")
        
        n_dims = X.shape[1]
        
        # Build the base function for this component (cached)
        cache_key = (component,)
        if cache_key not in _fn_cache:
            def base_fn(x):
                return model_apply(params, x)[:, component]
            _fn_cache[cache_key] = base_fn
        
        # Build derivative function by chaining, caching each level
        current_fn = _fn_cache[cache_key]
        
        for i, dim in enumerate(order):
            # Check if we've already built this partial derivative
            partial_key = (component,) + order[:i+1]
            
            if partial_key in _fn_cache:
                current_fn = _fn_cache[partial_key]
            else:
                # Create new derivative function using batched JVP
                prev_fn = current_fn
                dim_to_diff = dim
                
                def make_next_fn(f, d):
                    def next_fn(x):
                        # Tangent vector: 1.0 in dimension d, 0 elsewhere
                        # Shape: (batch_size, n_dims)
                        tangent = jnp.zeros_like(x)
                        tangent = tangent.at[:, d].set(1.0)
                        _, deriv = jax.jvp(f, (x,), (tangent,))
                        return deriv
                    return next_fn
                
                current_fn = make_next_fn(prev_fn, dim_to_diff)
                _fn_cache[partial_key] = current_fn
        
        # Evaluate and reshape to (batch_size, 1)
        result = current_fn(X)
        return result.reshape(-1, 1)
    
    return deriv_fn


def _make_derivative_fn_forward(model_apply, params):
    """
    Forward-mode AD derivative using jvp (Jacobian-vector product).
    Per-point version with vmap. Kept as fallback.
    """
    def deriv_fn(Y, X, component, order, create_graph=True):
        if len(order) == 0:
            raise ValueError("order tuple must have at least one element")
        
        n_dims = X.shape[1]
        eye = jnp.eye(n_dims)
        
        def forward_single_point(x):
            def f(xi):
                return model_apply(params, xi.reshape(1, -1))[0, component]
            
            # Chain jvp calls for higher-order derivatives
            if len(order) == 1:
                _, deriv = jax.jvp(f, (x,), (eye[order[0]],))
                return deriv
            elif len(order) == 2:
                def df(xi):
                    _, d = jax.jvp(f, (xi,), (eye[order[0]],))
                    return d
                _, deriv = jax.jvp(df, (x,), (eye[order[1]],))
                return deriv
            elif len(order) == 3:
                def df(xi):
                    _, d = jax.jvp(f, (xi,), (eye[order[0]],))
                    return d
                def d2f(xi):
                    _, d = jax.jvp(df, (xi,), (eye[order[1]],))
                    return d
                _, deriv = jax.jvp(d2f, (x,), (eye[order[2]],))
                return deriv
            else:
                def chain_jvp(f_current, remaining_dims):
                    if not remaining_dims:
                        return f_current
                    dim = remaining_dims[0]
                    def df(xi):
                        _, d = jax.jvp(f_current, (xi,), (eye[dim],))
                        return d
                    return chain_jvp(df, remaining_dims[1:])
                
                final_fn = chain_jvp(f, order)
                return final_fn(x)
        
        result = jax.vmap(forward_single_point)(X)
        return result.reshape(-1, 1)
    
    return deriv_fn


def _make_derivative_fn_reverse(model_apply, params):
    """
    Reverse-mode AD derivative using grad.
    
    Fallback for cases where reverse-mode might be preferred.
    """
    def deriv_fn(Y, X, component, order, create_graph=True):
        if len(order) == 0:
            raise ValueError("order tuple must have at least one element")
        
        def single_point_fn(x):
            y = model_apply(params, x.reshape(1, -1))
            return y[0, component]
        
        fn = single_point_fn
        for dim in order:
            def make_grad(f, d):
                def gf(x):
                    return jax.grad(f)(x)[d]
                return gf
            fn = make_grad(fn, dim)
        
        result = jax.vmap(fn)(X)
        return result.reshape(-1, 1)
    
    return deriv_fn


def derivative(Y: jnp.ndarray, X: jnp.ndarray, component: int, 
               order: Tuple[int, ...], create_graph: bool = True) -> jnp.ndarray:
    """
    Compute derivatives of Y with respect to X using JAX autodiff.
    
    API compatible with PyTorch backend.
    
    NOTE: For JIT compilation, use make_derivative_fn() instead and have
    the trainer pass the derivative function to your PDE.
    
    Args:
        Y: Output tensor of shape (batch_size, n_outputs) - used to determine 
           which network output to differentiate (via context)
        X: Input tensor of shape (batch_size, n_dims)
        component: Which output component of Y to differentiate (0-indexed)
        order: Tuple specifying derivative order with respect to each input dimension.
               - (0,) means dY/dx0
               - (1,) means dY/dx1  
               - (0, 0) means d2Y/dx0^2
               - (0, 1) means d2Y/(dx0 dx1)
        create_graph: Ignored in JAX (always creates graph). For API compatibility.
    
    Returns:
        Array of shape (batch_size, 1) containing derivative values.
        
    Example:
        # Same API as PyTorch:
        u_x = derivative(Y, X, component=0, order=(0,))
        u_xx = derivative(Y, X, component=0, order=(0, 0))
    """
    if len(order) == 0:
        raise ValueError("order tuple must have at least one element")
    
    # Get network function and params from context
    apply_fn, params = get_context()
    if apply_fn is None:
        raise RuntimeError(
            "No computation context set. derivative() must be called within "
            "a PDE function during training, or set context with set_context()."
        )
    
    # Use the pure implementation
    deriv_fn = make_derivative_fn(apply_fn, params)
    return deriv_fn(Y, X, component, order, create_graph)


def gradient(Y: jnp.ndarray, X: jnp.ndarray, component: int = 0,
             create_graph: bool = True) -> jnp.ndarray:
    """
    Compute the full gradient of Y with respect to X using batched JVP.
    
    API compatible with PyTorch backend.
    """
    apply_fn, params = get_context()
    if apply_fn is None:
        raise RuntimeError("No computation context set.")
    
    n_dims = X.shape[1]
    
    def forward_component(x):
        return apply_fn(params, x)[:, component]
    
    # Compute gradient for each dimension using batched JVP
    grads = []
    for dim in range(n_dims):
        tangent = jnp.zeros_like(X)
        tangent = tangent.at[:, dim].set(1.0)
        _, deriv = jax.jvp(forward_component, (X,), (tangent,))
        grads.append(deriv)
    
    return jnp.stack(grads, axis=-1)


def laplacian(Y: jnp.ndarray, X: jnp.ndarray, component: int = 0,
              create_graph: bool = True) -> jnp.ndarray:
    """
    Compute the Laplacian (sum of second derivatives) of Y using batched JVP.
    
    API compatible with PyTorch backend.
    """
    apply_fn, params = get_context()
    if apply_fn is None:
        raise RuntimeError("No computation context set.")
    
    n_dims = X.shape[1]
    
    def forward_component(x):
        return apply_fn(params, x)[:, component]
    
    # Sum of second derivatives: d²f/dx_i² for each dimension
    laplacian_sum = jnp.zeros(X.shape[0])
    
    for dim in range(n_dims):
        tangent = jnp.zeros_like(X)
        tangent = tangent.at[:, dim].set(1.0)
        
        # First derivative
        def first_deriv(x):
            _, d = jax.jvp(forward_component, (x,), (tangent,))
            return d
        
        # Second derivative (same direction)
        _, d2 = jax.jvp(first_deriv, (X,), (tangent,))
        laplacian_sum = laplacian_sum + d2
    
    return laplacian_sum.reshape(-1, 1)


def divergence(Y: jnp.ndarray, X: jnp.ndarray,
               components: Tuple[int, ...] = None,
               create_graph: bool = True) -> jnp.ndarray:
    """
    Compute divergence of vector field using batched JVP.
    
    divergence = sum_i (d(Y[:, component_i]) / d(X[:, i]))
    
    API compatible with PyTorch backend.
    """
    apply_fn, params = get_context()
    if apply_fn is None:
        raise RuntimeError("No computation context set.")
    
    n_dims = X.shape[1]
    if components is None:
        components = tuple(range(n_dims))
    
    div_sum = jnp.zeros(X.shape[0])
    
    for i, comp in enumerate(components):
        def forward_comp(x, c=comp):
            return apply_fn(params, x)[:, c]
        
        # Derivative of component 'comp' w.r.t. dimension 'i'
        tangent = jnp.zeros_like(X)
        tangent = tangent.at[:, i].set(1.0)
        _, deriv = jax.jvp(forward_comp, (X,), (tangent,))
        div_sum = div_sum + deriv
    
    return div_sum.reshape(-1, 1)


class DifferentialOperators:
    """Context manager for differential operators."""
    
    def __init__(self, apply_fn: Callable, params: Dict):
        self.apply_fn = apply_fn
        self.params = params
    
    def __enter__(self):
        set_context(self.apply_fn, self.params)
        return self
    
    def __exit__(self, *args):
        clear_context()
