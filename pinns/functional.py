import torch
from typing import Tuple


def derivative(Y: torch.Tensor, X: torch.Tensor, component: int, 
               order: Tuple[int, ...], create_graph: bool = True) -> torch.Tensor:
    """
    Compute derivatives of Y with respect to X using automatic differentiation.
    
    Args:
        Y: Output tensor of shape (batch_size, n_outputs) from a neural network.
        X: Input tensor of shape (batch_size, n_dims) that Y depends on.
           Must have requires_grad=True.
        component: Which output component of Y to differentiate (0-indexed).
        order: Tuple specifying the derivative order with respect to each input dimension.
               Each element is the input dimension index to differentiate.
               - (0,) means ∂Y/∂x₀
               - (1,) means ∂Y/∂x₁
               - (0, 0) means ∂²Y/∂x₀²
               - (0, 1) means ∂²Y/(∂x₀∂x₁)
               - (0, 0, 0) means ∂³Y/∂x₀³
        create_graph: Whether to create computational graph for higher-order gradients.
                     Default: True (needed for training).
    
    Returns:
        Tensor of shape (batch_size, 1) containing the derivative values.
        
    Example:
        # Network with 2 inputs (t, x) and 1 output (u)
        X = torch.randn(100, 2, requires_grad=True)
        Y = network(X)  # shape (100, 1)
        
        # First derivative: ∂u/∂t
        u_t = derivative(Y, X, component=0, order=(0,))
        
        # First derivative: ∂u/∂x  
        u_x = derivative(Y, X, component=0, order=(1,))
        
        # Second derivative: ∂²u/∂x²
        u_xx = derivative(Y, X, component=0, order=(1, 1))
        
        # Mixed derivative: ∂²u/(∂t∂x)
        u_tx = derivative(Y, X, component=0, order=(0, 1))
        
        # For a 2-output network, derivative of second output:
        v_x = derivative(Y, X, component=1, order=(1,))
    """
    if not X.requires_grad:
        raise ValueError("X must have requires_grad=True to compute derivatives")
    
    if len(order) == 0:
        raise ValueError("order tuple must have at least one element")
    
    # Extract the component
    y = Y[:, component:component + 1]
    
    # Apply derivatives sequentially
    result = y
    for dim in order:
        grads = torch.autograd.grad(
            outputs=result,
            inputs=X,
            grad_outputs=torch.ones_like(result),
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True
        )[0]
        # If gradient is None (unused input), return zeros
        if grads is None:
            grads = torch.zeros_like(X)
        result = grads[:, dim:dim + 1]
    
    return result


def gradient(Y: torch.Tensor, X: torch.Tensor, component: int = 0,
             create_graph: bool = True) -> torch.Tensor:
    """
    Compute the full gradient of Y with respect to X.
    
    Args:
        Y: Output tensor of shape (batch_size, n_outputs) from a neural network.
        X: Input tensor of shape (batch_size, n_dims) that Y depends on.
        component: Which output component of Y to differentiate (0-indexed).
        create_graph: Whether to create computational graph for higher-order gradients.
    
    Returns:
        Tensor of shape (batch_size, n_dims) containing the gradient [∂Y/∂x₀, ∂Y/∂x₁, ...].
        
    Example:
        X = torch.randn(100, 2, requires_grad=True)
        Y = network(X)
        
        # Get gradient: [∂u/∂t, ∂u/∂x]
        grad_u = gradient(Y, X, component=0)
    """
    if not X.requires_grad:
        raise ValueError("X must have requires_grad=True to compute gradients")
    
    y = Y[:, component:component + 1]
    
    grads = torch.autograd.grad(
        outputs=y,
        inputs=X,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=True
    )[0]
    
    return grads


def laplacian(Y: torch.Tensor, X: torch.Tensor, component: int = 0,
              create_graph: bool = True) -> torch.Tensor:
    """
    Compute the Laplacian of Y with respect to X: ∇²Y = ∂²Y/∂x₀² + ∂²Y/∂x₁² + ...
    
    Args:
        Y: Output tensor of shape (batch_size, n_outputs) from a neural network.
        X: Input tensor of shape (batch_size, n_dims) that Y depends on.
        component: Which output component of Y to differentiate (0-indexed).
        create_graph: Whether to create computational graph for higher-order gradients.
    
    Returns:
        Tensor of shape (batch_size, 1) containing the Laplacian values.
        
    Example:
        # Laplacian for 2D: ∂²u/∂x² + ∂²u/∂y²
        lap_u = laplacian(Y, X, component=0)
    """
    n_dims = X.shape[1]
    
    lap = torch.zeros(X.shape[0], 1, device=X.device, dtype=X.dtype)
    
    for dim in range(n_dims):
        lap = lap + derivative(Y, X, component, (dim, dim), create_graph=create_graph)
    
    return lap


def divergence(Y: torch.Tensor, X: torch.Tensor, 
               components: Tuple[int, ...] = None,
               create_graph: bool = True) -> torch.Tensor:
    """
    Compute the divergence of a vector field: ∇·Y = ∂Y₀/∂x₀ + ∂Y₁/∂x₁ + ...
    
    Args:
        Y: Output tensor of shape (batch_size, n_outputs) representing a vector field.
        X: Input tensor of shape (batch_size, n_dims) that Y depends on.
        components: Which output components form the vector field. 
                   Default: (0, 1, ..., n_dims-1)
        create_graph: Whether to create computational graph for higher-order gradients.
    
    Returns:
        Tensor of shape (batch_size, 1) containing the divergence values.
        
    Example:
        # For velocity field (u, v) with inputs (x, y):
        # div = ∂u/∂x + ∂v/∂y
        div_vel = divergence(Y, X, components=(0, 1))
    """
    n_dims = X.shape[1]
    
    if components is None:
        components = tuple(range(n_dims))
    
    if len(components) != n_dims:
        raise ValueError(f"components must have {n_dims} elements to match input dimensions")
    
    div = torch.zeros(X.shape[0], 1, device=X.device, dtype=X.dtype)
    
    for dim, comp in enumerate(components):
        div = div + derivative(Y, X, comp, (dim,), create_graph=create_graph)
    
    return div
