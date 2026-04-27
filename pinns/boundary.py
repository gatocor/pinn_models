import torch
import numpy as np
from dataclasses import dataclass
from typing import Union, Callable, Tuple, Optional, List

def _call_value_function(value_fn, x: torch.Tensor) -> torch.Tensor:
    """
    Helper to call a value function, handling both numpy and torch functions.
    
    Converts tensor to numpy for the function call, then converts result back to tensor.
    This allows users to define BC functions using numpy operations.
    """
    # Convert to numpy for user-defined functions that may expect numpy
    x_np = x.detach().cpu().numpy()
    result = value_fn(x_np)
    
    # Convert back to tensor
    if isinstance(result, np.ndarray):
        result = torch.tensor(result, device=x.device, dtype=x.dtype)
    elif not isinstance(result, torch.Tensor):
        result = torch.tensor(result, device=x.device, dtype=x.dtype)
    else:
        result = result.to(device=x.device, dtype=x.dtype)
    
    if result.dim() > 1:
        result = result.squeeze(-1)
    
    return result


@dataclass
class DirichletBC:
    """
    Dirichlet boundary condition: u(x) = value on the boundary.
    
    Args:
        boundary (tuple): Tuple specifying the boundary location.
                         Each element corresponds to a dimension:
                         - 0: lower boundary (e.g., x_min)
                         - 1: upper boundary (e.g., x_max)
                         - None: not constrained in this dimension
                         Example: (0, None) = x_min plane in 2D
                                  (None, 1) = y_max plane in 2D
                                  (0, 0) = corner at (x_min, y_min)
        value (float, torch.Tensor, or Callable): The boundary value.
                         If callable, should take x as numpy array (batch_size, n_dims) 
                         and return values of shape (batch_size,) or (batch_size, 1).
        component (int): Output component index this condition applies to.
                        Default: 0 (first output)
        subdomain (tuple, optional): Constrain sampling to a subdomain of the boundary.
                         Tuple of (min_value, max_value) specifying the actual coordinate
                         limits for the free (non-fixed) dimension of the boundary.
                         Example: subdomain=(-0.02, 0.02) restricts sampling to 
                         the range x ∈ (-0.02, 0.02) for a boundary at t=0.
        name (str, optional): Name for this BC (used in plots and compile dict).
        sampling_method: Sampling method for generating points. Can be:
                        - "uniform" (default): Standard uniform random sampling
                        - "latin_hypercube" or "lhs": Latin Hypercube Sampling
                        - "sobol": Sobol quasi-random sequence
                        - "halton": Halton quasi-random sequence
                        - Callable: Custom function (n_points, n_dims, rng) -> ndarray in [0,1]^n
        sampling_transform (Callable, optional): Custom transform function for sampling.
                        Takes samples in [0,1]^n and returns transformed coordinates.
                        This is like an inverse CDF. Points outside domain are rejected.
        
    Example:
        # u = 0 at x_min boundary for first output component
        bc1 = DirichletBC(boundary=(0, None), value=0.0, component=0)
        
        # u = sin(y) at x_max boundary (using numpy)
        bc2 = DirichletBC(boundary=(1, None), value=lambda x: np.sin(x[:, 1]), component=0)
        
        # Temperature = 100 at bottom boundary (y_min)
        bc3 = DirichletBC(boundary=(None, 0), value=100.0, component=0)
        
        # IC only in crack region: x ∈ (-0.02, 0.02) at t=0 boundary
        bc4 = DirichletBC(boundary=(None, 0), value=IC, component=0, 
                          subdomain=(-0.02, 0.02))
        
        # Custom sampling with more points near x=0 (inverse CDF transform)
        def gaussian_transform(u):
            from scipy.stats import norm
            return norm.ppf(u, loc=0, scale=0.05)  # Gaussian centered at 0
        bc5 = DirichletBC(boundary=(None, 0), value=IC, component=0, 
                          sampling_transform=gaussian_transform)
    """
    boundary: Tuple
    value: Union[float, torch.Tensor, Callable]
    component: int = 0
    subdomain: Optional[Tuple] = None
    name: Optional[str] = None
    sampling_method: Union[str, Callable] = "uniform"
    sampling_transform: Optional[Callable] = None
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the boundary value at given points.
        
        Args:
            x: Input tensor of shape (batch_size, n_dims)
            
        Returns:
            Tensor of shape (batch_size,) with boundary values
        """
        if callable(self.value):
            return _call_value_function(self.value, x)
        else:
            return torch.full((x.shape[0],), self.value, device=x.device, dtype=x.dtype)
    
    def get_boundary_dims(self) -> list:
        """Get list of (dimension, side) tuples for this boundary."""
        dims = []
        for i, side in enumerate(self.boundary):
            if side is not None:
                dims.append((i, 'lower' if side == 0 else 'upper'))
        return dims


@dataclass
class NeumannBC:
    """
    Neumann boundary condition: du/dn = value on the boundary.
    
    The normal derivative is computed as the derivative with respect to
    the dimension perpendicular to the boundary.
    
    Args:
        boundary (tuple): Tuple specifying the boundary location.
                         Each element corresponds to a dimension:
                         - 0: lower boundary (e.g., x_min)
                         - 1: upper boundary (e.g., x_max)
                         - None: not constrained in this dimension
                         Example: (0, None) = x_min plane, normal is -x direction
                                  (1, None) = x_max plane, normal is +x direction
        value (float, torch.Tensor, or Callable): The normal derivative value.
                         If callable, should take x as numpy array (batch_size, n_dims) 
                         and return values of shape (batch_size,) or (batch_size, 1).
        component (int): Output component index this condition applies to.
                        Default: 0 (first output)
        name (str, optional): Name for this BC (used in plots and compile dict).
        sampling_method: Sampling method for generating points. Can be:
                        - "uniform" (default): Standard uniform random sampling
                        - "latin_hypercube" or "lhs": Latin Hypercube Sampling
                        - "sobol": Sobol quasi-random sequence
                        - "halton": Halton quasi-random sequence
                        - Callable: Custom function (n_points, n_dims, rng) -> ndarray in [0,1]^n
        sampling_transform (Callable, optional): Custom transform function for sampling.
        
    Example:
        # du/dx = 0 at x_min (zero flux)
        bc1 = NeumannBC(boundary=(0, None), value=0.0, component=0)
        
        # du/dy = -1 at y_max (heat flux)
        bc2 = NeumannBC(boundary=(None, 1), value=-1.0, component=0)
        
        # Spatially varying flux (using numpy)
        bc3 = NeumannBC(boundary=(1, None), value=lambda x: x[:, 1]**2, component=0)
        
        # BC with subdomain constraint: y ∈ (0.4, 0.6) on the x_max boundary
        bc4 = NeumannBC(boundary=(1, None), value=0.0, component=0,
                        subdomain=(0.4, 0.6))
    """
    boundary: Tuple
    value: Union[float, torch.Tensor, Callable]
    component: int = 0
    subdomain: Optional[Tuple] = None
    name: Optional[str] = None
    sampling_method: Union[str, Callable] = "uniform"
    sampling_transform: Optional[Callable] = None
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the boundary derivative value at given points.
        
        Args:
            x: Input tensor of shape (batch_size, n_dims)
            
        Returns:
            Tensor of shape (batch_size,) with derivative values
        """
        if callable(self.value):
            return _call_value_function(self.value, x)
        else:
            return torch.full((x.shape[0],), self.value, device=x.device, dtype=x.dtype)
    
    def get_boundary_dims(self) -> list:
        """Get list of (dimension, side) tuples for this boundary."""
        dims = []
        for i, side in enumerate(self.boundary):
            if side is not None:
                dims.append((i, 'lower' if side == 0 else 'upper'))
        return dims
    
    def get_normal_direction(self) -> Tuple[int, int]:
        """
        Get the normal direction for this boundary.
        
        Returns:
            Tuple of (dimension, sign) where sign is -1 for lower boundary
            and +1 for upper boundary.
        """
        for i, side in enumerate(self.boundary):
            if side is not None:
                sign = -1 if side == 0 else 1
                return (i, sign)
        raise ValueError("No boundary dimension specified")


@dataclass  
class RobinBC:
    """
    Robin (mixed) boundary condition: a*u + b*du/dn = value on the boundary.
    
    Args:
        boundary (tuple): Tuple specifying the boundary location.
                         Each element corresponds to a dimension:
                         - 0: lower boundary
                         - 1: upper boundary
                         - None: not constrained in this dimension
        alpha (float): Coefficient for u term.
        beta (float): Coefficient for du/dn term.
        value (float, torch.Tensor, or Callable): The boundary value.
                         If callable, should take x as numpy array (batch_size, n_dims) 
                         and return values of shape (batch_size,) or (batch_size, 1).
        component (int): Output component index this condition applies to.
                        Default: 0 (first output)
        name (str, optional): Name for this BC (used in plots and compile dict).
        sampling_method: Sampling method for generating points. Can be:
                        - "uniform" (default): Standard uniform random sampling
                        - "latin_hypercube" or "lhs": Latin Hypercube Sampling
                        - "sobol": Sobol quasi-random sequence
                        - "halton": Halton quasi-random sequence
                        - Callable: Custom function (n_points, n_dims, rng) -> ndarray in [0,1]^n
        sampling_transform (Callable, optional): Custom transform function for sampling.
        
    Example:
        # Convective BC: h*u + k*du/dn = h*u_inf
        bc = RobinBC(boundary=(1, None), alpha=10.0, beta=1.0, value=100.0, component=0)
    """
    boundary: Tuple
    alpha: float
    beta: float
    value: Union[float, torch.Tensor, Callable]
    component: int = 0
    subdomain: Optional[Tuple] = None
    name: Optional[str] = None
    sampling_method: Union[str, Callable] = "uniform"
    sampling_transform: Optional[Callable] = None
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the boundary value at given points.
        
        Args:
            x: Input tensor of shape (batch_size, n_dims)
            
        Returns:
            Tensor of shape (batch_size,) with boundary values
        """
        if callable(self.value):
            return _call_value_function(self.value, x)
        else:
            return torch.full((x.shape[0],), self.value, device=x.device, dtype=x.dtype)
    
    def get_boundary_dims(self) -> list:
        """Get list of (dimension, side) tuples for this boundary."""
        dims = []
        for i, side in enumerate(self.boundary):
            if side is not None:
                dims.append((i, 'lower' if side == 0 else 'upper'))
        return dims
    
    def get_normal_direction(self) -> Tuple[int, int]:
        """
        Get the normal direction for this boundary.
        
        Returns:
            Tuple of (dimension, sign) where sign is -1 for lower boundary
            and +1 for upper boundary.
        """
        for i, side in enumerate(self.boundary):
            if side is not None:
                sign = -1 if side == 0 else 1
                return (i, sign)
        raise ValueError("No boundary dimension specified")

@dataclass
class PointsetBC:
    """
    Pointset boundary/data condition: u(x_i) = y_i for given data points.
    
    This condition enforces that the network output matches given values
    at specific input points. Useful for:
    - Data-driven constraints
    - Initial conditions
    - Experimental data fitting
    - Interior data constraints
    
    Args:
        inputs (torch.Tensor or np.ndarray): Input points of shape (n_points, n_dims).
        outputs (torch.Tensor or np.ndarray): Target values of shape (n_points,) or 
                                              (n_points, n_outputs).
        component (int or None): Output component index this condition applies to.
                                If None and outputs is 2D, applies to all components.
                                Default: 0 (first output)
        
    Example:
        import numpy as np
        
        # Single output component
        x_data = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        y_data = np.array([0.0, 0.5, 1.0])
        bc = PointsetBC(inputs=x_data, outputs=y_data, component=0)
        
        # Multiple output components
        y_data_multi = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])  # shape (3, 2)
        bc = PointsetBC(inputs=x_data, outputs=y_data_multi, component=None)
        
        # Initial condition at t=0
        x_initial = np.column_stack([np.linspace(0, 1, 100), np.zeros(100)])  # (x, t=0)
        u_initial = np.sin(np.pi * x_initial[:, 0])
        ic = PointsetBC(inputs=x_initial, outputs=u_initial, component=0)
    """
    inputs: Union[torch.Tensor, 'np.ndarray']
    outputs: Union[torch.Tensor, 'np.ndarray']
    component: Optional[int] = 0
    name: Optional[str] = None
    
    def __post_init__(self):
        """Convert inputs to tensors if needed."""
        import numpy as np
        
        if isinstance(self.inputs, np.ndarray):
            self.inputs = torch.from_numpy(self.inputs).float()
        if isinstance(self.outputs, np.ndarray):
            self.outputs = torch.from_numpy(self.outputs).float()
        
        # Ensure inputs is 2D
        if self.inputs.dim() == 1:
            self.inputs = self.inputs.unsqueeze(-1)
        
        # Ensure outputs is at least 1D
        if self.outputs.dim() == 0:
            self.outputs = self.outputs.unsqueeze(0)
    
    def get_inputs(self, device='cpu', dtype=torch.float32) -> torch.Tensor:
        """
        Get input points as tensor.
        
        Args:
            device: PyTorch device
            dtype: PyTorch dtype
            
        Returns:
            Tensor of shape (n_points, n_dims)
        """
        return self.inputs.to(device=device, dtype=dtype)
    
    def get_outputs(self, device='cpu', dtype=torch.float32) -> torch.Tensor:
        """
        Get output values as tensor.
        
        Args:
            device: PyTorch device
            dtype: PyTorch dtype
            
        Returns:
            Tensor of shape (n_points,) or (n_points, n_outputs)
        """
        return self.outputs.to(device=device, dtype=dtype)
    
    def __len__(self):
        """Return number of data points."""
        return self.inputs.shape[0]
    
    @property
    def n_points(self):
        """Number of data points."""
        return self.inputs.shape[0]
    
    @property
    def n_dims(self):
        """Input dimensionality."""
        return self.inputs.shape[1]


class BoundaryConditions:
    """
    Collection of boundary conditions for a PINN problem.
    
    Example:
        bcs = BoundaryConditions()
        
        # Add Dirichlet BCs
        bcs.add(DirichletBC(boundary=(0, None), value=0.0, component=0))
        bcs.add(DirichletBC(boundary=(1, None), value=1.0, component=0))
        
        # Add Neumann BC
        bcs.add(NeumannBC(boundary=(None, 0), value=0.0, component=0))
        
        # Add data points
        bcs.add(PointsetBC(inputs=x_data, outputs=y_data, component=0))
        
        # Get all conditions
        for bc in bcs:
            print(bc)
    """
    
    def __init__(self):
        self.dirichlet: list[DirichletBC] = []
        self.neumann: list[NeumannBC] = []
        self.robin: list[RobinBC] = []
        self.pointset: list[PointsetBC] = []
    
    def add(self, bc: Union[DirichletBC, NeumannBC, RobinBC, PointsetBC]):
        """Add a boundary condition."""
        if isinstance(bc, DirichletBC):
            self.dirichlet.append(bc)
        elif isinstance(bc, NeumannBC):
            self.neumann.append(bc)
        elif isinstance(bc, RobinBC):
            self.robin.append(bc)
        elif isinstance(bc, PointsetBC):
            self.pointset.append(bc)
        else:
            raise TypeError(f"Unknown boundary condition type: {type(bc)}")
    
    def __iter__(self):
        """Iterate over all boundary conditions."""
        yield from self.dirichlet
        yield from self.neumann
        yield from self.robin
        yield from self.pointset
    
    def __len__(self):
        """Total number of boundary conditions."""
        return len(self.dirichlet) + len(self.neumann) + len(self.robin) + len(self.pointset)
    
    def __repr__(self):
        return (
            f"BoundaryConditions("
            f"dirichlet={len(self.dirichlet)}, "
            f"neumann={len(self.neumann)}, "
            f"robin={len(self.robin)}, "
            f"pointset={len(self.pointset)})"
        )


# ============================================================================
# Mesh-domain boundary conditions
# ============================================================================

@dataclass
class MeshNodeBC:
    """
    Boundary condition applied at a selected subset of mesh nodes.

    Created by :meth:`DomainMesh.add_dirichlet` and
    :meth:`DomainMesh.add_neumann`.  Node positions are pre-resolved at
    construction time so sampling is just a random draw from the stored nodes.

    Args:
        node_positions: ``(n_selected, spatial_dims)`` array of node coordinates.
        value: Target value — scalar float or ``(x_np) -> np.ndarray`` callable.
        bc_type: ``"dirichlet"`` or ``"neumann"``.
        component: Output component index (default 0).
        name: Label used in compile dicts and plots.
        t_mode: Time sampling strategy for spatiotemporal domains:
                ``None`` for purely spatial domains,
                ``"all"`` to sample the BC at random times in [t_min, t_max],
                ``"t_min"`` for initial-condition enforcement (t = t_min),
                ``"t_max"`` for final-condition enforcement (t = t_max).
        t_min: Domain time minimum (set automatically by :class:`DomainMesh`).
        t_max: Domain time maximum (set automatically by :class:`DomainMesh`).
        normals: Optional ``(n_selected, spatial_dims)`` per-node outward unit
                 normals (required for spatial Neumann BCs).

    Example::

        domain = DomainMesh(mesh, t_interval=[0, 1])

        # u = 0 on left-wall nodes at all times
        domain.add_dirichlet(
            select=lambda v: v[:, 0] < 1e-6,
            value=0.0, component=0, name="left_wall"
        )

        # u = sin(pi*y) initial condition at t=0
        domain.add_dirichlet(
            select=np.arange(100),
            value=lambda x: np.sin(np.pi * x[:, 1]),
            component=0, name="ic", t_mode="t_min"
        )
    """
    node_positions: 'np.ndarray'          # (n_selected, spatial_dims)
    value: Union[float, Callable]
    bc_type: str                          # "dirichlet" or "neumann"
    component: int = 0
    name: Optional[str] = None
    t_mode: Optional[str] = None         # None | "all" | "t_min" | "t_max"
    t_min: float = 0.0
    t_max: float = 1.0
    edges: Optional['np.ndarray'] = None         # (n_edges, 2) vertex index pairs into the full mesh
    edge_lengths: Optional['np.ndarray'] = None  # (n_edges,)
    edge_normals: Optional['np.ndarray'] = None  # (n_edges, 2) outward unit normals per edge

    def get_value(self, x) -> np.ndarray:
        """Return target values as a numpy array (backend-agnostic)."""
        if callable(self.value):
            if hasattr(x, 'detach'):  # torch tensor
                x_np = x.detach().cpu().numpy()
            else:
                x_np = np.asarray(x)
            result = self.value(x_np)
            return np.asarray(result, dtype=np.float32).squeeze()
        n = x.shape[0]
        return np.full(n, self.value, dtype=np.float32)


@dataclass
class MeshCustomBC:
    """
    Custom boundary condition applied at a selected subset of mesh nodes.

    Instead of a fixed ``value`` + ``component`` target, you provide a
    **residual function** ``f`` with the same signature as ``pde_fn``::

        f(x, y, params, derivative) -> residual  # shape (n,) or tuple of (n,)

    The trainer minimises ``mean(f(...)²)``.  Use this for mixed BCs such as
    traction conditions in elasticity where the residual involves derivatives
    of several output components.

    Created by :meth:`DomainMesh.add_bc`.

    Args:
        node_positions: ``(n_nodes, spatial_dims)`` sampled node coordinates.
        f: Residual callable with signature
           ``f(x, y, params, derivative) -> array`` **or** a tuple of arrays.
        name: Label used in compile/weight dicts and plots.
        edges: ``(n_edges, 2)`` vertex-pair array (for sampling, optional).
        edge_lengths: ``(n_edges,)`` edge lengths (for sampling, optional).
    """
    node_positions: 'np.ndarray'         # (n_nodes, spatial_dims)
    f:              Callable             # residual function
    name:           Optional[str] = None
    output_names:   Optional[List[str]] = None  # per-output names when f returns a tuple
    t_mode:         Optional[str] = None  # None | "all" | "t_min" | "t_max"
    edges:          Optional['np.ndarray'] = None
    edge_lengths:   Optional['np.ndarray'] = None
    # ── weak-form fields ───────────────────────────────────────────────────────
    is_weak:        bool = False          # True if f accepts phi (line-integral RHS)
    weak_fn:        Optional[Callable] = None  # original f with phi signature


@dataclass
class MeshDirichletBC:
    """
    Dirichlet boundary condition on a mesh-based domain.

    Used with :class:`DomainMesh`.  The ``boundary_type`` selects which part of
    the domain is constrained:

    - ``"surface"``  – the mesh surface (all faces).  Sampling distributes
      points over the surface and, if a time interval is present, sweeps
      uniformly in time.
    - ``"t_min"``    – the initial-time plane (t = t_min), spatially sampled
      from the mesh interior.
    - ``"t_max"``    – the final-time plane (t = t_max), spatially sampled
      from the mesh interior.

    Args:
        boundary_type: One of ``"surface"``, ``"t_min"``, ``"t_max"``.
        value: The Dirichlet value.  Scalar or callable with signature
               ``(x: np.ndarray) -> np.ndarray``.
        component: Output component index (default 0).
        name: Name used in compile dicts and plots.
    """
    boundary_type: str
    value: Union[float, Callable]
    component: int = 0
    name: Optional[str] = None

    def get_value(self, x: 'torch.Tensor') -> 'torch.Tensor':
        if callable(self.value):
            return _call_value_function(self.value, x)
        return torch.full((x.shape[0],), self.value, device=x.device, dtype=x.dtype)


@dataclass
class MeshNeumannBC:
    """
    Neumann boundary condition on a mesh-based domain.

    Supported ``boundary_type`` values:

    - ``"t_min"``  – initial-time plane; normal points in the –t direction.
    - ``"t_max"``  – final-time plane;   normal points in the +t direction.
    - ``"surface"`` is **not** supported for Neumann conditions because the
      outward normal is face-dependent and requires storing per-point normals.
      Use a :class:`MeshDirichletBC` or a custom :class:`PointsetBC` instead.

    Args:
        boundary_type: ``"t_min"`` or ``"t_max"``.
        value: The normal-derivative value.  Scalar or callable.
        component: Output component index (default 0).
        name: Name used in compile dicts and plots.
        spatial_dims: Number of spatial dimensions in the domain.  Set
                      automatically by :meth:`DomainMesh.add_neumann`.
    """
    boundary_type: str
    value: Union[float, Callable]
    component: int = 0
    name: Optional[str] = None
    spatial_dims: int = 0

    def get_value(self, x: 'torch.Tensor') -> 'torch.Tensor':
        if callable(self.value):
            return _call_value_function(self.value, x)
        return torch.full((x.shape[0],), self.value, device=x.device, dtype=x.dtype)

    def get_normal_direction(self):
        """Return (normal_dim, normal_sign) for the time axis."""
        if self.boundary_type == 't_min':
            return self.spatial_dims, -1
        elif self.boundary_type == 't_max':
            return self.spatial_dims, 1
        raise ValueError(
            f"Neumann BC on boundary_type='{self.boundary_type}' is not supported. "
            "Use 't_min' or 't_max'."
        )

