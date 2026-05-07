import torch
import numpy as np
from dataclasses import dataclass, field
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
    subspace: Optional[List] = None
    time_subspace: Optional[Tuple] = None
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
        
        # Restrict to y ∈ (0.4, 0.6) on the x_max boundary (2-D domain)
        bc4 = NeumannBC(boundary=(1, None), value=0.0, component=0,
                        subspace=[(0.4, 0.6)])
    """
    boundary: Tuple
    value: Union[float, torch.Tensor, Callable]
    component: int = 0
    subspace: Optional[List] = None
    time_subspace: Optional[Tuple] = None
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

    @property
    def normal_dim(self) -> int:
        """Index of the dimension perpendicular to this boundary."""
        dim, _ = self.get_normal_direction()
        return dim


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
    subspace: Optional[List] = None
    time_subspace: Optional[Tuple] = None
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

    @property
    def normal_dim(self) -> int:
        """Index of the dimension perpendicular to this boundary."""
        dim, _ = self.get_normal_direction()
        return dim

@dataclass
class PointsetBC:
    """
    Pointset boundary/data condition: u(x_i) = y_i for given data points.

    Instead of a single target scalar per point you now provide an
    ``(N, K)`` target array together with a ``components`` list that maps
    each column of the target to a network output index.  The trainer
    minimises one MSE sub-loss per column:

    .. math::

        \\mathcal{L}_k = \\dfrac{1}{N}\\sum_{i=1}^{N}
            \\bigl(u_{\\mathtt{components}[k]}(x_i) - y_{ik}\\bigr)^2

    Each sub-loss is registered under its own name for independent weight
    control.  If only a single column is supplied the sub-loss keeps the
    base ``name``; otherwise the names are ``name_0``, ``name_1``, … (or the
    list supplied via ``output_names``).

    Args:
        inputs: Input coordinates of shape ``(N, n_dims)``.
        outputs: Target values of shape ``(N, K)``.  A 1-D array of length
                 ``N`` is treated as ``(N, 1)`` (single column).
        components: Network output index for each column of *outputs*.
                    A single integer is equivalent to ``[int]``.
        name: Base label used in weight dicts and plots.
        output_names: Optional per-column labels.  If omitted, names are
                      ``name_0``, ``name_1``, … (or just ``name`` when
                      there is only one column).

    Example::

        import numpy as np

        x_data = np.random.rand(200, 2)   # (N, 2) coordinates
        y_data = np.random.rand(200, 3)   # (N, 3) targets for u0, u1, u3

        bc = PointsetBC(
            inputs=x_data,
            outputs=y_data,
            components=[0, 1, 3],
            name='obs',
        )
        # Sub-losses: 'obs_0', 'obs_1', 'obs_2'
    """
    inputs:       Union[torch.Tensor, 'np.ndarray']
    outputs:      Union[torch.Tensor, 'np.ndarray']
    components:   Union[int, List[int]] = 0
    name:         Optional[str] = None
    output_names: Optional[List[str]] = None

    def __post_init__(self):
        import numpy as np

        # ── convert to float32 tensors ────────────────────────────────────
        if isinstance(self.inputs, np.ndarray):
            self.inputs = torch.from_numpy(self.inputs).float()
        elif not isinstance(self.inputs, torch.Tensor):
            self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        if isinstance(self.outputs, np.ndarray):
            self.outputs = torch.from_numpy(self.outputs).float()
        elif not isinstance(self.outputs, torch.Tensor):
            self.outputs = torch.tensor(self.outputs, dtype=torch.float32)

        # ── ensure 2-D shapes ─────────────────────────────────────────────
        if self.inputs.dim() == 1:
            self.inputs = self.inputs.unsqueeze(-1)
        if self.outputs.dim() == 1:
            self.outputs = self.outputs.unsqueeze(-1)  # (N,) → (N, 1)

        # ── normalise components to list ──────────────────────────────────
        if isinstance(self.components, int):
            self.components = [self.components]

        # ── validate shape consistency ────────────────────────────────────
        n_cols = self.outputs.shape[1]
        if n_cols != len(self.components):
            raise ValueError(
                f"PointsetBC '{self.name}': outputs has {n_cols} column(s) but "
                f"components has {len(self.components)} element(s).  They must match."
            )

    # ── convenience accessors ─────────────────────────────────────────────

    def get_input_names(self) -> List[str]:
        """Per-column sub-loss names (resolved against ``name``)."""
        base = self.name or 'pointset'
        if self.output_names is not None:
            return list(self.output_names)
        if len(self.components) == 1:
            return [base]
        return [f'{base}_{k}' for k in range(len(self.components))]

    def get_inputs(self, device='cpu', dtype=torch.float32) -> torch.Tensor:
        """Input coordinates as a ``(N, n_dims)`` tensor."""
        return self.inputs.to(device=device, dtype=dtype)

    def get_outputs(self, device='cpu', dtype=torch.float32) -> torch.Tensor:
        """Target values as a ``(N, K)`` tensor."""
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

    Instantiated directly and appended to ``domain.boundary_conditions``,
    or created by a :class:`~pinns.problem.Problem` / trainer helper.  Node
    positions are pre-resolved at construction time so sampling is just a
    random draw from the stored nodes.

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
            component=0, name="ic",
            time_window=[0.0, 0.0],  # t_min=0
        )
    """
    node_positions: 'np.ndarray'          # (n_selected, spatial_dims)
    value: Union[float, Callable]
    bc_type: str                          # "dirichlet" or "neumann"
    component: int = 0
    name: Optional[str] = None
    # time_window encodes when this BC is active:
    #   None              – purely spatial domain (no time axis)
    #   [t_a, t_b]        – continuous interval; sampled uniformly in [t_a, t_b]
    #   [t0, t1, t2, ...] – discrete list of exact time points (≥3 elements)
    #   [t_a, t_b]  with t_a==t_b  – fixed single time point
    time_window: Optional[Union[List, Tuple]] = None
    t_min: float = 0.0
    t_max: float = 1.0
    node_indices: Optional['np.ndarray'] = None  # (n_nodes,) int indices into the full mesh vertex array
    edges: Optional['np.ndarray'] = None         # (n_edges, 2) vertex index pairs into the full mesh
    edge_lengths: Optional['np.ndarray'] = None  # (n_edges,)
    edge_normals: Optional['np.ndarray'] = None  # (n_edges, 2) outward unit normals per edge

    # ── backward-compat property ──────────────────────────────────────────
    @property
    def t_mode(self) -> Optional[str]:
        """Derived from ``time_window`` for backward compatibility.

        Returns one of ``None``, ``"t_min"``, ``"t_max"``, or ``"all"``.
        """
        tw = self.time_window
        if tw is None:
            return None
        pts = [float(v) for v in tw]
        if len(pts) == 0:
            return None
        if len(pts) == 1:
            a = pts[0]
            if abs(a - self.t_min) < 1e-10:
                return "t_min"
            if abs(a - self.t_max) < 1e-10:
                return "t_max"
            return "all"
        # 2-element interval (continuous) or multi-point discrete list
        a, b = pts[0], pts[-1]
        if abs(a - self.t_min) < 1e-10 and abs(b - self.t_min) < 1e-10:
            return "t_min"
        if abs(a - self.t_max) < 1e-10 and abs(b - self.t_max) < 1e-10:
            return "t_max"
        return "all"

    def is_full_time_coverage(self) -> bool:
        """True if the BC is active over the entire time domain ``[t_min, t_max]``.

        Used by :class:`~pinns.problem_weak.ProblemWeak` to decide which BCs
        can be imposed as hard constraints.  Returns ``True`` for purely
        spatial domains (no time axis) as well as for continuous-interval BCs
        whose window spans ``[t_min, t_max]`` exactly.
        """
        tw = self.time_window
        if tw is None:
            return True   # spatial domain, always active
        if len(tw) != 2:
            return False  # discrete list of individual time points
        a, b = float(tw[0]), float(tw[1])
        return abs(a - self.t_min) < 1e-10 and abs(b - self.t_max) < 1e-10

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
class PeriodicBC:
    """
    Periodic boundary condition pairing nodes on two boundaries.

    Enforces :math:`u(\\mathbf{x}_A) = u(\\mathbf{x}_B)` for matched node
    pairs from edge set A and edge set B by minimising
    :math:`\\text{mean}\\bigl((u(\\mathbf{x}_A) - u(\\mathbf{x}_B))^2\\bigr)`.

    Created by :meth:`DomainMesh.add_periodic`.

    Args:
        node_positions_a: ``(n_pairs, spatial_dims)`` — source boundary nodes.
        node_positions_b: ``(n_pairs, spatial_dims)`` — matched target nodes.
        component: Output component to enforce periodicity on, or ``None`` for
                   all components simultaneously.
        name: Label used in weights dicts.
    """
    node_positions_a: 'np.ndarray'        # (n_pairs, 2)
    node_positions_b: 'np.ndarray'        # (n_pairs, 2)  — matched to a
    component:        Optional[int] = None # None = all components
    name:             Optional[str] = None
    match_x_derivative: bool = False       # also penalise u_x(a) - u_x(b)
    bc_type:          str = 'periodic'    # sentinel for trainer dispatch


@dataclass
class CubicPeriodicBC:
    """
    Periodic boundary condition for :class:`~pinns.domain.DomainCubic` domains.

    Instead of pre-computed node arrays, stores only which spatial dimension is
    periodic.  The trainer samples the required point pairs automatically at
    compile time.

    Created by :meth:`~pinns.domain.DomainCubic.add_periodic`.

    Args:
        dim: Spatial (or temporal) dimension index that is periodic.
        n_pairs: Number of collocation pairs to sample.
        component: Output component to enforce, or ``None`` to enforce all
                   components (each gets its own sub-loss named
                   ``name_0``, ``name_1``, …).
        name: Base label used in weights dicts.
        match_x_derivative: If ``True``, also penalise
            :math:`|\\partial_x u(x_l) - \\partial_x u(x_r)|^2`.
    """
    dim:                int
    n_pairs:            int = 200
    component:          Optional[int] = None
    name:               Optional[str] = None
    match_x_derivative: bool = True
    bc_type:            str = 'cubic_periodic'


@dataclass
class InitialConditionBC:
    """
    Initial condition: u(x, t_min) = value over the full spatial domain.

    This is a dedicated BC type (not a special-cased :class:`DirichletBC`)
    so it can be recognised, plotted, and filtered independently.

    Created by :meth:`~pinns.domain.DomainCubic.add_initial`.

    Args:
        boundary: Tuple ``(None, …, None, 0)`` — all spatial dims free,
                  time dimension pinned to lower face.  Computed automatically
                  by :meth:`add_initial`.
        value: IC value.  Scalar or callable ``(x: np.ndarray) -> np.ndarray``
               taking points ``(n, n_dims)`` and returning ``(n,)`` or
               ``(n, n_comp)``.
        component: Output component index (default 0).
        name: Label used in weights dicts and plots.
        sampling_method: Sampling method (``'uniform'``, ``'lhs'``, …).
        sampling_transform: Optional inverse-CDF transform for sampling.
    """
    boundary:           tuple
    value:              Union[float, Callable]
    component:          int = 0
    name:               Optional[str] = None
    sampling_method:    Union[str, Callable] = 'uniform'
    sampling_transform: Optional[Callable] = None
    bc_type:            str = 'initial_condition'

    def get_value(self, x) -> 'torch.Tensor':
        """Evaluate the IC value at points *x* (numpy array or Tensor)."""
        if callable(self.value):
            return _call_value_function(self.value, x)
        import torch as _torch
        device = x.device if isinstance(x, _torch.Tensor) else 'cpu'
        dtype  = x.dtype  if isinstance(x, _torch.Tensor) else _torch.float32
        return _torch.full((x.shape[0],), float(self.value),
                           device=device, dtype=dtype)


@dataclass
class CubicCustomBC:
    """
    Custom boundary condition for :class:`~pinns.domain.DomainCubic` domains
    with a user-defined residual callable.

    Instead of a fixed ``value`` + ``component`` target, you supply a
    **residual function** ``f`` with the same signature as the PDE residual::

        f(x, y, params_dict, derivative) -> residual  # (n,) or tuple of (n,)
        f(x, y, params_dict)             -> residual
        f(x, y)                          -> residual

    The trainer minimises ``mean(residual²)``.  When ``f`` returns a tuple the
    loss is split into one sub-loss per element, named
    ``<name>_0``, ``<name>_1``, … (unless ``output_names`` is provided).

    Created by :meth:`~pinns.domain.DomainCubic.add_custom`.

    Args:
        boundary: Face specification — a tuple where each element is
                  ``0`` (lower face), ``1`` (upper face), or ``None``
                  (free in that dimension).  Use
                  :meth:`~pinns.domain.DomainCubic._parse_boundary_str` to
                  convert string labels such as ``'xmin'``.
        f: Residual callable.  Signature may be 2-, 3-, or 4-argument
           (see above).
        name: Base label used in weight dicts and plots.
        output_names: Optional per-output labels (when ``f`` returns a tuple).
                      If omitted, names are auto-generated as ``name_0``,
                      ``name_1``, …
        subspace: List of ``(lo, hi)`` tuples — one per **free** spatial
                  dimension of the boundary face — restricting sampling to
                  that sub-range on each free spatial axis.
        time_subspace: ``(t_lo, t_hi)`` restricting sampling to a time
                       sub-interval.  Only valid for time-dependent domains.
        sampling_method: Sampling strategy (``'uniform'``, ``'lhs'``, …).
        sampling_transform: Optional inverse-CDF transform for sampling.
    """
    boundary:           tuple
    f:                  Callable
    name:               Optional[str] = None
    output_names:       Optional[List[str]] = None
    subspace:           Optional[List] = None
    time_subspace:      Optional[tuple] = None
    sampling_method:    Union[str, Callable] = 'uniform'
    sampling_transform: Optional[Callable] = None
    bc_type:            str = 'cubic_custom'


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

