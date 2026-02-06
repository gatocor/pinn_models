import torch
from typing import Callable, List, Dict, Any, Optional, Union
from dataclasses import dataclass, field

from .boundary import DirichletBC, NeumannBC, RobinBC, PointsetBC
from .domain import DomainCubic, DomainCubicPartition


@dataclass
class Problem:
    """
    A forward problem for Physics-Informed Neural Networks.
    
    Combines a domain with boundary conditions, physics residual function, and parameters.
    
    Boundary conditions are now added directly to the domain using methods like
    domain.add_dirichlet(), domain.add_neumann(), etc.
    
    Args:
        domain (DomainCubic or DomainCubicPartition): The computational domain with boundary 
                          conditions already added.
        pde_fn (Callable): The PDE residual function with signature:
                          pde_fn(x, y, params) -> residual
                          - x: Input tensor of shape (batch_size, n_dims)
                          - y: Network output tensor of shape (batch_size, n_outputs)
                          - params: Dictionary with keys:
                              - "fixed": User-provided fixed parameters (constants, coefficients)
                              - "infer": Parameters to be inferred (for inverse problems, future)
                              - "internal": Training state (global_step, step) for curriculum learning
                          Returns: Residual tensor of shape (batch_size,) or (batch_size, n_eqs)
        params (dict): Dictionary of fixed problem parameters (constants, coefficients, etc.)
                      These are passed as params["fixed"] to the PDE function.
        input_names (list): Names for input dimensions. Must match domain n_dims.
        output_names (list): Names for output components. Length determines n_outputs.
        output_range (list, optional): Output range for unnormalization, per output component.
                          Each element is (ymin, ymax) tuple or None for that output.
                          Example: [(0, 1), None, (-1, 1)] for 3 outputs.
                          If a single tuple is provided, it applies to all outputs.
                          Default: None (no unnormalization)
        solution (Callable, optional): The analytical/reference solution function with signature:
                          solution(x, params) -> y
                          - x: Input array/tensor of shape (batch_size, n_dims)
                          - params: Same dictionary structure as pde_fn
                          Returns: Solution array/tensor of shape (batch_size, n_outputs)
                          If provided, the error between predicted and true solution
                          will be computed during training and shown in plots.
        
    Example:
        # Define domain with boundary conditions
        domain = DomainCubic([0, 0], [1, 1])
        domain.add_dirichlet((0, None), value=0.0, component=0, name="left")   # u=0 at x_min
        domain.add_dirichlet((1, None), value=1.0, component=0, name="right")  # u=1 at x_max
        
        # Define the heat equation with curriculum learning
        def heat_equation(x, y, params):
            alpha = params["fixed"]['alpha']
            step = params["internal"]['global_step']
            
            # Curriculum: gradually increase weight on higher-order terms
            curriculum_weight = min(1.0, step / 10000)
            
            u_t = pinns.derivative(y, x, 0, (0,))
            u_xx = pinns.derivative(y, x, 0, (1, 1))
            return u_t - alpha * u_xx * curriculum_weight
        
        # Create problem
        problem = Problem(
            domain=domain,
            pde_fn=heat_equation,
            params={'alpha': 0.01},  # Goes to params["fixed"]
            input_names=['x', 't'],
            output_names=['u']
        )
    """
    domain: Union[DomainCubic, DomainCubicPartition]
    pde_fn: Callable[[torch.Tensor, torch.Tensor, Dict], torch.Tensor]
    params: Dict[str, Any] = field(default_factory=dict)
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    output_range: Optional[Union[tuple, List[Optional[tuple]]]] = None
    solution: Optional[Callable] = None
    
    def __post_init__(self):
        # Get n_dims from domain
        self.n_dims = self.domain.n_dims
        
        # Derive n_outputs from output_names
        self.n_outputs = len(self.output_names)
        
        # Validate input_names
        if not self.input_names:
            raise ValueError("input_names is required and cannot be empty")
        if len(self.input_names) != self.n_dims:
            raise ValueError(f"input_names has {len(self.input_names)} elements but domain has {self.n_dims} dimensions")
        
        # Validate output_names
        if not self.output_names:
            raise ValueError("output_names is required and cannot be empty")
        
        # Normalize output_range to list format and validate
        if self.output_range is not None:
            if isinstance(self.output_range, tuple) and len(self.output_range) == 2:
                # Single tuple - apply to all outputs
                if not isinstance(self.output_range[0], (list, tuple)):
                    self.output_range = [self.output_range] * self.n_outputs
            
            # Validate length matches n_outputs
            if isinstance(self.output_range, list):
                if len(self.output_range) != self.n_outputs:
                    raise ValueError(
                        f"output_range has {len(self.output_range)} elements but "
                        f"n_outputs is {self.n_outputs}. They must match."
                    )
    
    @property
    def xmin(self):
        """Get domain lower bounds."""
        return self.domain.xmin
    
    @property
    def xmax(self):
        """Get domain upper bounds."""
        return self.domain.xmax
    
    @property
    def boundary_conditions(self):
        """Get boundary conditions from the domain."""
        return self.domain.boundary_conditions
    
    def compute_pde_residual(self, x: torch.Tensor, y: torch.Tensor, 
                             internal: Dict[str, Any] = None) -> torch.Tensor:
        """
        Compute the PDE residual at given points.
        
        Args:
            x: Input points tensor of shape (batch_size, n_dims)
            y: Network output tensor of shape (batch_size, n_outputs)
            internal: Internal training state dict with keys like:
                     - 'global_step': Total training steps across all compile() calls
                     - 'step': Training step within current compile() call
                     Useful for curriculum learning.
            
        Returns:
            Residual tensor of shape (batch_size,) or (batch_size, n_eqs)
        """
        # Build structured params dict
        params = {
            "fixed": self.params,
            "infer": {},  # Reserved for future inverse problem support
            "internal": internal if internal is not None else {'global_step': 0, 'step': 0}
        }
        return self.pde_fn(x, y, params)
    
    def compute_bc_residual(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute boundary condition residuals at given boundary points.
        
        Args:
            x: Boundary points tensor of shape (batch_size, n_dims)
            y: Network output tensor of shape (batch_size, n_outputs)
            
        Returns:
            Dictionary mapping BC type to residual tensors
        """
        residuals = {}
        
        for i, bc in enumerate(self.boundary_conditions):
            key = f"bc_{i}_{type(bc).__name__}"
            
            if isinstance(bc, DirichletBC):
                # u - value = 0
                target = bc.get_value(x)
                residuals[key] = y[:, bc.component] - target
                
            elif isinstance(bc, NeumannBC):
                # du/dn - value = 0
                normal_dim = bc.get_normal_dimension()
                u = y[:, bc.component:bc.component+1]
                grads = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                           create_graph=True)[0]
                du_dn = grads[:, normal_dim]
                
                # Adjust sign based on boundary side
                sign = bc.get_normal_sign()
                target = bc.get_value(x)
                residuals[key] = sign * du_dn - target
                
            elif isinstance(bc, RobinBC):
                # alpha * u + beta * du/dn - gamma = 0
                normal_dim = bc.get_normal_dimension()
                u = y[:, bc.component:bc.component+1]
                grads = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                           create_graph=True)[0]
                du_dn = grads[:, normal_dim]
                
                sign = bc.get_normal_sign()
                alpha, beta, gamma = bc.get_coefficients(x)
                residuals[key] = alpha * y[:, bc.component] + beta * sign * du_dn - gamma
                
            elif isinstance(bc, PointsetBC):
                # u - target_values = 0
                residuals[key] = y[:, bc.component] - bc.values
        
        return residuals
    
    def get_dirichlet_conditions(self) -> List[DirichletBC]:
        """Get all Dirichlet boundary conditions."""
        return [bc for bc in self.boundary_conditions if isinstance(bc, DirichletBC)]
    
    def get_neumann_conditions(self) -> List[NeumannBC]:
        """Get all Neumann boundary conditions."""
        return [bc for bc in self.boundary_conditions if isinstance(bc, NeumannBC)]
    
    def get_robin_conditions(self) -> List[RobinBC]:
        """Get all Robin boundary conditions."""
        return [bc for bc in self.boundary_conditions if isinstance(bc, RobinBC)]
    
    def get_pointset_conditions(self) -> List[PointsetBC]:
        """Get all Pointset boundary conditions."""
        return [bc for bc in self.boundary_conditions if isinstance(bc, PointsetBC)]
    
    def update_params(self, **kwargs):
        """Update problem parameters."""
        self.params.update(kwargs)
    
    def __repr__(self):
        n_bcs = len(self.boundary_conditions)
        return (
            f"Problem(domain={self.domain}, n_dims={self.n_dims}, n_outputs={self.n_outputs}, "
            f"n_boundary_conditions={n_bcs}, params={list(self.params.keys())})"
        )
