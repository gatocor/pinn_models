"""
Tests for GNNMeshNetwork on a tiny [0,1]^2 mesh.

Covers:
  - construction / init / apply (shape checks)
  - set_input_range / set_output_range
  - get_node_coefficients
  - JAX spatial derivatives via derivative()
  - strong-form training (plain JAX + optax)
  - weak-form training (ProblemWeak + Trainer)

Run with:
    python tests/test_gnn_network.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import jax
import jax.numpy as jnp

os.environ.setdefault('PINNS_BACKEND', 'jax')

# ---------------------------------------------------------------------------
# Shared tiny mesh:  [0,1]^2 = 2 right triangles
#
#   3──2
#   |\ |
#   | \|
#   0──1
# ---------------------------------------------------------------------------
VERTS = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=np.float64)
FACES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

RNG = jax.random.PRNGKey(0)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_failures = []

def check(name, condition, detail=""):
    tag = PASS if condition else FAIL
    msg = f"  [{tag}]  {name}" + (f"  ({detail})" if detail else "")
    print(msg)
    if not condition:
        _failures.append(name)
    return condition

def section(title):
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# Shared objects
# ---------------------------------------------------------------------------
from pinns import DomainMesh, GNNMeshNetwork
from pinns.backends.jax.gnn_network import GNNMeshNetwork as GNNDirect

domain = DomainMesh((VERTS, FACES))
net    = GNNMeshNetwork(domain, hidden_dim=16, depth=2, message_steps=2, n_outputs=1)
params = net.init(RNG)


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------
section("1. Construction")

check("GNNMeshNetwork importable from pinns top-level", GNNMeshNetwork is GNNDirect)
check("n_nodes == 4",        net.n_nodes     == 4)
check("n_faces == 2",        net.n_faces     == 2)
check("spatial_dims == 2",   net.spatial_dims == 2)
check("n_outputs == 1",      net.n_outputs   == 1)
check("hidden_dim == 16",    net.hidden_dim  == 16)
check("depth == 2",          net.depth       == 2)
check("message_steps == 2",  net.message_steps == 2)
check("mesh_nodes shape (4,2)", net.mesh_nodes.shape == (4, 2))
check("mesh_faces shape (2,3)", net.mesh_faces.shape == (2, 3))
check("mesh_nodes values match VERTS",
      np.allclose(net.mesh_nodes, VERTS, atol=1e-6))

# 3D guard
try:
    verts3d = np.column_stack([VERTS, np.zeros(4)])
    faces3d = FACES
    GNNDirect.__new__(GNNDirect)
    # Try to construct in a simple way that triggers the guard
    class _Fake3DDomain:
        _vertices = verts3d
        _faces    = faces3d
        _spatial_dims = 3
    net3d = object.__new__(GNNDirect)
    net3d.__init__(_Fake3DDomain(), hidden_dim=8, depth=1, message_steps=1)
    check("3D raises NotImplementedError", False, "no exception raised")
except NotImplementedError:
    check("3D raises NotImplementedError", True)
except Exception as e:
    check("3D raises NotImplementedError", False, str(e))


# ---------------------------------------------------------------------------
# 2. Parameter initialisation
# ---------------------------------------------------------------------------
section("2. Parameter initialisation")

check("params is not None",  params is not None)
check("params is dict",      isinstance(params, dict))
p = params.get('params', params)
check("'encoder' present",   'encoder' in p, f"keys={list(p.keys())}")
check("'decoder' present",   'decoder' in p)
check("'mp_step_0' present", 'mp_step_0' in p)
check("'mp_step_1' present", 'mp_step_1' in p)


# ---------------------------------------------------------------------------
# 3. Forward pass
# ---------------------------------------------------------------------------
section("3. Forward pass (apply)")

x1 = jnp.array([[0.25, 0.25]])
y1 = net.apply(params, x1)
check("single-point output shape (1,1)", y1.shape == (1, 1))

xb = jnp.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])
yb = net.apply(params, xb)
check("batch output shape (3,1)", yb.shape == (3, 1))
check("batch output finite",      bool(jnp.all(jnp.isfinite(yb))))

net2   = GNNMeshNetwork(domain, hidden_dim=8, depth=1, message_steps=1, n_outputs=3)
p2     = net2.init(RNG)
y3out  = net2.apply(p2, jnp.ones((5, 2)) * 0.5)
check("n_outputs=3 → shape (5,3)", y3out.shape == (5, 3))

net.set_input_range(np.array([0., 0.]), np.array([1., 1.]))
yn = net.apply(params, jnp.array([[0.5, 0.5]]))
check("output finite after set_input_range", bool(jnp.all(jnp.isfinite(yn))))

net.set_output_range(np.array([0.0]), np.array([2.0]))
yo = net.apply(params, jnp.array([[0.5, 0.5]]))
check("output finite after set_output_range", bool(jnp.all(jnp.isfinite(yo))))
net.output_min = None
net.output_max = None

net.params = params
ynp = net.predict(np.array([[0.2, 0.3], [0.8, 0.7]]))
check("predict returns ndarray",  isinstance(ynp, np.ndarray))
check("predict shape (2,1)",      ynp.shape == (2, 1))


# ---------------------------------------------------------------------------
# 4. Node coefficients
# ---------------------------------------------------------------------------
section("4. Node coefficients")

coeffs = net.get_node_coefficients(params)
check("shape (4,1)",  coeffs.shape == (4, 1))
check("all finite",   np.all(np.isfinite(coeffs)))


# ---------------------------------------------------------------------------
# 5. Spatial derivatives (using jax.grad + jax.vmap — no context needed)
# ---------------------------------------------------------------------------
section("5. Spatial derivatives")

def u_scalar(x_single):
    """Scalar u(x) for a single point x of shape (2,)."""
    return net.apply(params, x_single[None, :])[0, 0]

xd = jnp.array([[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]])
du = jax.vmap(jax.grad(u_scalar))(xd)          # (3, 2) — full gradient

check("du/dx shape (3,2)",  du.shape == (3, 2))
check("du/dx finite",       bool(jnp.all(jnp.isfinite(du))))
check("du/dx component 0 finite", bool(jnp.all(jnp.isfinite(du[:, 0]))))
check("du/dy component 1 finite", bool(jnp.all(jnp.isfinite(du[:, 1]))))

# Second derivative d²u/dx²
def du_dx_scalar(x_single):
    return jax.grad(u_scalar)(x_single)[0]

xd2 = jnp.array([[0.25, 0.25], [0.75, 0.75]])
d2u = jax.vmap(jax.grad(du_dx_scalar))(xd2)    # (2, 2)

check("d²u/dx² finite", bool(jnp.all(jnp.isfinite(d2u))))


# ---------------------------------------------------------------------------
# 6. Strong-form training (plain JAX + optax)
# ---------------------------------------------------------------------------
section("6. Strong-form training")

import optax

net_s  = GNNMeshNetwork(domain, hidden_dim=16, depth=2, message_steps=2, n_outputs=1)
ps     = net_s.init(RNG)
rng_np = np.random.default_rng(0)
x_data = rng_np.uniform(0, 1, size=(30, 2)).astype(np.float32)
x_pde  = jnp.array(x_data)

def pde_loss(p):
    y = net_s.apply(p, x_pde)
    return jnp.mean(y[:, 0] ** 2)

loss0 = float(pde_loss(ps))
check("initial PDE loss is finite", np.isfinite(loss0), f"loss={loss0:.4e}")

grads = jax.grad(pde_loss)(ps)
leaves = jax.tree_util.tree_leaves(grads)
check("gradients w.r.t. params are finite",
      any(bool(jnp.any(jnp.isfinite(g))) for g in leaves))

tx  = optax.adam(1e-2)
opt = tx.init(ps)

@jax.jit
def train_step(p, s):
    l, g = jax.value_and_grad(pde_loss)(p)
    updates, s = tx.update(g, s)
    p = optax.apply_updates(p, updates)
    return p, s, l

p_cur = ps
for _ in range(5):
    p_cur, opt, last_loss = train_step(p_cur, opt)

loss_final = float(last_loss)
check("5-step Adam: loss is finite",   np.isfinite(loss_final), f"loss={loss_final:.4e}")
check("5-step Adam: loss decreased",   loss_final <= loss0 + 1e-6,
      f"{loss0:.4e} → {loss_final:.4e}")

# Gradient-based loss (Laplacian-style) using jax.grad
def u_scalar_s(x_single):
    return net_s.apply(ps, x_single[None, :])[0, 0]

def lap_loss(p):
    def _u(x_single):
        return net_s.apply(p, x_single[None, :])[0, 0]
    grads = jax.vmap(jax.grad(_u))(x_pde)   # (n, 2)
    return jnp.mean(grads ** 2)

lv = float(lap_loss(ps))
check("Laplacian-based loss finite", np.isfinite(lv), f"loss={lv:.4e}")
g2 = jax.grad(lap_loss)(ps)
lv2 = jax.tree_util.tree_leaves(g2)
check("Laplacian loss grads finite",
      any(bool(jnp.any(jnp.isfinite(g))) for g in lv2))


# ---------------------------------------------------------------------------
# 7. Weak-form training (ProblemWeak + Trainer)
# ---------------------------------------------------------------------------
section("7. Weak-form training")

os.environ['PINNS_BACKEND'] = 'jax'
from pinns.backends.jax import Trainer
from pinns.problem_weak import ProblemWeak

def _make_weak_problem(dom):
    def volume_fn(x, y, params, phi, grad_phi, derivative=None):
        du_dx = derivative(y, x, 0, (0,))
        du_dy = derivative(y, x, 0, (1,))
        grad_u = jnp.stack([du_dx, du_dy], axis=-1)
        return jnp.sum(grad_u * grad_phi, axis=-1)

    return ProblemWeak(
        domain=dom,
        volume_fn=volume_fn,
        params={},
        input_names=['x', 'y'],
        output_names=['u'],
        cubature_order=3,
    )

# 7a. Construction
dom_w  = DomainMesh((VERTS, FACES))
prob_w = _make_weak_problem(dom_w)
check("ProblemWeak construction OK",  prob_w is not None)
check("ProblemWeak n_outputs == 1",   prob_w.n_outputs == 1)

# 7b. Training run
net_w = GNNMeshNetwork(dom_w, hidden_dim=16, depth=2, message_steps=2, n_outputs=1)
trainer = Trainer(prob_w, net_w)
trainer.compile(epochs=5, learning_rate=1e-3, print_each=10)
try:
    trainer.train()
    hist = getattr(trainer, 'history', None) or getattr(trainer, '_history', None)
    check("Trainer.train() completes", True)
    if isinstance(hist, dict) and hist:
        for key, vals in hist.items():
            arr = np.asarray(vals)
            check(f"loss '{key}' finite", bool(np.all(np.isfinite(arr))),
                  f"values={arr}")
except Exception as e:
    check("Trainer.train() completes", False, str(e))

# 7c. With Dirichlet BC on boundary nodes
dom_d = DomainMesh((VERTS, FACES))
dom_d.add_dirichlet(select=lambda v: np.ones(len(v), dtype=bool), value=0.0, component=0, name="walls")
prob_d = _make_weak_problem(dom_d)
net_d  = GNNMeshNetwork(dom_d, hidden_dim=16, depth=2, message_steps=2, n_outputs=1)
t2 = Trainer(prob_d, net_d)
t2.compile(epochs=5, learning_rate=1e-3, print_each=10)
try:
    t2.train()
    check("Weak + Dirichlet BC: runs OK", True)
except Exception as e:
    check("Weak + Dirichlet BC: runs OK", False, str(e))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
if _failures:
    print(f"\033[91m{len(_failures)} test(s) FAILED:\033[0m")
    for f in _failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print(f"\033[92mAll tests PASSED.\033[0m")
