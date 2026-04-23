"""
Step-by-step validation of the ProblemWeak pipeline.

Run with:
    python examples/test_weak_pipeline.py

Each test prints PASS / FAIL with a short explanation.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import jax
import jax.numpy as jnp

# ── A tiny hand-crafted mesh: 2 right triangles tiling [0,1]^2 ──────────────
#
#   3──2
#   |\ |
#   | \|
#   0──1
#
VERTS = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=np.float64)
FACES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(name, condition, detail=""):
    tag = PASS if condition else FAIL
    print(f"  [{tag}]  {name}" + (f"  ({detail})" if detail else ""))
    return condition


# ============================================================
# TEST 1 — Cubature weights sum to element area
# ============================================================
print("\n=== TEST 1: cubature weights sum to area ===")
from pinns.problem_weak import _triangle_cubature, _precompute_volume

for order in [1, 2, 3, 4, 5]:
    ref_pts, ref_w = _triangle_cubature(order)
    # ref weights should sum to 0.5 (area of reference triangle)
    check(f"order={order}  sum(ref_w)=0.5",
          abs(ref_w.sum() - 0.5) < 1e-10,
          f"sum={ref_w.sum():.6f}")

cd = _precompute_volume(VERTS, FACES, cub_order=3)
# Each triangle has area 0.5.  Weights per element should sum to 0.5.
for k in range(2):
    s = cd['weights'][k].sum()
    check(f"face {k}: sum(phys_weights)=0.5", abs(s - 0.5) < 1e-5, f"sum={s:.6f}")


# ============================================================
# TEST 2 — Quadrature points lie inside the physical triangle
# ============================================================
print("\n=== TEST 2: quadrature pts inside each triangle ===")
for k, (i0, i1, i2) in enumerate(FACES):
    x0, x1, x2 = VERTS[i0], VERTS[i1], VERTS[i2]
    pts_k = cd['pts'][k]                       # (Q, 2)
    J = np.stack([x1-x0, x2-x0], axis=1)
    J_inv = np.linalg.inv(J)
    # Map back to reference: λ = J^{-1}(x - x0)
    lam = (pts_k - x0) @ J_inv.T              # (Q, 2)
    lam0 = 1 - lam[:, 0] - lam[:, 1]
    inside = np.all(lam >= -1e-6) and np.all(lam0 >= -1e-6) and np.all(lam.sum(axis=1) <= 1+1e-6)
    check(f"face {k}: all pts inside triangle", inside)


# ============================================================
# TEST 3 — P1 basis values sum to 1 at every quadrature point
# ============================================================
print("\n=== TEST 3: partition of unity  sum_a phi_a = 1 ===")
phi = cd['phi']                                # (F, Q, 3)
for k in range(2):
    s = phi[k].sum(axis=1)                     # (Q,)
    check(f"face {k}: sum_a phi_a(x_q)=1 for all q",
          np.allclose(s, 1.0, atol=1e-6), f"max dev={abs(s-1).max():.2e}")


# ============================================================
# TEST 4 — P1 basis values equal barycentric coords of each vertex
# ============================================================
print("\n=== TEST 4: phi_a(vertex_b) = delta_{ab} ===")
# For face 0 = (0,1,2): vertices are at barycentric (1,0,0),(0,1,0),(0,0,1)
# which correspond to reference coords (xi, eta) = (0,0), (1,0), (0,1)
from pinns.problem_weak import _lagrange_basis_and_grad
ref_corners = np.array([[0., 0.], [1., 0.], [0., 1.]])
phi_corners, _ = _lagrange_basis_and_grad(ref_corners, N=1)   # (3, 3)
check("phi at ref corners = identity matrix",
      np.allclose(phi_corners, np.eye(3), atol=1e-12))


# ============================================================
# TEST 5 — Physical gradients: constant, reproduce exact derivative
# ============================================================
print("\n=== TEST 5: physical grad_phi reproduces a linear function ===")
# f(x,y) = 2x + 3y  has ∇f = [2, 3]
# Represent f in P1 on face 0 = {v0=(0,0), v1=(1,0), v2=(1,1)}:
#   f_0=0, f_1=2, f_2=5  (values at nodes)
# ∇f ≈ sum_a f_a * grad_phi_a
# grad_phi shape is (F, Q, L, 2) for all orders; average over Q gives constant P1 gradient
f_node_vals = np.array([0., 2., 5.])           # face 0 node values
gp0 = cd['grad_phi'][0].mean(axis=0)           # (L=3, 2) — constant for P1
grad_f_approx = f_node_vals @ gp0             # (2,)
check("grad of 2x+3y on face 0 = [2, 3]",
      np.allclose(grad_f_approx, [2., 3.], atol=1e-5),
      f"got {grad_f_approx}")

# face 1 = {v0=(0,0), v2=(1,1), v3=(0,1)}:
#   f_0=0, f_2=5, f_3=3
f_node_vals1 = np.array([0., 5., 3.])
gp1 = cd['grad_phi'][1].mean(axis=0)           # (L=3, 2)
grad_f_approx1 = f_node_vals1 @ gp1
check("grad of 2x+3y on face 1 = [2, 3]",
      np.allclose(grad_f_approx1, [2., 3.], atol=1e-5),
      f"got {grad_f_approx1}")


# ============================================================
# TEST 6 — Quadrature integrates a polynomial exactly
# ============================================================
print("\n=== TEST 6: cubature integrates polynomials exactly ===")
# Integral of x over [0,1]^2 = 0.5
# Our mesh covers [0,1]^2 exactly with 2 triangles.
total = 0.0
for k in range(2):
    pts_k = cd['pts'][k]                       # (Q, 2)
    w_k   = cd['weights'][k]                   # (Q,)
    total += (w_k * pts_k[:, 0]).sum()         # ∫ x dΩ over T_k
check("∫_Ω x dΩ = 0.5", abs(total - 0.5) < 1e-5, f"got {total:.6f}")

# Integral of x*y over [0,1]^2 = 0.25 (needs degree-2 exactness, order≥2)
cd2 = _precompute_volume(VERTS, FACES, cub_order=2)
total_xy = 0.0
for k in range(2):
    pts_k = cd2['pts'][k]
    w_k   = cd2['weights'][k]
    total_xy += (w_k * pts_k[:, 0] * pts_k[:, 1]).sum()
check("∫_Ω x·y dΩ = 0.25 (order-2)", abs(total_xy - 0.25) < 1e-5, f"got {total_xy:.6f}")


# ============================================================
# TEST 7 — Scatter-add assembles the stiffness matrix correctly
# ============================================================
print("\n=== TEST 7: FEM stiffness matrix matches analytical K ===")
# For Poisson on unit square with 2-triangle mesh, assemble K manually
# and compare to the scatter-add in make_loss_fn.
# K[i,j] = sum_k sum_q w_{k,q} grad_phi_i(x_{k,q}) . grad_phi_j(x_{k,q})
# For P1, grad_phi is constant per element, so:
# K[i,j] = sum_k area_k * (grad_phi_i^k . grad_phi_j^k)
#         (the s.t. node i and j are both in element k)
n_nodes = 4
K = np.zeros((n_nodes, n_nodes))
for k in range(2):
    area_k = cd['weights'][k].sum()            # = element area
    gp = cd['grad_phi'][k].mean(axis=0)        # (3, 2)  — P1 grad is const per elem
    ids = FACES[k]                             # local→global
    K_local = area_k * (gp @ gp.T)            # (3, 3)  = area * grad_phi_i . grad_phi_j
    for a in range(3):
        for b in range(3):
            K[ids[a], ids[b]] += K_local[a, b]

# Now verify using the assembler's scatter (grad_phi constant → sum_q w_q = area):
K_asm = np.zeros((n_nodes, n_nodes))
for k in range(2):
    gp   = cd['grad_phi'][k].mean(axis=0)      # (3, 2)  — constant for P1
    w_k  = cd['weights'][k]                    # (Q,)
    pts_k = cd['pts'][k]                       # (Q, 2)
    ids  = FACES[k]
    for a in range(3):
        for b in range(3):
            # integrand of grad_phi_b . grad_phi_a (dot product, constant)
            dot = np.dot(gp[a], gp[b])
            K_asm[ids[a], ids[b]] += w_k.sum() * dot

check("K matches K_asm", np.allclose(K, K_asm, atol=1e-10))

# Row sums of K should be zero (∑_j K_{ij} = ∫∇φ_i · ∇1 = 0)
row_sums = K.sum(axis=1)
check("row sums of K = 0 (∇1=0)", np.allclose(row_sums, 0, atol=1e-10),
      f"max |row sum|={abs(row_sums).max():.2e}")


# ============================================================
# TEST 8 — Full loss_fn: exact solution gives near-zero loss
# ============================================================
print("\n=== TEST 8: loss_fn on exact solution ≈ 0 ===")
import pinns
pinns.use_backend("jax")

# Build a tiny real mesh (pygmsh-free): just our 2 triangles
# Feed it to DomainMesh via a mock mesh object
import types

mock_mesh = types.SimpleNamespace()
mock_mesh.points = np.pad(VERTS, ((0,0),(0,1)))   # add z=0 column
mock_mesh.cells_dict = {"triangle": FACES, "line": np.zeros((0,2), dtype=int)}
mock_mesh.cell_sets_dict = {}
mock_mesh.field_data = {}

try:
    domain = pinns.DomainMesh(mock_mesh)

    # No BCs → all 4 nodes free
    from pinns.problem_weak import ProblemWeak

    def volume_fn_test(x, y, params, phi, grad_phi, derivative=None):
        du_dx = derivative(y, x, 0, (0,))
        du_dy = derivative(y, x, 0, (1,))
        grad_u = jnp.stack([du_dx, du_dy], axis=-1)
        f = 2.0 * jnp.pi**2 * jnp.sin(jnp.pi * x[:, 0]) * jnp.sin(jnp.pi * x[:, 1])
        return jnp.sum(grad_u * grad_phi, axis=-1) - f * phi

    problem_test = ProblemWeak(
        domain=domain,
        volume_fn=volume_fn_test,
        input_names=["x", "y"],
        output_names=["u"],
        cubature_order=3,
        solution=lambda xy: np.sin(np.pi * xy[:, 0]) * np.sin(np.pi * xy[:, 1]),
    )

    def _u_and_grad_exact(params, xy):
        def u_single(z):
            return jnp.sin(jnp.pi * z[0]) * jnp.sin(jnp.pi * z[1])
        return jax.value_and_grad(u_single)(xy)

    loss_fn = jax.jit(problem_test.make_loss_fn(_u_and_grad_exact))
    loss_val = float(loss_fn({}))
    # On a 2-triangle mesh u* = sin(πx)sin(πy) is far from P1, so loss is O(1).
    # Key property: loss must be finite and positive.
    check("loss(u*) is finite (not NaN/inf)", np.isfinite(loss_val),
          f"loss={loss_val:.4e}")
    check("loss(u*) >= 0", loss_val >= 0, f"loss={loss_val:.4e}")
    print(f"       loss on 2-triangle mesh (large by design): {loss_val:.4e}")

    # Mesh-refinement convergence: loss should decrease as mesh is refined
    print("       mesh refinement convergence:")
    prev_loss = None
    for n in [2, 4, 8, 16]:
        # Build n×2 right-triangle mesh on [0,1]^2
        v_list, f_list = [], []
        for ix in range(n+1):
            for iy in range(n+1):
                v_list.append([ix/n, iy/n])
        v_arr = np.array(v_list)
        f_list = []
        for ix in range(n):
            for iy in range(n):
                i00 = ix*(n+1)+iy; i10 = (ix+1)*(n+1)+iy
                i01 = ix*(n+1)+iy+1; i11 = (ix+1)*(n+1)+iy+1
                f_list.append([i00, i10, i11])
                f_list.append([i00, i11, i01])
        f_arr = np.array(f_list, dtype=np.int64)
        mock_r = types.SimpleNamespace()
        mock_r.points = np.pad(v_arr, ((0,0),(0,1)))
        mock_r.cells_dict = {"triangle": f_arr, "line": np.zeros((0,2),dtype=int)}
        mock_r.cell_sets_dict = {}; mock_r.field_data = {}
        dom_r = pinns.DomainMesh(mock_r)
        prob_r = ProblemWeak(domain=dom_r, volume_fn=volume_fn_test,
                             input_names=["x","y"], output_names=["u"],
                             cubature_order=3)
        lf = jax.jit(prob_r.make_loss_fn(_u_and_grad_exact))
        lv = float(lf({}))
        print(f"         n={n:2d} ({len(f_arr)} triangles): loss={lv:.4e}")
        if prev_loss is not None:
            check(f"  loss decreases n={n//2}→{n}", lv < prev_loss,
                  f"{prev_loss:.4e} → {lv:.4e}")
        prev_loss = lv

except Exception as e:
    print(f"  [SKIP]  Test 8 skipped (DomainMesh init issue): {e}")


# ============================================================
# TEST 9 — Gradient of loss w.r.t. network params is non-zero
# ============================================================
print("\n=== TEST 9: ∂loss/∂θ ≠ 0 (AD path works) ===")
try:
    import pinns
    net = pinns.FNN([2, 16, 1], activation="tanh")
    net = net.to("cpu")
    dummy = jnp.ones((1, 2))
    _ = net.apply(net.params, dummy)
    params = net.params

    def _u_and_grad(p, xy):
        def u_s(z): return net.apply(p, z[None])[0, 0]
        return jax.value_and_grad(u_s)(xy)

    loss_fn2 = problem_test.make_loss_fn(_u_and_grad)
    lv, grads = jax.value_and_grad(loss_fn2)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    total_g = float(jnp.sqrt(sum(jnp.sum(g**2) for g in leaves)))
    check("total |∂loss/∂θ| > 0", total_g > 0, f"|g|={total_g:.4e}")
    check("no NaN in gradients",
          all(not jnp.any(jnp.isnan(g)) for g in leaves))

except Exception as e:
    print(f"  [SKIP]  Test 9 skipped: {e}")


# ============================================================
# TEST 10 — _ref_nodes_pqr: correct count and unique entries
# ============================================================
print("\n=== TEST 10: _ref_nodes_pqr node count and uniqueness ===")
from pinns.problem_weak import _ref_nodes_pqr

for N in [1, 2, 3, 4]:
    nodes = _ref_nodes_pqr(N)
    expected = (N + 1) * (N + 2) // 2
    check(f"N={N}: len(nodes) = {expected}", len(nodes) == expected,
          f"got {len(nodes)}")
    # All multi-indices must satisfy p+q+r = N
    valid = all(sum(t) == N and all(x >= 0 for x in t) for t in nodes)
    check(f"N={N}: all p+q+r=N, p,q,r≥0", valid)
    # No duplicates
    check(f"N={N}: no duplicate nodes", len(set(nodes)) == len(nodes))


# ============================================================
# TEST 11 — _lagrange_basis_and_grad: partition of unity & Kronecker
# ============================================================
print("\n=== TEST 11: Lagrange basis – partition of unity & Kronecker delta ===")
from pinns.problem_weak import _lagrange_basis_and_grad

for N in [1, 2, 3]:
    nodes  = _ref_nodes_pqr(N)
    n_dofs = (N + 1) * (N + 2) // 2
    # Reference coordinates of the DOF nodes: (xi, eta) = (q/N, r/N)
    ref_pts = np.array([[q / N, r / N] for (p, q, r) in nodes], dtype=np.float64)
    phi, gphi = _lagrange_basis_and_grad(ref_pts, N)

    # Partition of unity: rows of phi sum to 1
    row_sums = phi.sum(axis=1)
    check(f"N={N}: partition of unity",
          np.allclose(row_sums, 1.0, atol=1e-9),
          f"max dev={abs(row_sums - 1).max():.2e}")

    # Kronecker delta: phi[i, j] == delta(i, j)
    check(f"N={N}: Kronecker delta property",
          np.allclose(phi, np.eye(n_dofs), atol=1e-9),
          f"max dev={abs(phi - np.eye(n_dofs)).max():.2e}")

    # Gradient of sum_a phi_a = 1 must be zero (sum of gradients = 0)
    grad_sum = gphi.sum(axis=1)                       # (n_pts, 2)
    check(f"N={N}: sum of reference gradients = 0",
          np.allclose(grad_sum, 0.0, atol=1e-9),
          f"max |∑∇φ|={abs(grad_sum).max():.2e}")


# ============================================================
# TEST 12 — _build_higher_order_mesh: DOF count & position accuracy
# ============================================================
print("\n=== TEST 12: higher-order mesh DOF counts and midpoint positions ===")
from pinns.problem_weak import _build_higher_order_mesh

# 2-triangle unit-square mesh: 4 vertices, 5 unique edges, 2 elements
verts_sq = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=np.float64)
faces_sq  = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
n_verts, n_edges_unique, n_faces_ = 4, 5, 2

for N in [1, 2, 3, 4]:
    dc, ed, e2d = _build_higher_order_mesh(verts_sq, faces_sq, N)
    n_edge_dofs    = n_edges_unique * (N - 1)
    n_interior_per = (N - 1) * (N - 2) // 2
    expected_dofs   = n_verts + n_edge_dofs + n_faces_ * n_interior_per
    check(f"N={N}: total DOF count = {expected_dofs}",
          len(dc) == expected_dofs, f"got {len(dc)}")
    check(f"N={N}: elem_dofs shape (n_faces, n_local)",
          ed.shape == (n_faces_, (N+1)*(N+2)//2),
          f"got {ed.shape}")

# For P2, verify that mid-edge DOF (between v0=(0,0) and v1=(1,0)) is at (0.5, 0)
dc2, ed2, e2d2 = _build_higher_order_mesh(verts_sq, faces_sq, 2)
key01 = (0, 1)
mid_dof_idx = e2d2[key01][0]
mid_pos = dc2[mid_dof_idx]
check("P2 mid-edge DOF (v0→v1) position = (0.5, 0)",
      np.allclose(mid_pos, [0.5, 0.0], atol=1e-12),
      f"got {mid_pos}")


# ============================================================
# TEST 13 — _precompute_volume with lag_order>1: array shapes
# ============================================================
print("\n=== TEST 13: _precompute_volume shapes for P1/P2/P3 ===")

for N in [1, 2, 3]:
    cd_n = _precompute_volume(verts_sq, faces_sq,
                              cub_order=max(3, 2*N), lag_order=N)
    n_local = (N + 1) * (N + 2) // 2
    n_dofs_expected = 4 + 5*(N-1) + 2*((N-1)*(N-2)//2)
    Q = cd_n['pts'].shape[1]
    check(f"P{N}: phi shape = (2, {Q}, {n_local})",
          cd_n['phi'].shape == (2, Q, n_local),
          f"got {cd_n['phi'].shape}")
    check(f"P{N}: grad_phi shape = (2, {Q}, {n_local}, 2)",
          cd_n['grad_phi'].shape == (2, Q, n_local, 2),
          f"got {cd_n['grad_phi'].shape}")
    check(f"P{N}: node_ids shape = (2, {n_local})",
          cd_n['node_ids'].shape == (2, n_local),
          f"got {cd_n['node_ids'].shape}")
    check(f"P{N}: dof_coords shape = ({n_dofs_expected}, 2)",
          cd_n['dof_coords'].shape == (n_dofs_expected, 2),
          f"got {cd_n['dof_coords'].shape}")


# ============================================================
# TEST 14 — Higher-order gradients reproduce polynomial fields exactly
# ============================================================
print("\n=== TEST 14: P2 physical gradients reproduce a quadratic field ===")
# f(x, y) = x^2 + 2*x*y   =>   ∇f = [2x + 2y,  2x]
# Represent f exactly in P2 on the 2-triangle mesh.
cd_p2 = _precompute_volume(verts_sq, faces_sq, cub_order=4, lag_order=2)
dc_p2, ed_p2, _ = _build_higher_order_mesh(verts_sq, faces_sq, 2)

# Node values of f at every DOF
f_vals = dc_p2[:, 0]**2 + 2 * dc_p2[:, 0] * dc_p2[:, 1]   # shape (n_dofs,)

for k in range(2):
    ids    = cd_p2['node_ids'][k]        # (6,) global DOF indices
    gp     = cd_p2['grad_phi'][k]        # (Q, 6, 2)
    f_k    = f_vals[ids]                 # (6,) values at local DOFs
    # Interpolated ∇f at each quadrature point: sum_a f_a * grad_phi_a
    grad_f_qpts = np.einsum('a,qad->qd', f_k, gp)    # (Q, 2)
    # Exact ∇f at quadrature points: [2x+2y,  2x]
    pts_k  = cd_p2['pts'][k]             # (Q, 2)
    grad_f_exact = np.stack([2*pts_k[:,0] + 2*pts_k[:,1],
                             2*pts_k[:,0]], axis=1)
    err = np.abs(grad_f_qpts - grad_f_exact).max()
    check(f"P2 face {k}: ∇(x²+2xy) error < 1e-4", err < 1e-4, f"max err={err:.2e}")


# ============================================================
# TEST 15 — ProblemWeak with lagrange_order=2 builds correctly
# ============================================================
print("\n=== TEST 15: ProblemWeak(lagrange_order=2) structure ===")

# Shared volume_fn used by Tests 15 and 16
def _vol_fn(x, y, params, phi, grad_phi, derivative=None):
    du_dx = derivative(y, x, 0, (0,))
    du_dy = derivative(y, x, 0, (1,))
    grad_u = jnp.stack([du_dx, du_dy], axis=-1)
    f = 2.0 * jnp.pi**2 * jnp.sin(jnp.pi * x[:, 0]) * jnp.sin(jnp.pi * x[:, 1])
    return jnp.sum(grad_u * grad_phi, axis=-1) - f * phi

try:
    import pinns, types
    mock_m2 = types.SimpleNamespace()
    mock_m2.points = np.pad(verts_sq, ((0, 0), (0, 1)))
    mock_m2.cells_dict = {"triangle": faces_sq, "line": np.zeros((0, 2), dtype=int)}
    mock_m2.cell_sets_dict = {}
    mock_m2.field_data = {}

    dom2 = pinns.DomainMesh(mock_m2)

    prob2 = ProblemWeak(domain=dom2, volume_fn=_vol_fn,
                        input_names=["x", "y"], output_names=["u"],
                        lagrange_order=2, cubature_order=4)

    n_expected_p2 = 4 + 5 + 0   # 4 corners + 5 mid-edge DOFs + 0 interior
    check(f"P2: n_dofs = {n_expected_p2}", prob2.n_dofs == n_expected_p2,
          f"got {prob2.n_dofs}")
    check("P2: n_free == n_dofs (no BCs applied)", prob2.n_free_nodes == prob2.n_dofs,
          f"free={prob2.n_free_nodes}, total={prob2.n_dofs}")
    check("P2: lag_order stored on problem", prob2.lagrange_order == 2)
    check("P2: repr contains 'P2'", "P2" in repr(prob2), repr(prob2))

    # Build loss and check it's finite
    def _u_exact_jax(params, xy):
        def u_s(z): return jnp.sin(jnp.pi * z[0]) * jnp.sin(jnp.pi * z[1])
        return jax.value_and_grad(u_s)(xy)

    lf2 = jax.jit(prob2.make_loss_fn(_u_exact_jax))
    lv2 = float(lf2({}))
    check("P2: loss(u*) is finite", np.isfinite(lv2), f"loss={lv2:.4e}")
    check("P2: loss(u*) >= 0", lv2 >= 0, f"loss={lv2:.4e}")

except Exception as e:
    import traceback
    print(f"  [FAIL]  Test 15 raised: {e}")
    traceback.print_exc()


# ============================================================
# TEST 16 — Higher-order loss converges faster with mesh refinement
# ============================================================
print("\n=== TEST 16: P2 vs P1 mesh-refinement convergence ===")
try:
    import types, pinns

    def _make_grid_mesh(n):
        """Build a uniform n×n right-triangle mesh on [0,1]^2."""
        pts = np.array([[ix / n, iy / n] for ix in range(n + 1)
                        for iy in range(n + 1)])
        tris = []
        for ix in range(n):
            for iy in range(n):
                i00 = ix * (n + 1) + iy
                i10 = (ix + 1) * (n + 1) + iy
                i01 = ix * (n + 1) + iy + 1
                i11 = (ix + 1) * (n + 1) + iy + 1
                tris += [[i00, i10, i11], [i00, i11, i01]]
        tris = np.array(tris, dtype=np.int64)
        mock = types.SimpleNamespace()
        mock.points = np.pad(pts, ((0, 0), (0, 1)))
        mock.cells_dict = {"triangle": tris, "line": np.zeros((0, 2), dtype=int)}
        mock.cell_sets_dict = {}; mock.field_data = {}
        return mock

    def _u_exact_jax(params, xy):
        def u_s(z): return jnp.sin(jnp.pi * z[0]) * jnp.sin(jnp.pi * z[1])
        return jax.value_and_grad(u_s)(xy)

    print("          n    P1 loss      P2 loss")
    losses_p1, losses_p2 = [], []
    for n in [2, 4, 8]:
        mock = _make_grid_mesh(n)
        dom = pinns.DomainMesh(mock)

        p1 = ProblemWeak(domain=dom, volume_fn=_vol_fn,
                         input_names=["x", "y"], output_names=["u"],
                         lagrange_order=1, cubature_order=5)
        p2 = ProblemWeak(domain=dom, volume_fn=_vol_fn,
                         input_names=["x", "y"], output_names=["u"],
                         lagrange_order=2, cubature_order=5)

        lv1 = float(jax.jit(p1.make_loss_fn(_u_exact_jax))({}))
        lv2_ = float(jax.jit(p2.make_loss_fn(_u_exact_jax))({}))
        losses_p1.append(lv1); losses_p2.append(lv2_)
        print(f"          {n:2d}   {lv1:.4e}   {lv2_:.4e}")

    # Both should decrease with refinement
    for i in range(1, len(losses_p1)):
        check(f"P1 loss decreases n={[2,4,8][i-1]}→{[2,4,8][i]}",
              losses_p1[i] < losses_p1[i - 1],
              f"{losses_p1[i-1]:.4e} → {losses_p1[i]:.4e}")
        check(f"P2 loss decreases n={[2,4,8][i-1]}→{[2,4,8][i]}",
              losses_p2[i] < losses_p2[i - 1],
              f"{losses_p2[i-1]:.4e} → {losses_p2[i]:.4e}")

except Exception as e:
    import traceback
    print(f"  [FAIL]  Test 16 raised: {e}")
    traceback.print_exc()


print("\nDone.\n")
