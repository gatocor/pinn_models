"""Tests for ModelSpectralSolver — spectral PDE model owning its integrator."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import pinns


# ─────────────────────────────────────────────────────────────────────────── #
#  Fixtures                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

@pytest.fixture
def kdv_model():
    """1-D KdV ModelSpectralSolver (closed-form params, no inference)."""
    eta_val = 1.0
    mu_val  = 0.022
    Nx = 64

    domain = pinns.DomainCubic(
        space=[(-1.0, 1.0)], time=(0.0, 1.0)
    )
    integrator = pinns.IntegratorETD2RK(dt=5e-4)
    model = pinns.ModelSpectralSolver(domain, ["u"], integrator, shape=Nx)

    model.set_linear_op(
        lambda K2, p: {"u": 1j * p["mu2"] * model.k * K2}
    )
    def nonlinear(state_hat, p):
        u = model.inverse(state_hat["u"])
        return {"u": -1j * p["eta"] / 2.0 * model.k * model.forward(u * u)}

    model.set_nonlinear_op(nonlinear)
    model.add_parameter("eta", eta_val)
    model.add_parameter("mu2", mu_val ** 2)
    model.add_initial(jnp.cos(jnp.pi * model.x))
    return model, domain, eta_val, mu_val


@pytest.fixture
def kdv_inv_model():
    """1-D KdV ModelSpectralSolver set up for inverse problem (mu is a free param)."""
    eta_val = 1.0
    mu_init = 0.02
    Nx = 64

    domain = pinns.DomainCubic(
        space=[(-1.0, 1.0)], time=(0.0, 1.0)
    )
    integrator = pinns.IntegratorETD2RK(dt=5e-4, checkpoint=True)
    model = pinns.ModelSpectralSolver(domain, ["u"], integrator, shape=Nx)

    model.set_linear_op(
        lambda K2, p: {"u": 1j * p["mu"] ** 2 * model.k * K2}
    )
    def nonlinear(state_hat, p):
        u = model.inverse(state_hat["u"])
        return {"u": -1j * p["eta"] / 2.0 * model.k * model.forward(u * u)}

    model.set_nonlinear_op(nonlinear)
    model.add_parameter("eta", eta_val)
    model.add_parameter("mu",  mu_init)
    model.add_initial(jnp.cos(jnp.pi * model.x))
    return model, domain, eta_val


# ─────────────────────────────────────────────────────────────────────────── #
#  Construction                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

def test_repr(kdv_model):
    model, *_ = kdv_model
    r = repr(model)
    assert "ModelSpectralSolver" in r
    assert "IntegratorETD2RK" in r


def test_wrong_domain_raises():
    integrator = pinns.IntegratorETD2RK(dt=1e-2)
    with pytest.raises(TypeError, match="DomainCubic"):
        pinns.ModelSpectralSolver(object(), ["u"], integrator, shape=16)


def test_wrong_integrator_raises():
    domain = pinns.DomainCubic(space=[(-1.0, 1.0)], time=(0.0, 1.0))
    with pytest.raises(TypeError, match="Integrator"):
        pinns.ModelSpectralSolver(domain, ["u"], object(), shape=16)


# ─────────────────────────────────────────────────────────────────────────── #
#  add_parameter API                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def test_add_parameter_scalar(kdv_model):
    model, *_ = kdv_model
    assert "eta" in model._params
    assert "mu2" in model._params


def test_add_parameter_chaining(kdv_model):
    model, domain, *_ = kdv_model
    integrator = pinns.IntegratorETD2RK(dt=1e-2)
    m = pinns.ModelSpectralSolver(domain, ["u"], integrator, shape=model.shape[0])
    ret = m.add_parameter("a", 1.0)
    assert ret is m


# ─────────────────────────────────────────────────────────────────────────── #
#  Validate                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def test_validate_missing_linear_op(kdv_model):
    model, domain, *_ = kdv_model
    integrator = pinns.IntegratorETD2RK(dt=1e-2)
    m = pinns.ModelSpectralSolver(domain, ["u"], integrator, shape=model.shape[0])
    m.set_nonlinear_op(lambda s, p: s)
    m.add_initial(jnp.zeros(model.x.shape))
    with pytest.raises(RuntimeError, match="linear_op"):
        m._validate()


def test_validate_missing_initial(kdv_model):
    model, domain, *_ = kdv_model
    integrator = pinns.IntegratorETD2RK(dt=1e-2)
    m = pinns.ModelSpectralSolver(domain, ["u"], integrator, shape=model.shape[0])
    m.set_linear_op(lambda K2, p: {"u": K2})
    m.set_nonlinear_op(lambda s, p: s)
    with pytest.raises(RuntimeError, match="initial"):
        m._validate()


# ─────────────────────────────────────────────────────────────────────────── #
#  _build_params — stop_gradient on non-trainable                            #
# ─────────────────────────────────────────────────────────────────────────── #

def test_build_params_flat(kdv_inv_model):
    model, *_ = kdv_inv_model
    p = model._build_params({"mu": jnp.array(0.033)})
    assert "mu" in p and "eta" in p
    assert float(p["mu"]) == pytest.approx(0.033)


def test_build_params_no_gradient_on_fixed():
    """Gradient must not flow through a frozen parameter."""
    domain = pinns.DomainCubic(space=[(-1.0, 1.0)], time=(0.0, 1.0))
    integrator = pinns.IntegratorETD2RK(dt=1e-2)
    model = pinns.ModelSpectralSolver(domain, ["u"], integrator, shape=16)
    model.add_parameter("a", 2.0)
    model.add_parameter("b", 3.0)

    def f(override):
        p = model._build_params(override)
        return p["a"] * p["b"]

    grad = jax.grad(f)({"a": jnp.array(2.0)})
    # df/da = b = 3 (a is trainable); df/db = 0 because b is frozen
    assert float(grad["a"]) == pytest.approx(3.0)
    assert "b" not in grad


# ─────────────────────────────────────────────────────────────────────────── #
#  solve()                                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

def test_solve_returns_dict(kdv_model):
    model, domain, *_ = kdv_model
    t_obs = np.linspace(0, 1, 20)
    U = model.solve(t_obs=t_obs)
    assert "u" in U
    assert U["u"].shape == (20, model.shape[0])


def test_solve_real_output(kdv_model):
    model, *_ = kdv_model
    t_obs = np.linspace(0, 1, 10)
    U = model.solve(t_obs=t_obs)
    assert np.isfinite(np.array(U["u"])).all()


def test_solve_uses_stored_t_obs(kdv_model):
    model, domain, *_ = kdv_model
    t_obs = np.linspace(0, 0.5, 15)
    model.add_observations(t_obs, {"u": np.zeros((15, model.shape[0]))})
    U = model.solve()
    assert U["u"].shape[0] == 15


# ─────────────────────────────────────────────────────────────────────────── #
#  apply()                                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

def test_apply_shape(kdv_inv_model):
    model, domain, *_ = kdv_inv_model
    t_obs = np.linspace(0, 1, 20)
    model.add_observations(t_obs, {"u": np.zeros((20, model.shape[0]))})

    x_pts = np.random.uniform(-1, 1, 50)
    t_pts = np.random.uniform(0,  1, 50)
    X = np.column_stack([x_pts, t_pts])

    out = model.apply({"mu": jnp.array(0.02)}, X)
    assert out.shape == (50, 1)


def test_apply_is_differentiable(kdv_inv_model):
    model, domain, *_ = kdv_inv_model
    t_obs = np.linspace(0, 0.1, 10)
    model.add_observations(t_obs, {"u": np.zeros((10, model.shape[0]))})

    x_pts = np.linspace(-0.9, 0.9, 20)
    t_pts = np.linspace(0.02, 0.09, 20)
    X = np.column_stack([x_pts, t_pts])

    def loss(params):
        out = model.apply(params, X)
        return jnp.mean(out ** 2)

    grad = jax.grad(loss)({"mu": jnp.array(0.02)})
    assert "mu" in grad
    assert jnp.isfinite(grad["mu"])


def test_apply_without_obs_raises(kdv_inv_model):
    model, *_ = kdv_inv_model
    X = np.zeros((5, 2))
    with pytest.raises(RuntimeError, match="observation times"):
        model.apply({"mu": jnp.array(0.02)}, X)


# ─────────────────────────────────────────────────────────────────────────── #
#  Inverse-problem gradient flow (integration test)                          #
# ─────────────────────────────────────────────────────────────────────────── #

def test_inverse_gradient_nonzero(kdv_inv_model):
    """Loss gradient w.r.t. mu must be nonzero (model is sensitive to mu)."""
    model, domain, eta_val = kdv_inv_model
    t_obs = np.linspace(0, 0.1, 10)   # short window — no aliasing divergence

    # Generate synthetic observations with the true mu
    true_model = pinns.ModelSpectralSolver(domain, ["u"], pinns.IntegratorETD2RK(dt=5e-4), shape=model.shape[0])
    true_model.set_linear_op(
        lambda K2, p: {"u": 1j * p["mu"] ** 2 * true_model.k * K2}
    )
    def nl(state_hat, p):
        u = true_model.inverse(state_hat["u"])
        return {"u": -1j * p["eta"] / 2 * true_model.k * true_model.forward(u * u)}
    true_model.set_nonlinear_op(nl)
    true_model.add_parameter("eta", eta_val)
    true_model.add_parameter("mu", 0.022)
    true_model.add_initial(jnp.cos(jnp.pi * true_model.x))
    U_true = np.array(true_model.solve(t_obs=t_obs)["u"])

    model.add_observations(t_obs, {"u": jnp.array(U_true)})

    def loss(params):
        U_pred = model.apply(params, _obs_to_X(t_obs, np.array(model.x)))
        U_obs  = jnp.array(U_true).reshape(-1)
        return jnp.mean((U_pred[:, 0] - U_obs) ** 2)

    grad = jax.grad(loss)({"mu": jnp.array(0.02)})
    assert float(jnp.abs(grad["mu"])) > 0.0


def _obs_to_X(t_obs, x_grid):
    """Meshgrid (t_obs, x_grid) → (Nt*Nx, 2) [x, t] array."""
    T, X = np.meshgrid(t_obs, x_grid, indexing="ij")
    return np.column_stack([X.ravel(), T.ravel()])
