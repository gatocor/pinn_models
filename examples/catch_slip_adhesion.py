"""
Catch-Slip Adhesion Model

Coupled ODE system for bond density phi(t) and bond extension u(t):

    dphi/dt = k_on * (phi_m - phi) - k_off(F) * phi
    du/dt   = v_0 - k_off(F) * u

with catch-slip off-rate:

    k_off(F) = k_c * exp(-F/F_c) + k_s * exp(F/F_s)

and force-extension relation  F = kappa * u.

Initial conditions: phi(0) = 0, u(0) = 0.
"""

import numpy as np
import torch
import pinns

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
k_on  = 1.0    # binding rate
phi_m = 1.0    # maximum bond density
k_c   = 1.0    # catch off-rate constant
F_c   = 10.0   # catch force scale
k_s   = 0.5    # slip off-rate constant
F_s   = 5.0    # slip force scale
kappa = 1.0    # spring stiffness
v_0   = 2.0    # pulling velocity

phi0  = 0.0    # initial bond density
u0    = 0.0    # initial extension
t_max = 20.0   # simulation time

# ---------------------------------------------------------------------------
# Estimate steady-state extension for output normalisation
# ---------------------------------------------------------------------------
u_ss = v_0 / (k_c + k_s)
for _ in range(50):
    F_ss   = kappa * u_ss
    koff   = k_c * np.exp(-F_ss / F_c) + k_s * np.exp(F_ss / F_s)
    u_ss   = v_0 / koff
F_ss    = kappa * u_ss
koff_ss = k_c * np.exp(-F_ss / F_c) + k_s * np.exp(F_ss / F_s)
phi_ss  = k_on * phi_m / (k_on + koff_ss)
u_max   = 2.0 * u_ss  # output range margin

print(f"Steady state:  u* = {u_ss:.4f},  phi* = {phi_ss:.4f},  F* = {F_ss:.4f}")

# ---------------------------------------------------------------------------
# Domain  [0, t_max]
# ---------------------------------------------------------------------------
domain = pinns.DomainCubicPartition(
    [np.linspace(0.0, t_max, 12)],
    overlap=0.5,
)

domain.add_dirichlet(boundary=(0,), value=phi0, component=0, name="initial_phi")
domain.add_dirichlet(boundary=(0,), value=u0,   component=1, name="initial_u")

# ---------------------------------------------------------------------------
# PDE residuals
# ---------------------------------------------------------------------------
def model_catchslip(X, U, params):
    p = params["fixed"]

    phi = U[:, 0:1]
    u   = U[:, 1:2]

    F     = p["kappa"] * u
    k_off = p["k_c"] * torch.exp(-F / p["F_c"]) + p["k_s"] * torch.exp(F / p["F_s"])

    phi_t = pinns.derivative(phi, X, 0, (0,))
    u_t   = pinns.derivative(u,   X, 0, (0,))

    res_phi = phi_t - p["k_on"] * (p["phi_m"] - phi) + k_off * phi
    res_u   = u_t   - p["v_0"]                        + k_off * u

    return (res_phi, res_u)

# ---------------------------------------------------------------------------
# Reference solution (numpy RK4)
# ---------------------------------------------------------------------------
def _rk4(ode, y0, t):
    """Fixed-step RK4 integrator. Returns solution array (n_steps, n_vars)."""
    y = np.empty((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t) - 1):
        h  = t[i + 1] - t[i]
        k1 = np.array(ode(t[i],           y[i]))
        k2 = np.array(ode(t[i] + h / 2,   y[i] + h / 2 * k1))
        k3 = np.array(ode(t[i] + h / 2,   y[i] + h / 2 * k2))
        k4 = np.array(ode(t[i] + h,        y[i] + h * k3))
        y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def model_catchslip_reference(X, params):
    p = params["fixed"]

    def ode(t, y):
        phi, u = y
        F     = p["kappa"] * u
        k_off = p["k_c"] * np.exp(-F / p["F_c"]) + p["k_s"] * np.exp(F / p["F_s"])
        return [
            p["k_on"] * (p["phi_m"] - phi) - k_off * phi,
            p["v_0"] - k_off * u,
        ]

    t_query = X[:, 0].flatten()
    t_dense = np.linspace(0.0, t_query.max(), 4000)
    sol     = _rk4(ode, [p["phi0"], p["u0"]], t_dense)

    phi_ref = np.interp(t_query, t_dense, sol[:, 0])
    u_ref   = np.interp(t_query, t_dense, sol[:, 1])

    return np.column_stack([phi_ref, u_ref])

# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------
problem = pinns.Problem(
    domain=domain,
    pde_fn=model_catchslip,
    input_names=["t"],
    output_names=["phi", "u"],
    params={
        "k_on":  k_on,
        "phi_m": phi_m,
        "k_c":   k_c,
        "F_c":   F_c,
        "k_s":   k_s,
        "F_s":   F_s,
        "kappa": kappa,
        "v_0":   v_0,
        "phi0":  phi0,
        "u0":    u0,
    },
    output_range=[(0.0, phi_m), (0.0, u_max)],
    solution=model_catchslip_reference,
)

# ---------------------------------------------------------------------------
# Network (FBPINN)
# ---------------------------------------------------------------------------
base_network = pinns.FNN([1, 32, 32, 2], activation="tanh")

network = pinns.FBPINN(
    domain,
    base_network,
    normalize_input=True,
    unnormalize_output=True,
)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
trainer = pinns.Trainer(problem, network)

trainer.compile(
    train_samples={
        "pde":         1000,
        "initial_phi": 1,
        "initial_u":   1,
    },
    test_samples={
        "pde":         1000,
        "initial_phi": 0,
        "initial_u":   0,
    },
    weights={
        "pde":         1.0,
        "initial_phi": 10.0,
        "initial_u":   10.0,
    },
    optimizer="adam",
    learning_rate=1e-3,
    epochs=15000,
    print_each=500,
    show_plots=True,
)

trainer.train()
