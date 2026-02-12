"""
2D Laser Ablation (x,y,t) with the SAME STRUCTURE as the 1+1 laser ablation notebook:

- DomainCubicPartition + sampler (NumPy + scipy.stats.ppf)
- input_transform folds x -> |x| and keeps sign(x)
- active_subdomains only for x >= 0
- output_transform:
    h = f_cut + t * h_hat
    v_x = t * sign(x) * tanh(|x|)^2 * v̂_x
    v_y = t * sign(x) * tanh(|x|)^2 * v̂_y
- PDE exactly as specified
"""

import numpy as np
import scipy.stats as sp
import torch
import pinns

k_p   = 1.0
k_d   = 1.0
mu    = 0.1
lam   = 1.0
gamma = 1.0

t_max = 0.1
x_border = 10.0
y_border = 10.0

cut_border_x = 0.10
cut_border_y = 0.04
cut_border_sigma = 0.005

w = 0.2
sigma_cut_sampling = 0.2

h_star = k_p / k_d

def sampler(X, params):
    p = params["fixed"]
    w = p["w"]
    x_border = p["x_border"]
    y_border = p["y_border"]
    sigma_cut_sampling = p["sigma_cut_sampling"]
    t_max = p["t_max"]

    n = int(np.round(X.shape[0] * w))

    x1 = sp.uniform.ppf(X[:n, 0:1], loc=-x_border, scale=2 * x_border)
    x2 = sp.norm.ppf(X[n:, 0:1], scale=sigma_cut_sampling)
    x = np.vstack((x1, x2))

    y1 = sp.uniform.ppf(X[:n, 1:2], loc=-y_border, scale=2 * y_border)
    y2 = sp.norm.ppf(X[n:, 1:2], scale=sigma_cut_sampling)
    y = np.vstack((y1, y2))

    t = t_max * X[:, 2:3]

    return np.hstack((x, y, t))

x_subdomains = [-10, -5.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.25, -0.1, -0.05,
                0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10]
x_subdomains = [i for i in x_subdomains if np.abs(i) <= x_border]
y_subdomains = [-10, -5.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.25, -0.1, -0.05,
                0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10]
y_subdomains = [i for i in y_subdomains if np.abs(i) <= y_border]
t_subdomains = np.linspace(0, t_max, 2)

domain = pinns.DomainCubicPartition(
    [x_subdomains, y_subdomains, t_subdomains],
    sampling_method="uniform",
    sampling_transform=sampler
)

def pde_ablation_2d(X: torch.Tensor, V: torch.Tensor, params: dict):
    p = params["fixed"]
    k_p   = p["k_p"]
    k_d   = p["k_d"]
    mu    = p["mu"]
    lam   = p["lam"]
    gamma = p["gamma"]

    h  = V[:, 0:1]
    vx = V[:, 1:2]
    vy = V[:, 2:3]

    h_t  = pinns.derivative(V, X, component=0, order=(2,))
    h_x  = pinns.derivative(V, X, component=0, order=(0,))
    h_y  = pinns.derivative(V, X, component=0, order=(1,))

    vx_x = pinns.derivative(V, X, component=1, order=(0,))
    vy_y = pinns.derivative(V, X, component=2, order=(1,))

    div_hv = (h_x * vx + h * vx_x) + (h_y * vy + h * vy_y)
    r_h = h_t + div_hv - (k_p - k_d * h)

    vx_xx = pinns.derivative(V, X, component=1, order=(0, 0))
    vx_yy = pinns.derivative(V, X, component=1, order=(1, 1))
    vy_xx = pinns.derivative(V, X, component=2, order=(0, 0))
    vy_yy = pinns.derivative(V, X, component=2, order=(1, 1))

    vy_xy = pinns.derivative(V, X, component=2, order=(0, 1))
    vx_xy = pinns.derivative(V, X, component=1, order=(0, 1))

    r_vx = (mu / 2.0) * (2.0 * vx_xx + vx_yy + vy_xy) + lam * h_x - gamma * vx
    r_vy = (mu / 2.0) * (vx_xy + vy_xx + 2.0 * vy_yy) + lam * h_y - gamma * vy

    return (r_h, r_vx, r_vy)

domain.add_dirichlet((0, None, None), value=0.0, component=1, name="vx_xmin")
domain.add_dirichlet((1, None, None), value=0.0, component=1, name="vx_xmax")
domain.add_dirichlet((None, 0, None), value=0.0, component=1, name="vx_ymin")
domain.add_dirichlet((None, 1, None), value=0.0, component=1, name="vx_ymax")

domain.add_dirichlet((0, None, None), value=0.0, component=2, name="vy_xmin")
domain.add_dirichlet((1, None, None), value=0.0, component=2, name="vy_xmax")
domain.add_dirichlet((None, 0, None), value=0.0, component=2, name="vy_ymin")
domain.add_dirichlet((None, 1, None), value=0.0, component=2, name="vy_ymax")

problem = pinns.Problem(
    domain=domain,
    pde_fn=pde_ablation_2d,
    input_names=["x", "y", "t"],
    output_names=["h", "vx", "vy"],
    params={
        "k_p": k_p,
        "k_d": k_d,
        "mu": mu,
        "lam": lam,
        "gamma": gamma,
        "w": w,
        "x_border": x_border,
        "y_border": y_border,
        "cut_border_x": cut_border_x,
        "cut_border_y": cut_border_y,
        "cut_border_sigma": cut_border_sigma,
        "sigma_cut_sampling": sigma_cut_sampling,
        "t_max": t_max,
    },
)

def input_transform(X: torch.Tensor, params: dict):
    x = X[:, 0:1]
    y = X[:, 1:2]
    t = X[:, 2:3]
    x_abs = torch.abs(x)
    y_abs = torch.abs(y)
    return torch.hstack((x_abs, y_abs, t))

def output_transform(X_in: torch.Tensor, Y: torch.Tensor, params: dict):

    p = params["fixed"]
    x_border = p["cut_border_x"]
    y_border = p["cut_border_y"]
    sigma = p["cut_border_sigma"]

    x = X_in[:, 0:1]
    y = X_in[:, 1:2]
    t = X_in[:, 2:3]

    h_hat  = Y[:, 0:1]
    vx_hat = Y[:, 1:2]
    vy_hat = Y[:, 2:3]

    s1_x = torch.sigmoid((x + x_border) / sigma)  # left edge
    s2_x = torch.sigmoid((x_border - x) / sigma)  # right edge
    inside_x = s1_x * s2_x
    
    s1_y = torch.sigmoid((y + y_border) / sigma)  # bottom edge
    s2_y = torch.sigmoid((y_border - y) / sigma)  # top edge
    inside_y = s1_y * s2_y
    
    f_cut = 1 - inside_x * inside_y

    h = f_cut + t * h_hat

    s_x = torch.sign(x)
    s_y = torch.sign(y)
    sym_x = s_x * torch.tanh(x) ** 2
    sym_y = s_y * torch.tanh(y) ** 2
    vx = t * sym_x * vx_hat
    vy = t * sym_y * vy_hat

    return torch.hstack((h, vx, vy))

baseNetwork = pinns.FNN(
    [3, 64, 3],
    activation="tanh"
)

active_mask = [sd.xmin[0] >= 0.0 and sd.xmin[1] >= 0.0 for sd in domain.subdomains]

network = pinns.FBPINN(
    domain,
    baseNetwork,
    normalize_input=True,
    unnormalize_output=True,
    input_transform=input_transform,
    output_transform=output_transform,
    active_subdomains=active_mask
)

trainer = pinns.Trainer(problem, network)

trainer.compile(
    train_samples={
        "pde": 1000,
        "vx_xmin": 1000, "vx_xmax": 1000, "vx_ymin": 1000, "vx_ymax": 1000,
        "vy_xmin": 1000, "vy_xmax": 1000, "vy_ymin": 1000, "vy_ymax": 1000,
    },
    test_samples={
        "pde": 2000,
        "vx_xmin": 200, "vx_xmax": 200, "vx_ymin": 200, "vx_ymax": 200,
        "vy_xmin": 200, "vy_xmax": 200, "vy_ymin": 200, "vy_ymax": 200,
    },
    weights={
        "pde": 1.0,
        "vx_xmin": 1.0, "vx_xmax": 1.0, "vx_ymin": 1.0, "vx_ymax": 1.0,
        "vy_xmin": 1.0, "vy_xmax": 1.0, "vy_ymin": 1.0, "vy_ymax": 1.0,
    },
    optimizer="adam",
    learning_rate=1e-5,
    epochs=50000,
    batch_size=3000,  # Mini-batch to reduce GPU memory usage
    print_each=100,
    show_plots=True,
    show_subdomains=False,
    show_sampling_points=False,
    # 3D slices: plot x-y plane at different t values
    plot_regions=[
        ((-1.0,1.0), (-1.0,1.0), 0.0),   # t=0.0 (initial)
        (None, None, t_max),   # t=t_max (final)
    ],
    plot_n_points=100,
    profile=False,
)

trainer.train()

