import numpy as np
import config as cfg
import equations as eqs
from initial_conditions import setup_initial_conditions
from equations import f1, f2
from solver import solve_system
from auxiliary import galerkin_approx  # Direct import for convenience

# Prepare initial data and source terms
data = setup_initial_conditions()

# Run the solver
tild_u, tild_v, cond_u, cond_v = solve_system(data, f1, f2)

""" Approximation of unknown functions at each temporal layer """

def approx_u(N: int, k: int, ell: float, x: np.ndarray) -> np.ndarray:
    """ Evaluate Galerkin approximation of u at timestep k (k ≥ 2) """
    if k < 2:
        raise ValueError("Temporal index k must be ≥ 2")
    return galerkin_approx(N, ell, tild_u[k - 2], x)

def approx_v(N: int, k: int, ell: float, x: np.ndarray) -> np.ndarray:
    """ Evaluate Galerkin approximation of v at timestep k (k ≥ 2) """
    if k < 2:
        raise ValueError("Temporal index k must be ≥ 2")
    return galerkin_approx(N, ell, tild_v[k - 2], x)

print(np.abs(eqs.u(np.linspace(0, cfg.ell, 5), cfg.t[2])
      - approx_u(cfg.N, 2, cfg.ell, np.linspace(0, cfg.ell, 5))))