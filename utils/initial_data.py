# =========================
# IMPORT MODULES
# =========================

import numpy as np  # For efficient array handling and numerical operations

# Configuration: time step (tau), domain length (ell), time grid (t), and PDE parameters (a1, a2, alpha, etc.)
import utils.config as cfg

# Symbolic derivatives of displacement functions (u, v and their time/spatial derivatives)
from utils.symbolic_derivatives import SymbolicDerivatives as SD


# =========================
# UTILITY FUNCTION: TAYLOR EXPANSION
# =========================

def taylor_expansion(tau: float, func0, func1, func2):
    """
    Compute a second-order Taylor time approximation:
        f(x, τ) ≈ f(x, 0) + τ·f'(x, 0) + (τ² / 2)·f''(x, 0)

    Parameters:
        tau (float): Time step size (Δt or τ)
        func0 (callable): f(x, 0) — Function value at initial time
        func1 (callable): f'(x, 0) — First time derivative at initial time
        func2 (callable): f''(x, 0) — Second time derivative at initial time

    Returns:
        callable: Approximation of f(x, τ)
    """
    return lambda x: func0(x) + tau * func1(x) + 0.5 * tau**2 * func2(x)


# =========================
# SYMBOLIC FIELD DEFINITIONS
# =========================

# Symbolic displacement field u(x, t)
u = lambda x, t: SD.u(x, t)

# Symbolic rotation field v(x, t)
v = lambda x, t: SD.v(x, t)


# =========================
# INITIAL DATA CONSTRUCTOR FOR THE TIMOSHENKO BEAM
# =========================

def get_initial_data():
    """
    Construct symbolic source terms and initial data for solving the Timoshenko beam system.

    Returns:
        f1, f2 (callable): Forcing functions for u and v equations
        u0, u1 (callable): Initial data for displacement field u(x, 0) and its time-evolved value u(x, τ)
        v0, v1 (callable): Initial data for rotation field v(x, 0) and its time-evolved value v(x, τ)
    """

    # --- Source Terms ---
    f1 = lambda x, t: SD.f1(x, t)  # Longitudinal external force
    f2 = lambda x, t: SD.f2(x, t)  # Rotational external force

    # --- Initial Conditions for u(x, t) ---
    varphi0 = lambda x: u(x, 0)  # u(x, 0)
    varphi1 = lambda x: SD.diff1t_u(x, 0)  # ∂u/∂t at t=0

    # ∂²u/∂t² derived from the PDE:
    varphi2 = lambda x: (
        f1(x, 0)
        - cfg.a1 * SD.diff1x_v(x, 0)
        + (cfg.alpha + cfg.beta * SD.integr_term(0)) * SD.diff2x_u(x, 0)
    )

    # --- Initial Conditions for v(x, t) ---
    psi0 = lambda x: v(x, 0)  # v(x, 0)
    psi1 = lambda x: SD.diff1t_v(x, 0)  # ∂v/∂t at t=0

    # ∂²v/∂t² derived from the PDE:
    psi2 = lambda x: (
        f2(x, 0)
        + cfg.a2 * SD.diff1x_u(x, 0)
        + cfg.gamma * SD.diff2x_v(x, 0)
        - cfg.delta * psi0(x)
    )

    # --- Apply 2nd-order Taylor Expansion to compute approximate solution at t = τ ---
    u0 = varphi0
    u1 = taylor_expansion(cfg.tau, varphi0, varphi1, varphi2)

    v0 = psi0
    v1 = taylor_expansion(cfg.tau, psi0, psi1, psi2)

    return f1, f2, u0, u1, v0, v1


# =========================
# EXACT ANALYTICAL SOLUTION EVALUATOR
# =========================

def exact_solution(
    solution_type: str,
    unif_prt_spc: int = None,
    x_val: float = None,
    k: int = None
) -> np.ndarray | float:
    """
    Evaluate the exact analytical solution (from symbolic expression) of u(x, t) or v(x, t).

    Supports evaluation at:
    - A uniform spatial grid or a specific point `x_val`
    - A specific time index `k`, or full time evolution if `k` is None

    Parameters:
        solution_type (str): Choose 'u' for displacement or 'v' for rotation
        unif_prt_spc (int, optional): Number of uniform partitions over [0, ell] for grid evaluation
        x_val (float, optional): A specific spatial location in [0, ell]
        k (int, optional): Time index ∈ [0, n]; if provided, return only solution at t_k

    Returns:
        np.ndarray | float:
            - (n+1, len(x)) if full time evolution over a grid
            - 1D array if grid at single time t_k
            - float if point evaluation at x_val and time index k

    Raises:
        ValueError: If any argument is outside its expected domain
    """

    # --- Validate solution type ---
    if solution_type == 'u':
        func = u
    elif solution_type == 'v':
        func = v
    else:
        raise ValueError("solution_type must be either 'u' or 'v'.")

    # --- Validate and construct spatial domain ---
    if x_val is None and unif_prt_spc is None:
        raise ValueError("Either `unif_prt_spc` or `x_val` must be provided.")

    if x_val is not None:
        if not (0 <= x_val <= cfg.ell):
            raise ValueError(f"x_val = {x_val} is outside the spatial domain [0, {cfg.ell}].")
        x = np.array([x_val], dtype=float)  # Evaluate at a single spatial point
    else:
        x = np.linspace(0, float(cfg.ell), unif_prt_spc + 1)  # Uniform spatial grid

    # --- Evaluate symbolic function over all time steps ---
    values = np.array([func(x, t_i) for t_i in cfg.t])  # Shape: (n+1, len(x))

    # --- Handle time index output selection ---
    if k is not None:
        if not (0 <= k <= cfg.n):
            raise ValueError(f"k = {k} is invalid. Must be in range 0 to {cfg.n}.")
        return values[k]  # Either float (if x_val) or 1D array

    return values  # Entire time evolution across space