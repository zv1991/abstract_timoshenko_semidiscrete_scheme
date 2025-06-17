# =========================
# IMPORT MODULES
# =========================

import numpy as np  # For efficient array handling and numerical operations

# Import configuration constants for the PDE (e.g., time step `tau`, length `ell`, time array `t`, coefficients `a1`, `a2`, `alpha`, etc.)
import utils.config as cfg

# Import symbolic displacement fields and their partial derivatives from the symbolic derivative module
from utils.symbolic_derivatives import SymbolicDerivatives as SD


# ---------------------------------------------------------------------------
# UTILITY FUNCTION: SECOND-ORDER TAYLOR EXPANSION IN TIME
# ---------------------------------------------------------------------------

def taylor_expansion(tau: float, func0, func1, func2):
    """
    Compute a second-order Taylor expansion in time for a function f(x, t) at t = τ:
    
        f(x, τ) ≈ f(x, 0) + τ·f'(x, 0) + (τ² / 2)·f''(x, 0)

    Parameters:
        tau (float): Time step size (τ)
        func0 (callable): Function value at t = 0 → f(x, 0)
        func1 (callable): First time derivative at t = 0 → f'(x, 0)
        func2 (callable): Second time derivative at t = 0 → f''(x, 0)

    Returns:
        callable: A lambda function approximating f(x, τ)
    """
    return lambda x: func0(x) + tau * func1(x) + 0.5 * tau**2 * func2(x)


# ---------------------------------------------------------------------------
# SYMBOLIC DISPLACEMENT FIELDS
# ---------------------------------------------------------------------------

# Symbolic longitudinal displacement field u(x, t)
u = lambda x, t: SD.u(x, t)

# Symbolic rotational displacement field v(x, t)
v = lambda x, t: SD.v(x, t)


# ---------------------------------------------------------------------------
# INITIAL DATA GENERATOR FOR TIMOSHENKO BEAM SYSTEM
# ---------------------------------------------------------------------------

def get_initial_data():
    """
    Construct symbolic source terms and initial conditions for the Timoshenko beam PDE system.

    Returns:
        f1 (callable): Source term for the u-equation (longitudinal)
        f2 (callable): Source term for the v-equation (rotational)
        u0 (callable): Initial condition u(x, 0)
        u1 (callable): Approximate u(x, τ) via 2nd-order Taylor expansion
        v0 (callable): Initial condition v(x, 0)
        v1 (callable): Approximate v(x, τ) via 2nd-order Taylor expansion
    """

    # --- Source Terms ---
    f1 = lambda x, t: SD.f1(x, t)  # External forcing for longitudinal equation
    f2 = lambda x, t: SD.f2(x, t)  # External forcing for rotational equation

    # --- Initial Conditions for u(x, t) ---
    varphi0 = lambda x: u(x, 0)  # Displacement u at t = 0
    varphi1 = lambda x: SD.diff1t_u(x, 0)  # ∂u/∂t at t = 0

    # ∂²u/∂t² at t = 0 derived from PDE:
    varphi2 = lambda x: (
        f1(x, 0)
        - cfg.a1 * SD.diff1x_v(x, 0)
        + (cfg.alpha + cfg.beta * SD.integr_term(0)) * SD.diff2x_u(x, 0)
    )

    # --- Initial Conditions for v(x, t) ---
    psi0 = lambda x: v(x, 0)  # Displacement v at t = 0
    psi1 = lambda x: SD.diff1t_v(x, 0)  # ∂v/∂t at t = 0

    # ∂²v/∂t² at t = 0 derived from PDE:
    psi2 = lambda x: (
        f2(x, 0)
        + cfg.a2 * SD.diff1x_u(x, 0)
        + cfg.gamma * SD.diff2x_v(x, 0)
        - cfg.delta * psi0(x)
    )

    # --- Approximate Solutions at t = τ using Taylor Expansion ---
    u0 = varphi0
    u1 = taylor_expansion(cfg.tau, varphi0, varphi1, varphi2)

    v0 = psi0
    v1 = taylor_expansion(cfg.tau, psi0, psi1, psi2)

    return f1, f2, u0, u1, v0, v1


# ---------------------------------------------------------------------------
# EXACT ANALYTICAL SOLUTION EVALUATOR
# ---------------------------------------------------------------------------

def exact_solution(
    solution_type: str,
    unif_prt_spc: int = None,
    x_val: float = None,
    k: int = None
) -> np.ndarray:
    """
    Evaluate the exact analytical solution of u(x, t) or v(x, t) either over:
    - A uniform spatial grid, or
    - A specific spatial point.

    Can return:
    - The full time evolution (if k is None), or
    - A single time slice at t_k (if k is specified).

    Parameters:
        solution_type (str): 'u' for longitudinal, 'v' for rotational displacement.
        unif_prt_spc (int, optional): Number of uniform partitions of [0, ell].
        x_val (float, optional): Specific spatial location to evaluate.
        k (int, optional): Time index (0 ≤ k ≤ n). If given, returns only the k-th snapshot.

    Returns:
        np.ndarray: Solution values either at all times or a single snapshot.

    Raises:
        ValueError: If inputs are invalid.
    """

    # --- Select the corresponding symbolic function ---
    if solution_type == 'u':
        func = u
    elif solution_type == 'v':
        func = v
    else:
        raise ValueError("Invalid solution_type. Expected 'u' (deflection) or 'v' (rotation).")

    # --- Ensure spatial input is provided ---
    if x_val is None and unif_prt_spc is None:
        raise ValueError("Provide either `unif_prt_spc` (for grid) or `x_val` (for a point).")

    # --- Create spatial input ---
    if x_val is not None:
        if not (0 <= x_val <= cfg.ell):
            raise ValueError(f"x_val = {x_val} is outside the domain [0, {cfg.ell}].")
        x = np.array([x_val], dtype=float)  # Single-point evaluation
    else:
        x = np.linspace(0, float(cfg.ell), unif_prt_spc + 1)  # Uniform spatial grid

    # --- Evaluate function over all time steps ---
    values = np.array([func(x, t_i) for t_i in cfg.t])  # Shape: (n+1, len(x))

    # --- Return desired output ---
    if k is not None:
        if not (0 <= k <= cfg.n):
            raise ValueError(f"Invalid time index: k = {k}. Must satisfy 0 ≤ k ≤ {cfg.n}.")
        return values[k]

    return values