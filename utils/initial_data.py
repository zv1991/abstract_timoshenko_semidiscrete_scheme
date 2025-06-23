# =========================
# IMPORT MODULES
# =========================

import numpy as np  # Efficient numerical computations and array manipulation

# Configuration parameters: time step size, domain length, physical constants, and time grid
import utils.config as cfg

# Symbolic PDE expressions: displacement fields, derivatives, source terms, etc.
from utils.symbolic_derivatives import SymbolicDerivatives as SD


# =========================
# TAYLOR EXPANSION UTILITY
# =========================

def taylor_expansion(tau: float, func0, func1, func2):
    """
    Compute a second-order Taylor approximation in time:
        f(x, τ) ≈ f(x, 0) + τ·f'(x, 0) + ½·τ²·f''(x, 0)

    Parameters
    ----------
    tau : float
        Time step size (Δt or τ)
    func0 : callable
        f(x, 0), function at t=0
    func1 : callable
        f'(x, 0), first time derivative at t=0
    func2 : callable
        f''(x, 0), second time derivative at t=0

    Returns
    -------
    callable
        A function of x that approximates f(x, τ)
    """
    return lambda x: func0(x) + tau * func1(x) + 0.5 * tau**2 * func2(x)


# =========================
# SYMBOLIC FIELD DEFINITIONS
# =========================

# Symbolic displacement and rotation fields from the symbolic derivatives module
u = lambda x, t: SD.u(x, t)  # u(x, t): Displacement
v = lambda x, t: SD.v(x, t)  # v(x, t): Rotation


# =========================
# INITIAL DATA GENERATOR
# =========================

def get_initial_data():
    """
    Constructs initial data (displacement, velocity, forcing) using symbolic PDE definitions.

    Returns
    -------
    f1, f2 : callable
        Source terms for the u and v equations respectively.
    u0, u1 : callable
        u(x, 0) and u(x, τ): initial displacement and Taylor-expanded value.
    v0, v1 : callable
        v(x, 0) and v(x, τ): initial rotation and Taylor-expanded value.
    """

    # --- Source (forcing) terms ---
    f1 = lambda x, t: SD.f1(x, t)  # Forcing function in u-equation
    f2 = lambda x, t: SD.f2(x, t)  # Forcing function in v-equation

    # --- Displacement u(x, t) initial values ---
    varphi0 = lambda x: u(x, 0)  # Initial value u(x, 0)
    varphi1 = lambda x: SD.diff1t_u(x, 0)  # Time derivative ∂u/∂t at t=0

    # Second time derivative ∂²u/∂t² using PDE expression
    varphi2 = lambda x: (
        f1(x, 0)
        - cfg.a1 * SD.diff1x_v(x, 0)
        + (cfg.alpha + cfg.beta * SD.integr_term(0)) * SD.diff2x_u(x, 0)
    )

    # --- Rotation v(x, t) initial values ---
    psi0 = lambda x: v(x, 0)  # Initial value v(x, 0)
    psi1 = lambda x: SD.diff1t_v(x, 0)  # Time derivative ∂v/∂t at t=0

    # Second time derivative ∂²v/∂t² using PDE expression
    psi2 = lambda x: (
        f2(x, 0)
        + cfg.a2 * SD.diff1x_u(x, 0)
        + cfg.gamma * SD.diff2x_v(x, 0)
        - cfg.delta * psi0(x)
    )

    # --- Compute approximate initial values at t = τ via Taylor expansion ---
    u0 = varphi0
    u1 = taylor_expansion(cfg.tau, varphi0, varphi1, varphi2)

    v0 = psi0
    v1 = taylor_expansion(cfg.tau, psi0, psi1, psi2)

    return f1, f2, u0, u1, v0, v1


# =========================
# EXACT SOLUTION EVALUATOR
# =========================

def exact_solution(
    solution_type: str,
    unif_prt_spc: int = None,
    x_val: float = None,
    k: int = None
) -> np.ndarray | float:
    """
    Evaluate the exact symbolic solution u(x, t) or v(x, t) over a grid or at a point in time and space.

    Parameters
    ----------
    solution_type : str
        'u' for displacement or 'v' for rotation.
    unif_prt_spc : int, optional
        Number of spatial partitions for generating a grid.
    x_val : float, optional
        Single spatial point in [0, ell] to evaluate the function at.
    k : int, optional
        Time index (0 ≤ k ≤ n). If None, returns all time steps.

    Returns
    -------
    np.ndarray or float
        - Full time evolution array of shape (n+1, len(x)) if `k` is None and grid is specified.
        - 1D array at single time `k` across space.
        - Single float if both `x_val` and `k` are given.

    Raises
    ------
    ValueError
        If arguments are outside valid domains or inconsistent.
    """

    # Select the appropriate symbolic function
    if solution_type == 'u':
        func = u
    elif solution_type == 'v':
        func = v
    else:
        raise ValueError("solution_type must be 'u' or 'v'.")

    # Ensure spatial evaluation is properly defined
    if x_val is None and unif_prt_spc is None:
        raise ValueError("Specify either `x_val` (single point) or `unif_prt_spc` (grid).")

    if x_val is not None:
        if not (0 <= x_val <= cfg.ell):
            raise ValueError(f"x_val = {x_val} is outside the domain [0, {cfg.ell}].")
        x = np.array([x_val], dtype=float)  # Single-point evaluation
    else:
        x = np.linspace(0, cfg.ell, unif_prt_spc + 1)  # Uniform spatial grid

    # Evaluate over all time steps
    values = np.array([func(x, t_i) for t_i in cfg.t])  # Shape: (n+1, len(x))

    if k is not None:
        if not (0 <= k <= cfg.n):
            raise ValueError(f"Invalid time index k = {k}. Must be in range [0, {cfg.n}].")
        return values[k]  # Return slice at t_k

    return values  # Return full time evolution


# =========================
# EXACT SOLUTION FUNCTION GENERATOR
# =========================

def compute_exact_solution(
    solution_type: str,
    k: int = None,
    x_vals: float | int | list | np.ndarray = None
):
    """
    Construct or evaluate the exact symbolic solution u(x, t_k) or v(x, t_k).

    Parameters
    ----------
    solution_type : str
        Type of solution to return:
            'u' - Displacement field
            'v' - Rotation field
    k : int, optional
        Time step index (0 ≤ k ≤ cfg.n). If None, applies to all time steps.
    x_vals : float | int | list | np.ndarray, optional
        Spatial location(s) to evaluate the solution. If None, returns callable(s)
        instead of evaluated values.

    Returns
    -------
    callable or list of callables, or float or np.ndarray
        - If x_vals is None:
            → Single callable if k is provided
            → List of callables if k is None
        - If x_vals is provided:
            → Single float if x_vals is scalar and k is given
            → NumPy array of evaluations otherwise

    Raises
    ------
    ValueError
        If `solution_type` is not 'u' or 'v', or if k is out of valid range.
    TypeError
        If x_vals is of an unsupported type.
    """

    # --- Select the symbolic solution function based on solution type ---
    if solution_type == 'u':
        func = u  # Symbolic displacement function: u(x, t)
    elif solution_type == 'v':
        func = v  # Symbolic rotation function: v(x, t)
    else:
        raise ValueError("solution_type must be either 'u' or 'v'.")

    # --- Helper function to normalize and validate x_vals ---
    def validate_and_convert_x_vals(x_input):
        """
        Ensures x_vals is of an acceptable type and converts it to float or ndarray.

        Parameters
        ----------
        x_input : float | int | list | np.ndarray | None
            Input spatial location(s)

        Returns
        -------
        float or np.ndarray or None
        """
        if isinstance(x_input, (float, int)):
            return float(x_input)
        elif isinstance(x_input, list):
            return np.array(x_input, dtype=float)
        elif isinstance(x_input, np.ndarray):
            return x_input.astype(float)
        elif x_input is None:
            return None
        else:
            raise TypeError("x_vals must be float, int, list, or np.ndarray.")

    x_vals = validate_and_convert_x_vals(x_vals)

    # --- Construct callable u(x, t_k) or v(x, t_k) at time index k_idx ---
    def construct_exact_function_at_k(k_idx: int):
        """
        Returns a function representing u(x, t_k) or v(x, t_k).

        Parameters
        ----------
        k_idx : int
            Time index (0 ≤ k_idx ≤ cfg.n)

        Returns
        -------
        callable
            Function of one spatial variable x
        """
        if not (0 <= k_idx <= cfg.n):
            raise ValueError(f"Time index k = {k_idx} must be in range [0, {cfg.n}].")
        return lambda x: func(x, cfg.t[k_idx])

    # --- Case 1: Single time step evaluation ---
    if k is not None:
        fn = construct_exact_function_at_k(k)
        return fn if x_vals is None else fn(x_vals)

    # --- Case 2: All time steps ---
    all_functions = [construct_exact_function_at_k(k_idx) for k_idx in range(cfg.n + 1)]
    return all_functions if x_vals is None else np.array([fn(x_vals) for fn in all_functions])