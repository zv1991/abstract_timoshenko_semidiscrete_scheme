# ----------------------------------------
# Module Imports
# ----------------------------------------

import numpy as np  # For efficient numerical operations, especially on arrays

# Configuration parameters for the beam problem.
# Includes time step size `tau`, domain length `ell`, coefficients (a1, a2, alpha, etc.), and time grid `t`.
import utils.config as cfg

# Symbolic derivatives and expressions for displacement, rotation, and their derivatives.
from utils.symbolic_derivatives import SymbolicDerivatives as SD


# ----------------------------------------
# Benchmark Class Definition
# ----------------------------------------

class TimoshenkoBenchmark:
    """
    Benchmarking class for solving the nonlinear Timoshenko beam problem.

    Maintains stateful access to computed initial data (displacement, rotation, sources).
    """

    def __init__(self):
        """
        Initialize placeholders for symbolic field evaluations.
        These are computed once and reused to avoid redundancy.
        """
        self.f1 = None  # Source term for the u equation
        self.f2 = None  # Source term for the v equation
        self.u0 = None  # Displacement at t = 0
        self.u1 = None  # Displacement at t = τ (via Taylor expansion)
        self.v0 = None  # Rotation at t = 0
        self.v1 = None  # Rotation at t = τ (via Taylor expansion)

    # ----------------------------------------
    # Internal helper to retrieve symbolic function
    # ----------------------------------------

    def _get_solution_function(self, solution_type: str):
        """
        Return the symbolic function corresponding to the solution type.

        Parameters
        ----------
        solution_type : str
            Either 'u' (displacement) or 'v' (rotation)

        Returns
        -------
        callable
            Corresponding symbolic function

        Raises
        ------
        ValueError
            If solution_type is not 'u' or 'v'
        """
        if solution_type == 'u':
            return self.u
        elif solution_type == 'v':
            return self.v
        else:
            raise ValueError("solution_type must be 'u' or 'v'.")

    # ----------------------------------------
    # Time Discretization: Taylor Expansion
    # ----------------------------------------

    def taylor_expansion(self, tau: float, func0, func1, func2):
        """
        Perform a second-order Taylor expansion for a time-dependent function:
        f(x, τ) ≈ f(x, 0) + τ·f'(x, 0) + ½·τ²·f''(x, 0)

        Parameters
        ----------
        tau : float
            Time step size.
        func0 : callable
            Function value at t=0.
        func1 : callable
            First time derivative at t=0.
        func2 : callable
            Second time derivative at t=0.

        Returns
        -------
        callable
            Approximated function f(x, τ)
        """
        return lambda x: func0(x) + tau * func1(x) + 0.5 * tau**2 * func2(x)

    # ----------------------------------------
    # Accessors for Symbolic Fields
    # ----------------------------------------

    def u(self, x, t):
        """Symbolic displacement function u(x, t)."""
        return SD.u(x, t)

    def v(self, x, t):
        """Symbolic rotation function v(x, t)."""
        return SD.v(x, t)

    # ----------------------------------------
    # Initial Data Construction
    # ----------------------------------------

    def get_initial_data(self):
        """
        Compute initial displacement and rotation using symbolic expressions and Taylor expansion.

        Returns
        -------
        tuple:
            f1, f2: callable source terms for u and v equations.
            u0, u1: displacement at t=0 and t=τ.
            v0, v1: rotation at t=0 and t=τ.
        """
        # --- Source terms ---
        self.f1 = lambda x, t: SD.f1(x, t)
        self.f2 = lambda x, t: SD.f2(x, t)

        # --- Displacement u(x, t) initial data ---
        varphi0 = lambda x: self.u(x, 0)                 # u(x, 0)
        varphi1 = lambda x: SD.diff1t_u(x, 0)            # ∂u/∂t at t=0
        varphi2 = lambda x: (                            # ∂²u/∂t² from PDE
            self.f1(x, 0)
            - cfg.a1 * SD.diff1x_v(x, 0)
            + (cfg.alpha + cfg.beta * SD.integr_term(0)) * SD.diff2x_u(x, 0)
        )

        # --- Rotation v(x, t) initial data ---
        psi0 = lambda x: self.v(x, 0)                    # v(x, 0)
        psi1 = lambda x: SD.diff1t_v(x, 0)               # ∂v/∂t at t=0
        psi2 = lambda x: (                               # ∂²v/∂t² from PDE
            self.f2(x, 0)
            + cfg.a2 * SD.diff1x_u(x, 0)
            + cfg.gamma * SD.diff2x_v(x, 0)
            - cfg.delta * psi0(x)
        )

        # --- Apply Taylor expansion to get values at t = τ ---
        self.u0 = varphi0
        self.u1 = self.taylor_expansion(cfg.tau, varphi0, varphi1, varphi2)
        self.v0 = psi0
        self.v1 = self.taylor_expansion(cfg.tau, psi0, psi1, psi2)

        return self.f1, self.f2, self.u0, self.u1, self.v0, self.v1

    # ----------------------------------------
    # Evaluate Exact Symbolic Solutions on Grid
    # ----------------------------------------

    def exact_solution_on_grid(self, solution_type: str, unif_prt_spc: int = None,
                                x_val: float = None, k: int = None) -> np.ndarray | float:
        """
        Evaluate exact solution u(x, t) or v(x, t) at given spatial points and time step.

        Parameters
        ----------
        solution_type : str
            'u' for displacement or 'v' for rotation.
        unif_prt_spc : int, optional
            Number of uniform spatial partitions (if grid is desired).
        x_val : float, optional
            Evaluate at a specific point.
        k : int, optional
            Time step index.

        Returns
        -------
        np.ndarray or float
            Result at all or specific (x, t) locations.
        """
        func = self._get_solution_function(solution_type)

        if x_val is None and unif_prt_spc is None:
            raise ValueError("Specify either x_val or unif_prt_spc.")

        # Build spatial array
        if x_val is not None:
            if not (0 <= x_val <= cfg.ell):
                raise ValueError(f"x_val = {x_val} is outside domain [0, {cfg.ell}].")
            x = np.array([x_val])
        else:
            x = np.linspace(0, cfg.ell, unif_prt_spc + 1)

        # Evaluate over time
        values = np.array([func(x, t_i) for t_i in cfg.t])

        if k is not None:
            if not (0 <= k <= cfg.n):
                raise ValueError(f"Time index k = {k} out of range [0, {cfg.n}].")
            return values[k]

        return values

    # ----------------------------------------
    # Construct Callable Exact Solution Functions
    # ----------------------------------------

    def callable_exact_solution(self, solution_type: str, k: int = None,
                                x_vals: float | int | list | np.ndarray = None):
        """
        Return callable(s) or evaluated value(s) of the exact solution at specific time/space.

        Parameters
        ----------
        solution_type : str
            'u' (displacement) or 'v' (rotation)
        k : int, optional
            Specific time step index.
        x_vals : float | int | list | np.ndarray, optional
            Spatial positions to evaluate (or return callables if None).

        Returns
        -------
        callable | list[callable] | float | np.ndarray
            Callable function(s) or evaluated result(s)
        """
        func = self._get_solution_function(solution_type)

        def validate_and_convert_x_vals(x_input):
            """Normalize supported types into float or ndarray."""
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

        def construct_exact_function_at_k(k_idx: int):
            if not (0 <= k_idx <= cfg.n):
                raise ValueError(f"Time index k = {k_idx} must be in [0, {cfg.n}].")
            return lambda x: func(x, cfg.t[k_idx])

        if k is not None:
            fn = construct_exact_function_at_k(k)
            return fn if x_vals is None else fn(x_vals)

        all_functions = [construct_exact_function_at_k(k_idx) for k_idx in range(cfg.n + 1)]
        return all_functions if x_vals is None else np.array([fn(x_vals) for fn in all_functions])