# ----------------------------------------
# Module Imports
# ----------------------------------------

import numpy as np  # For efficient numerical operations, especially on arrays

# Configuration parameters for the beam problem
# Includes time step size `tau`, domain length `ell`, constants `a1, a2, alpha`, etc., and time grid `t`
import utils.config as cfg

# Symbolic derivatives and expressions for displacement, rotation, and their derivatives
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
        self.f1 = None  # Source term for u equation
        self.f2 = None  # Source term for v equation
        self.u0 = None  # u(x, 0)
        self.u1 = None  # u(x, τ)
        self.v0 = None  # v(x, 0)
        self.v1 = None  # v(x, τ)

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
            f1, f2: callables for source terms.
            u0, u1: displacement at t=0 and t=τ.
            v0, v1: rotation at t=0 and t=τ.
        """
        # --- Source terms for PDEs ---
        self.f1 = lambda x, t: SD.f1(x, t)  # Forcing function in u-equation
        self.f2 = lambda x, t: SD.f2(x, t)  # Forcing function in v-equation

        # --- Displacement u(x, t) initial values ---
        varphi0 = lambda x: self.u(x, 0)       # Initial value u(x, 0)
        varphi1 = lambda x: SD.diff1t_u(x, 0)  # Time derivative ∂u/∂t at t=0

        # Second time derivative ∂²u/∂t² using the PDE for u
        varphi2 = lambda x: (
            self.f1(x, 0)
            - cfg.a1 * SD.diff1x_v(x, 0)
            + (cfg.alpha + cfg.beta * SD.integr_term(0)) * SD.diff2x_u(x, 0)
        )

        # --- Rotation v(x, t) initial values ---
        psi0 = lambda x: self.v(x, 0)       # Initial value v(x, 0)
        psi1 = lambda x: SD.diff1t_v(x, 0)  # Time derivative ∂v/∂t at t=0

        # Second time derivative ∂²v/∂t² using the PDE for v
        psi2 = lambda x: (
            self.f2(x, 0)
            + cfg.a2 * SD.diff1x_u(x, 0)
            + cfg.gamma * SD.diff2x_v(x, 0)
            - cfg.delta * psi0(x)
        )

        # --- Compute Taylor-expanded values at t = τ ---
        self.u0 = varphi0
        self.u1 = self.taylor_expansion(cfg.tau, varphi0, varphi1, varphi2)
        self.v0 = psi0
        self.v1 = self.taylor_expansion(cfg.tau, psi0, psi1, psi2)

        return self.f1, self.f2, self.u0, self.u1, self.v0, self.v1

    # ----------------------------------------
    # Evaluate Exact Symbolic Solutions
    # ----------------------------------------

    def exact_solution(self, solution_type: str, unif_prt_spc: int = None,
                       x_val: float = None, k: int = None) -> np.ndarray | float:
        """
        Evaluate exact solution u(x, t) or v(x, t) at given spatial points and time step.

        Parameters
        ----------
        solution_type : str
            'u' for displacement or 'v' for rotation.
        unif_prt_spc : int, optional
            Number of uniform partitions in space.
        x_val : float, optional
            Specific spatial location to evaluate.
        k : int, optional
            Time step index.

        Returns
        -------
        np.ndarray or float
            Solution at the requested location/time.
        """
        # Choose symbolic function based on solution type
        if solution_type == 'u':
            func = self.u
        elif solution_type == 'v':
            func = self.v
        else:
            raise ValueError("solution_type must be 'u' or 'v'.")

        # Input validation for spatial arguments
        if x_val is None and unif_prt_spc is None:
            raise ValueError("Specify either x_val or unif_prt_spc.")

        # Generate spatial input array
        if x_val is not None:
            if not (0 <= x_val <= cfg.ell):
                raise ValueError(f"x_val = {x_val} is outside domain [0, {cfg.ell}].")
            x = np.array([x_val])  # Evaluate at a single point
        else:
            x = np.linspace(0, cfg.ell, unif_prt_spc + 1)  # Uniform grid

        # Evaluate symbolic function at all time steps
        values = np.array([func(x, t_i) for t_i in cfg.t])  # Shape: (n+1, len(x))

        if k is not None:
            if not (0 <= k <= cfg.n):
                raise ValueError(f"Time index k = {k} out of range [0, {cfg.n}].")
            return values[k]  # Return solution at a specific time index

        return values  # Return full time evolution

    # ----------------------------------------
    # Construct Exact Solution Functions
    # ----------------------------------------

    def compute_exact_solution(self, solution_type: str, k: int = None,
                               x_vals: float | int | list | np.ndarray = None):
        """
        Return callable(s) or evaluated value(s) of the exact solution at specific time/space.

        Parameters
        ----------
        solution_type : str
            'u' (displacement) or 'v' (rotation)
        k : int, optional
            Time step index. If None, includes all time steps.
        x_vals : float | int | list | np.ndarray, optional
            Spatial positions to evaluate.

        Returns
        -------
        callable | list[callable] | float | np.ndarray
            Function(s) or evaluated values at requested space/time.
        """
        # Choose symbolic function
        if solution_type == 'u':
            func = self.u
        elif solution_type == 'v':
            func = self.v
        else:
            raise ValueError("solution_type must be either 'u' or 'v'.")

        # Validate and standardize input type
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

        # Function generator for fixed time index
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
                raise ValueError(f"Time index k = {k_idx} must be in [0, {cfg.n}].")
            return lambda x: func(x, cfg.t[k_idx])

        # Single time step: return function or its evaluation
        if k is not None:
            fn = construct_exact_function_at_k(k)
            return fn if x_vals is None else fn(x_vals)

        # All time steps: return list of functions or evaluations
        all_functions = [construct_exact_function_at_k(k_idx) for k_idx in range(cfg.n + 1)]
        return all_functions if x_vals is None else np.array([fn(x_vals) for fn in all_functions])