# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # Efficient numerical operations on arrays

# Configuration module with model parameters
import utils.config as cfg

# Known symbolic solutions and their derivatives for the Timoshenko system
import utils.known_solns as ks


# ======================================================
# TIMOSHENKO BENCHMARK CLASS
# ======================================================

class TimoshenkoBenchmark:
    """
    Benchmarking class for the nonlinear Timoshenko beam system.

    Responsibilities:
    -----------------
    - Provide initial conditions (u₀, u₁, v₀, v₁) and source terms (f₁, f₂)
    - Optionally expose first spatial derivatives of u and v
    - Interface with exact symbolic solutions for numerical validation
    """

    def __init__(self):
        """Initialize placeholders for symbolic initial data and their derivatives."""
        self.f1 = None  # Source term for u-equation
        self.f2 = None  # Source term for v-equation

        self.u0 = self.u1 = None  # Displacement u(x, 0), u(x, τ)
        self.v0 = self.v1 = None  # Rotation v(x, 0), v(x, τ)

        self.du0 = self.du1 = None  # ∂u/∂x at t=0, τ
        self.dv0 = self.dv1 = None  # ∂v/∂x at t=0, τ

    # ------------------------------------------------------
    # Symbolic Displacement and Rotation Fields
    # ------------------------------------------------------

    @staticmethod
    def u(x, t):
        """Exact symbolic displacement u(x, t)."""
        return ks.u(x, t)

    @staticmethod
    def v(x, t):
        """Exact symbolic rotation v(x, t)."""
        return ks.v(x, t)

    # ------------------------------------------------------
    # Load Initial Data from Symbolic Solutions
    # ------------------------------------------------------

    def get_initial_data(self):
        """
        Set initial field values and source terms from symbolic expressions.

        Returns
        -------
        tuple:
            (f₁, f₂, u₀, u₁, v₀, v₁, du₀, du₁, dv₀, dv₁)
        """
        # Source terms for the PDE system
        self.f1 = lambda x, t: ks.f1(x, t)
        self.f2 = lambda x, t: ks.f2(x, t)

        # Displacement at t=0 and t=τ
        self.u0 = lambda x: self.u(x, 0)
        self.u1 = lambda x: self.u(x, cfg.tau)

        # Rotation at t=0 and t=τ
        self.v0 = lambda x: self.v(x, 0)
        self.v1 = lambda x: self.v(x, cfg.tau)

        # First spatial derivatives
        self.du0 = lambda x: ks.diff1x_u(x, 0)
        self.du1 = lambda x: ks.diff1x_u(x, cfg.tau)

        self.dv0 = lambda x: ks.diff1x_v(x, 0)
        self.dv1 = lambda x: ks.diff1x_v(x, cfg.tau)

        return (
            self.f1, self.f2,
            self.u0, self.u1,
            self.v0, self.v1,
            self.du0, self.du1,
            self.dv0, self.dv1
        )

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
    
    # ----------------------------------------
    # Internal Dispatch
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