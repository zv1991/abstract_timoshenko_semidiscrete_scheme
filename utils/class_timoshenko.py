# ======================================
# Import standard and custom libraries
# ======================================

import numpy as np  # Core numerical operations, vector math, time-space grid construction

# Custom utility modules
import utils.auxiliary as aux  # Contains Galerkin basis functions and modal projection utilities
import utils.solver as soln    # Numerical solver for modal ODE systems derived from PDEs


class TimoshenkoModelSolver:
    """
    Solver for the nonlinear Timoshenko beam system using a Galerkin approximation.
    Converts the governing PDEs into a reduced system of ODEs via modal decomposition,
    which is then solved numerically.
    """

    def __init__(
        self,
        ell: float,
        T: float,
        alpha: float, beta: float, gamma: float, delta: float,
        a1: float, a2: float,
        n: int, N: int,
        f1, f2,
        u0, u1, v0, v1
    ):
        """
        Initialize simulation configuration and trigger the numerical solution.

        Parameters
        ----------
        ell : float
            Domain length of the beam [0, ell].
        T : float
            Final simulation time.
        alpha, beta, gamma, delta : float
            Physical and damping coefficients for the beam model.
        a1, a2 : float
            Coupling coefficients in the u/v system.
        n : int
            Number of time steps (discretization in time).
        N : int
            Number of Galerkin basis functions (discretization in space).
        f1, f2 : callable
            External forcing functions f1(x, t) and f2(x, t).
        u0, u1, v0, v1 : callable
            Initial conditions: displacements and velocities at t = 0.
        """
        # Store configuration
        self.ell = ell
        self.T = T
        self.n = n
        self.N = N
        self.tau = T / n  # Time step size
        self.t = np.linspace(0, T, n + 1)  # Time grid: [0, τ, ..., T]

        # Model parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.a1 = a1
        self.a2 = a2

        # Forcing functions and initial conditions
        self.f1, self.f2 = f1, f2
        self.u0, self.u1 = u0, u1
        self.v0, self.v1 = v0, v1
        self.u_initial = [u0, u1]
        self.v_initial = [v0, v1]

        # Solve ODE system upon initialization
        self.tilde_u, self.tilde_v, self.cond_u, self.cond_v = self.solve_system()

    def solve_system(self):
        """
        Solve the time-evolution of modal coefficients via Galerkin projection.

        Returns
        -------
        tilde_u, tilde_v : np.ndarray
            Modal coefficients (N-dimensional) for u(x, t) and v(x, t).
        cond_u, cond_v : np.ndarray
            Diagnostics: condition numbers of system matrices over time.
        """
        return soln.solve_system(
            u_initial=self.u_initial,
            v_initial=self.v_initial,
            f1=self.f1,
            f2=self.f2
        )

    def galerkin_approx_solution(
        self,
        solution_type: str,
        unif_prt_spc: int = None,
        x_val: float = None,
        k: int = None
    ) -> np.ndarray | float:
        """
        Reconstruct the physical solution u(x, t) or v(x, t) from modal coefficients.

        Parameters
        ----------
        solution_type : str
            Either 'u' (displacement) or 'v' (rotation).
        unif_prt_spc : int, optional
            Number of intervals in spatial grid. Required if x_val not specified.
        x_val : float, optional
            Evaluate solution at this single spatial point.
        k : int, optional
            Specific time index to evaluate. If None, return all time steps.

        Returns
        -------
        np.ndarray or float
            Solution evaluated at requested space-time grid, or full space-time array.
        """
        if solution_type not in {'u', 'v'}:
            raise ValueError("solution_type must be 'u' or 'v'.")

        coeffs = self.tilde_u if solution_type == 'u' else self.tilde_v
        init_0 = self.u0 if solution_type == 'u' else self.v0
        init_1 = self.u1 if solution_type == 'u' else self.v1

        # Determine spatial grid
        if x_val is None and unif_prt_spc is None:
            raise ValueError("Specify either 'x_val' or 'unif_prt_spc'.")
        if x_val is not None:
            if not (0 <= x_val <= self.ell):
                raise ValueError(f"x_val = {x_val} must lie in [0, {self.ell}].")
            x = np.array([x_val])
        else:
            x = np.linspace(0, self.ell, unif_prt_spc + 1)

        # Evaluate initial conditions
        row0 = init_0(x)
        row1 = init_1(x)

        # Evaluate modal contributions (from k = 2 onward)
        others = aux.galerkin_approx(self.ell, coeffs, x)
        all_results = np.vstack([row0, row1, others])

        # Return specific time step or all
        if k is not None:
            if not (0 <= k <= self.n):
                raise ValueError(f"k = {k} is out of valid range [0, {self.n}].")
            return all_results[k][0] if x_val is not None else all_results[k]
        return all_results

    def compute_ansatz(self, solution_type: str, k: int = None):
        """
        Return Galerkin-based approximation of the solution at a specific time index or full series.

        Parameters
        ----------
        solution_type : str
            'u' or 'v' for type of solution.
        k : int, optional
            Specific time index to return callable at time t_k. If None, return full time series evaluator.

        Returns
        -------
        callable or list of callable
            - If k is given: returns a function u(x) at time t_k
            - If k is None: returns a list of u(x) functions for each time step
        """
        if solution_type not in {'u', 'v'}:
            raise ValueError("solution_type must be 'u' or 'v'.")

        def generate_basis():
            """
            Construct Galerkin basis functions φₘ(x) using orthogonal polynomials.

            Returns
            -------
            list of callable
                Basis functions φ₁(x), ..., φ_N(x)
            """
            return [(lambda m: (lambda x: aux.phi_m(m, self.ell, x)))(m + 1) for m in range(self.N)]

        def evaluate_at_time_k(k_idx: int, x=None):
            """
            Evaluate Galerkin expansion at time index `k_idx`.

            Parameters
            ----------
            k_idx : int
                Time index.
            x : float or array-like, optional
                If provided, evaluates immediately at spatial point(s).

            Returns
            -------
            callable or float or np.ndarray
                Function u(x) or v(x), or its evaluated value(s).
            """
            if k_idx == 0:
                fn = self.u0 if solution_type == 'u' else self.v0
            elif k_idx == 1:
                fn = self.u1 if solution_type == 'u' else self.v1
            else:
                coeffs = self.tilde_u[k_idx - 2] if solution_type == 'u' else self.tilde_v[k_idx - 2]
                basis = generate_basis()
                fn = lambda x_val: sum(c * phi(x_val) for c, phi in zip(coeffs, basis))

            if x is not None:
                x = np.asarray(x, dtype=float)
                return fn(x) if x.ndim == 0 else np.array([fn(xi) for xi in x])
            return fn

        if k is not None:
            if not isinstance(k, int) or not (0 <= k <= self.n):
                raise ValueError(f"k = {k} is outside valid range [0, {self.n}].")
            return evaluate_at_time_k(k)

        def evaluate_all_timesteps(x):
            """
            Evaluate Galerkin expansion at all time steps at point(s) x.

            Parameters
            ----------
            x : float or array-like
                Spatial input.

            Returns
            -------
            list
                [u₀(x), u₁(x), ..., uₙ(x)]
            """
            x = float(x) if np.isscalar(x) else np.asarray(x, dtype=float)
            return [evaluate_at_time_k(k_idx, x) for k_idx in range(self.n + 1)]

        return evaluate_all_timesteps