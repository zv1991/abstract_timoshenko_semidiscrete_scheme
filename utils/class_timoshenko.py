# ======================================
# Import standard and custom libraries
# ======================================

import numpy as np  # For efficient numerical computations and array operations

# Custom modules for Galerkin approximation and solver functionality
import utils.auxiliary as aux  # Provides Galerkin projection evaluation
import utils.solver as soln    # Contains system solver logic for modal coefficients


class TimoshenkoModelSolver:
    """
    Solves the nonlinear Timoshenko beam PDE system using Galerkin projection.
    Converts PDEs into ODEs using modal decomposition and solves them numerically.
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
        Initializes the Timoshenko model solver.

        Parameters
        ----------
        ell : float
            Length of the spatial domain [0, ell].
        T : float
            Final simulation time.
        alpha, beta, gamma, delta : float
            Model coefficients for material and damping properties.
        a1, a2 : float
            Coupling coefficients between u and v equations.
        n : int
            Number of time steps.
        N : int
            Number of Galerkin basis functions (spatial modes).
        f1, f2 : callable
            External forcing functions f1(x, t) and f2(x, t).
        u0, u1, v0, v1 : callable
            Initial displacement/velocity for u(x,0), u(x,τ), v(x,0), v(x,τ).
        """

        # Physical domain and time discretization
        self.ell = ell                      # Beam length
        self.T = T                          # Final simulation time
        self.n = n                          # Number of time steps
        self.N = N                          # Number of Galerkin modes (spatial)
        self.tau = T / n                    # Time step size Δt
        self.t = np.linspace(0, T, n + 1)   # Time grid of length (n + 1)

        # PDE coefficients
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.a1 = a1
        self.a2 = a2

        # External forcing terms
        self.f1 = f1
        self.f2 = f2

        # Initial values for u and v equations
        self.u0 = u0
        self.u1 = u1
        self.v0 = v0
        self.v1 = v1
        self.u_initial = [u0, u1]
        self.v_initial = [v0, v1]

        # Solve system on initialization and store results
        self.tilde_u, self.tilde_v, self.cond_u, self.cond_v = self.solve_system()

    def solve_system(self):
        """
        Solve the time-evolved system using Galerkin projection and return results.

        Returns
        -------
        tilde_u : np.ndarray
            Modal coefficients for u(x, t) over time.
        tilde_v : np.ndarray
            Modal coefficients for v(x, t) over time.
        cond_u : np.ndarray
            Condition numbers for the linear system in u-equation (for diagnostics).
        cond_v : np.ndarray
            Condition numbers for the linear system in v-equation (for diagnostics).
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
        Evaluate the Galerkin-approximated solution u(x, t) or v(x, t).

        Parameters
        ----------
        solution_type : str
            Type of solution: 'u' (transverse displacement) or 'v' (rotational displacement).
        unif_prt_spc : int, optional
            Number of spatial grid partitions (used when `x_val` is not specified).
        x_val : float, optional
            Evaluate at specific spatial coordinate in [0, ell].
        k : int, optional
            Time index. If specified, returns solution at time t_k; else returns all time steps.

        Returns
        -------
        np.ndarray | float
            Evaluated solution:
            - If `k` is None: np.ndarray of shape (n+1, len(x))
            - If `x_val` is provided and `k` is specified: float
            - If only `k` is specified: 1D np.ndarray
        """

        # Select initial data and coefficients based on solution type
        if solution_type == 'u':
            init_0 = self.u0
            init_1 = self.u1
            coeffs = self.tilde_u
        elif solution_type == 'v':
            init_0 = self.v0
            init_1 = self.v1
            coeffs = self.tilde_v
        else:
            raise ValueError("Invalid solution_type. Use 'u' or 'v'.")

        # Validate and generate spatial grid
        if x_val is None and unif_prt_spc is None:
            raise ValueError("You must specify either 'unif_prt_spc' or 'x_val'.")

        if x_val is not None:
            if not (0 <= x_val <= self.ell):
                raise ValueError(f"x_val = {x_val} is outside [0, {self.ell}].")
            x = np.array([x_val])  # Treat as 1D grid with one point
        else:
            x = np.linspace(0, self.ell, unif_prt_spc + 1)  # Uniform spatial grid

        # Evaluate initial conditions
        row0 = init_0(x)  # t = 0
        row1 = init_1(x)  # t = τ

        # Galerkin solution for t ≥ 2 (from modal coefficients)
        others = aux.galerkin_approx(self.ell, coeffs, x)  # Shape: (n-1, len(x))

        # Assemble full time evolution
        all_results = np.vstack([row0, row1, others])  # Shape: (n+1, len(x))

        # Return result for single time step or all
        if k is not None:
            if not (0 <= k <= self.n):
                raise ValueError(f"k = {k} is outside valid range [0, {self.n}].")
            return all_results[k][0] if x_val is not None else all_results[k]

        return all_results