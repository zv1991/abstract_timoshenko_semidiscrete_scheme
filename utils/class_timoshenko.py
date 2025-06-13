# ======================================
# Import standard and custom libraries
# ======================================

import numpy as np  # For efficient numerical computations and array handling

# Custom utility modules
import utils.auxiliary as aux  # Auxiliary functions for projections and Galerkin evaluations
import utils.solver as soln    # Solver for evolving modal coefficients using Galerkin time stepping


class TimoshenkoModelSolver:
    """
    Solver for nonlinear Timoshenko beam equations using a Galerkin projection method.
    Transforms the PDE system into ODEs using modal decomposition, which are then solved numerically.
    """

    def __init__(self, ell: float, T: float,
                 alpha: float, beta: float, gamma: float, delta: float,
                 a1: float, a2: float,
                 n: int, N: int,
                 f1, f2,
                 u0, u1, v0, v1):
        """
        Initialize the solver with physical parameters, time domain, and initial/boundary data.

        Parameters:
        - ell (float): Length of the beam domain [0, ell]
        - T (float): Total simulation time
        - alpha, beta, gamma, delta (float): Material/model parameters
        - a1, a2 (float): Coupling coefficients in the PDE system
        - n (int): Number of time steps
        - N (int): Number of spatial Galerkin modes
        - f1, f2 (callable): External forcing functions, f1(x, t) and f2(x, t)
        - u0, u1, v0, v1 (callable): Initial functions that provide starting values for the numerical solution process
        """
        # Time and space discretization
        self.ell = ell                      # Spatial domain length
        self.T = T                          # Final simulation time
        self.n = n                          # Number of time steps
        self.N = N                          # Number of Galerkin modes
        self.tau = T / n                    # Time step size Δt
        self.t = np.linspace(0, T, n + 1)   # Time grid: 0 to T with (n+1) points

        # Model constants
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.a1 = a1
        self.a2 = a2

        # External forcing functions
        self.f1 = f1
        self.f2 = f2

        # Initial data for displacement and rotation
        self.u0 = u0  # u(x,0)
        self.u1 = u1  # ∂u/∂t(x,0)
        self.v0 = v0  # v(x,0)
        self.v1 = v1  # ∂v/∂t(x,0)
        self.u_initial = [u0, u1]
        self.v_initial = [v0, v1]

        # Solve the PDE using Galerkin method and store results
        self.tilde_u, self.tilde_v, self.cond_u, self.cond_v = self.solve_system()

    def solve_system(self):
        """
        Run the Galerkin solver to compute time evolution of modal coefficients.

        Returns:
        - tilde_u (ndarray): Modal coefficients of u(x, t) over time
        - tilde_v (ndarray): Modal coefficients of v(x, t) over time
        - cond_u (ndarray): Condition numbers of system matrix for u
        - cond_v (ndarray): Condition numbers of system matrix for v
        """
        return soln.solve_system(
            u_initial=self.u_initial,
            v_initial=self.v_initial,
            f1=self.f1,
            f2=self.f2
        )

    def galerkin_approx_u(self, unif_prt_spc: int = None, x_val: float = None, k: int = None):
        """
        Evaluate the Galerkin approximation of displacement u(x, t).

        Parameters:
        - unif_prt_spc (int): Number of points in uniform spatial grid [0, ell]
        - x_val (float): Evaluate u(x, t) at a specific x-location
        - k (int): Time step index (0 ≤ k ≤ n). If None, return values at all time steps.

        Returns:
        - np.ndarray or float: u(x, t) approximation at specified spatial/time location
        """
        if x_val is None and unif_prt_spc is None:
            raise ValueError("Specify either 'unif_prt_spc' for a grid or 'x_val' for a point.")

        # --- Spatial discretization ---
        if x_val is not None:
            if not (0 <= x_val <= self.ell):
                raise ValueError(f"x_val={x_val} is outside domain [0, {self.ell}]")
            x = np.array([x_val])
        else:
            x = np.linspace(0, self.ell, unif_prt_spc + 1)

        # --- Evaluate initial data (t = 0 and t = τ) ---
        row0 = self.u0(x)  # u(x, 0)
        row1 = self.u1(x)  # u(x, τ)

        # --- Evaluate Galerkin approximation for t ≥ 2 ---
        others = aux.galerkin_approx(self.ell, self.tilde_u, x)

        # --- Return value(s) based on time index ---
        if k is None:
            return np.vstack([row0, row1, others])
        if not (0 <= k <= self.n):
            raise ValueError(f"Invalid time index: k={k}, must be 0 ≤ k ≤ {self.n}.")
        if k == 0:
            return row0[0] if x_val is not None else row0
        elif k == 1:
            return row1[0] if x_val is not None else row1
        else:
            result = others[k - 2]
            return result[0] if x_val is not None else result

    def galerkin_approx_v(self, unif_prt_spc: int = None, x_val: float = None, k: int = None):
        """
        Evaluate the Galerkin approximation of rotation v(x, t).

        Parameters:
        - unif_prt_spc (int): Number of points in uniform spatial grid [0, ell]
        - x_val (float): Evaluate v(x, t) at a specific x-location
        - k (int): Time step index (0 ≤ k ≤ n). If None, return values at all time steps.

        Returns:
        - np.ndarray or float: v(x, t) approximation at specified spatial/time location
        """
        if x_val is None and unif_prt_spc is None:
            raise ValueError("Specify either 'unif_prt_spc' for a grid or 'x_val' for a point.")

        # --- Spatial discretization ---
        if x_val is not None:
            if not (0 <= x_val <= self.ell):
                raise ValueError(f"x_val={x_val} is outside domain [0, {self.ell}]")
            x = np.array([x_val])
        else:
            x = np.linspace(0, self.ell, unif_prt_spc + 1)

        # --- Evaluate initial data (t = 0 and t = τ) ---
        row0 = self.v0(x)  # v(x, 0)
        row1 = self.v1(x)  # v(x, τ)

        # --- Evaluate Galerkin approximation for t ≥ 2 ---
        others = aux.galerkin_approx(self.ell, self.tilde_v, x)

        # --- Return value(s) based on time index ---
        if k is None:
            return np.vstack([row0, row1, others])
        if not (0 <= k <= self.n):
            raise ValueError(f"Invalid time index: k={k}, must be 0 ≤ k ≤ {self.n}.")
        if k == 0:
            return row0[0] if x_val is not None else row0
        elif k == 1:
            return row1[0] if x_val is not None else row1
        else:
            result = others[k - 2]
            return result[0] if x_val is not None else result