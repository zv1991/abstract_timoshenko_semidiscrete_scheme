# ======================================
# Import standard and custom libraries
# ======================================

# NumPy: Used for efficient array operations and numerical computing
import numpy as np

# Custom module with helper functions for Galerkin projection and numerical routines
import utils.auxiliary as aux

# Symbolic module for defining initial conditions for the Timoshenko model
import utils.initial_conditions_symb as ic

# Numerical solver module that implements the Galerkin method for PDEs
import utils.solver as soln


class TimoshenkoModelSolver:
    """
    Solver class for the nonlinear Timoshenko beam equations using the Galerkin method.
    This approach reduces the coupled PDE system to a system of ODEs via modal decomposition.
    """

    def __init__(self, ell, T,
                 alpha, beta, gamma, delta, a1, a2,
                 n, N,
                 f1, f2,
                 u0, u1,
                 v0, v1):
        """
        Initialize the solver with problem parameters.

        Parameters:
        - ell (float): Length of the spatial domain.
        - T (float): Total time for simulation.
        - alpha, beta, gamma, delta (float): Material and model coefficients.
        - a1, a2 (float): Coupling coefficients for the PDE system.
        - n (int): Number of time steps.
        - N (int): Number of Galerkin modes (basis functions).
        - f1, f2 (callable): External forces (functions of x and t).
        - u0, u1, v0, v1 (callable): Initial condition functions for displacement and velocity.
        """
        self.ell = ell
        self.T = T

        # Model parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.a1 = a1
        self.a2 = a2

        # Time discretization
        self.n = n
        self.N = N
        self.tau = T / n  # Time step size
        self.t = np.linspace(0, T, n + 1)  # Discrete time grid

        # Source functions and initial conditions
        self.f1 = f1
        self.f2 = f2
        self.u0 = u0
        self.u1 = u1
        self.v0 = v0
        self.v1 = v1

        # Solve the system and retrieve modal coefficients and condition numbers
        self.tilde_u, self.tilde_v, self.cond_u, self.cond_v = self.solve_system()

    def solve_system(self):
        """
        Solve the reduced system of ODEs obtained by Galerkin projection.

        Returns:
        - tilde_u, tilde_v (np.ndarray): Modal coefficients for u and v over time.
        - cond_u, cond_v (list): Condition numbers of system matrices at each step.
        """
        return soln.solve_system(
            data=ic.setup_initial_conditions(),
            f1=self.f1,
            f2=self.f2
        )

    def galerkin_approx_u(self, unif_prt_spc: int = None, x_val: float = None, k: int = None):
        """
        Compute the Galerkin approximation of u(x, t).

        Parameters:
        - unif_prt_spc (int, optional): Number of uniform spatial points (used if x_val is None).
        - x_val (float, optional): Specific spatial location to evaluate u(x, t).
        - k (int, optional): Time step index (0 ≤ k ≤ n). If None, return full time evolution.

        Returns:
        - np.ndarray or float: Approximate solution u(x, t) at given x and t.
        """
        if x_val is None and unif_prt_spc is None:
            raise ValueError("Either 'unif_prt_spc' or 'x_val' must be specified.")

        # Generate spatial points
        if x_val is not None:
            if not (0 <= x_val <= self.ell):
                raise ValueError("x_val must lie within [0, ell].")
            x = np.array([x_val])
        else:
            x = np.linspace(0, self.ell, unif_prt_spc + 1)

        # Evaluate initial data
        row0 = self.u0(x)
        row1 = self.u1(x)

        # Evaluate modal approximation for later time steps (k ≥ 2)
        others = aux.galerkin_approx(self.ell, self.tilde_u, x)

        # Handle output based on k
        if k is None:
            return np.vstack([row0, row1, others])
        if not (0 <= k <= self.n):
            raise ValueError("Time step k must satisfy 0 ≤ k ≤ n.")
        if k == 0:
            return row0[0] if x_val is not None else row0
        elif k == 1:
            return row1[0] if x_val is not None else row1
        else:
            result = others[k - 2]
            return result[0] if x_val is not None else result

    def galerkin_approx_v(self, unif_prt_spc: int = None, x_val: float = None, k: int = None):
        """
        Compute the Galerkin approximation of v(x, t).

        Parameters:
        - unif_prt_spc (int, optional): Number of uniform spatial points (used if x_val is None).
        - x_val (float, optional): Specific spatial location to evaluate v(x, t).
        - k (int, optional): Time step index (0 ≤ k ≤ n). If None, return full time evolution.

        Returns:
        - np.ndarray or float: Approximate solution v(x, t) at given x and t.
        """
        if x_val is None and unif_prt_spc is None:
            raise ValueError("Either 'unif_prt_spc' or 'x_val' must be specified.")

        # Generate spatial points
        if x_val is not None:
            if not (0 <= x_val <= self.ell):
                raise ValueError("x_val must lie within [0, ell].")
            x = np.array([x_val])
        else:
            x = np.linspace(0, self.ell, unif_prt_spc + 1)

        # Evaluate initial data
        row0 = self.v0(x)
        row1 = self.v1(x)

        # Evaluate modal approximation for later time steps (k ≥ 2)
        others = aux.galerkin_approx(self.ell, self.tilde_v, x)

        # Handle output based on k
        if k is None:
            return np.vstack([row0, row1, others])
        if not (0 <= k <= self.n):
            raise ValueError("Time step k must satisfy 0 ≤ k ≤ n.")
        if k == 0:
            return row0[0] if x_val is not None else row0
        elif k == 1:
            return row1[0] if x_val is not None else row1
        else:
            result = others[k - 2]
            return result[0] if x_val is not None else result