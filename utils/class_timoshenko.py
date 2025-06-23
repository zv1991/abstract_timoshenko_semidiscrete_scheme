# ======================================
# Import standard and custom libraries
# ======================================

import numpy as np  # NumPy is used for numerical computations, especially for arrays and linear algebra.

# Custom utility modules containing problem-specific numerical routines
import utils.auxiliary as aux  # Provides Galerkin basis functions and projection utilities.
import utils.solver as soln    # Contains the time integrator for reduced-order ODE systems.


class TimoshenkoModelSolver:
    """
    Solver for the nonlinear Timoshenko beam system using Galerkin projection.

    This class performs a spatial discretization via Galerkin modal decomposition to reduce
    the PDE system into an ODE system, and then integrates it over time to track modal dynamics.
    """

    def __init__(
        self,
        ell: float, T: float,
        alpha: float, beta: float, gamma: float, delta: float,
        a1: float, a2: float,
        n: int, N: int,
        f1, f2,
        u0, u1, v0, v1,
        # Optional parameters for solve_system
        h: float = 1e-3, derivmeth: str = 'nd', tol: float = 1e-6, method: str = 'hglq',
        max_n: int = 50, max_depth: int = 20, n_points: int = 10
    ):
        """
        Initializes the Timoshenko beam model and solver parameters.

        Args:
            ell (float): Beam length.
            T (float): Final simulation time.
            alpha, beta, gamma, delta (float): Physical material/damping parameters.
            a1, a2 (float): Coupling constants.
            n (int): Number of time steps.
            N (int): Number of Galerkin basis modes.
            f1, f2 (callable): External force functions.
            u0, u1, v0, v1 (callable): Initial condition functions at t=0 and t=τ.
            h, derivmeth, tol, method, max_n, max_depth, n_points: Numerical solver parameters.
        """

        # === Temporal discretization ===
        self.ell = ell
        self.T = T
        self.n = n
        self.N = N
        self.tau = T / n
        self.t = np.linspace(0, T, n + 1)

        # === Physical parameters ===
        self.alpha, self.beta = alpha, beta
        self.gamma, self.delta = gamma, delta
        self.a1, self.a2 = a1, a2

        # === Initial conditions and external forcing ===
        self.f1, self.f2 = f1, f2
        self.u0, self.u1 = u0, u1
        self.v0, self.v1 = v0, v1
        self.u_initial = [u0, u1]
        self.v_initial = [v0, v1]

        # === Solver configuration ===
        self.h = h
        self.derivmeth = derivmeth
        self.tol = tol
        self.method = method
        self.max_n = max_n
        self.max_depth = max_depth
        self.n_points = n_points

        # === Solve the reduced-order system immediately ===
        self.tilde_u, self.tilde_v, self.cond_u, self.cond_v = self.solve_system()

    def solve_system(self):
        """
        Calls the external solver for the Galerkin-reduced ODE system.

        Returns:
            tuple: Modal coefficients for displacement and rotation (tilde_u, tilde_v),
                   and condition numbers (cond_u, cond_v).
        """
        return soln.solve_system(
            u_initial=self.u_initial,
            v_initial=self.v_initial,
            f1=self.f1,
            f2=self.f2,
            h=self.h,
            derivmeth=self.derivmeth,
            tol=self.tol,
            method=self.method,
            max_n=self.max_n,
            max_depth=self.max_depth,
            n_points=self.n_points
        )

    def galerkin_approx_solution(
        self,
        solution_type: str,
        unif_prt_spc: int = None,
        x_val: float = None,
        k: int = None
    ) -> np.ndarray | float:
        """
        Reconstructs the approximate solution u(x,t) or v(x,t) via modal coefficients.

        Args:
            solution_type (str): 'u' for displacement, 'v' for rotation.
            unif_prt_spc (int): Number of uniform spatial points (for full reconstruction).
            x_val (float): Specific spatial location.
            k (int): Time step index for evaluation (optional).

        Returns:
            np.ndarray or float: Full space-time approximation, time slice, or point value.
        """

        if solution_type not in {'u', 'v'}:
            raise ValueError("solution_type must be either 'u' or 'v'.")

        coeffs = self.tilde_u if solution_type == 'u' else self.tilde_v
        init_0 = self.u0 if solution_type == 'u' else self.v0
        init_1 = self.u1 if solution_type == 'u' else self.v1

        # Determine spatial grid
        if x_val is not None:
            if not (0 <= x_val <= self.ell):
                raise ValueError(f"x_val = {x_val} out of bounds [0, {self.ell}].")
            x = np.array([x_val])
        elif unif_prt_spc is not None:
            x = np.linspace(0, self.ell, unif_prt_spc + 1)
        else:
            raise ValueError("Provide either 'x_val' or 'unif_prt_spc'.")

        # Evaluate initial conditions
        row0 = init_0(x)
        row1 = init_1(x)

        # Evaluate using Galerkin coefficients for remaining time steps
        others = aux.galerkin_approx(self.ell, coeffs, x)
        all_results = np.vstack([row0, row1, others])

        if k is not None:
            if not (0 <= k <= self.n):
                raise ValueError(f"k = {k} must be within [0, {self.n}].")
            return all_results[k][0] if x_val is not None else all_results[k]

        return all_results

    def compute_ansatz(
        self,
        solution_type: str,
        k: int = None,
        x_vals: float | int | list | np.ndarray = None
    ):
        """
        Constructs or evaluates the spatial function u(x, t_k) or v(x, t_k).

        Args:
            solution_type (str): 'u' for displacement, 'v' for rotation.
            k (int): Time step index.
            x_vals (float | list | np.ndarray): Specific spatial location(s).

        Returns:
            callable or np.ndarray or float: Callable spatial function(s) or evaluated result(s).
        """

        if solution_type not in {'u', 'v'}:
            raise ValueError("solution_type must be 'u' or 'v'.")

        # === Normalize x input ===
        def validate_and_convert_x_vals(x_input):
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

        # === Basis function generator ===
        def generate_basis():
            return [
                (lambda m: (lambda x: aux.phi_m(m, self.ell, x)))(m + 1)
                for m in range(self.N)
            ]

        # === Construct spatial function at given time step ===
        def construct_function_at_k(k_idx: int):
            if k_idx == 0:
                return self.u0 if solution_type == 'u' else self.v0
            elif k_idx == 1:
                return self.u1 if solution_type == 'u' else self.v1

            coeffs = self.tilde_u[k_idx - 2] if solution_type == 'u' else self.tilde_v[k_idx - 2]
            basis = generate_basis()
            return lambda x: sum(c * phi(x) for c, phi in zip(coeffs, basis))

        # === Single time step ===
        if k is not None:
            if not (0 <= k <= self.n):
                raise ValueError(f"k = {k} must be in range [0, {self.n}].")
            fn = construct_function_at_k(k)
            return fn if x_vals is None else fn(x_vals)

        # === All time steps ===
        functions = [construct_function_at_k(k_idx) for k_idx in range(self.n + 1)]
        return functions if x_vals is None else np.array([fn(x_vals) for fn in functions])