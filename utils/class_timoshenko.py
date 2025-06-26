# --------------------------------------------------------------------------- #
"""
TimoshenkoModelSolver: Galerkin-based Solver for Nonlinear Timoshenko Beam PDEs

This class numerically solves the coupled nonlinear system of partial differential
equations modeling the dynamic behavior of a Timoshenko beam. It uses a Legendre–
Galerkin method to reduce the PDEs to a system of time-dependent ordinary 
differential equations (ODEs) by projecting the solution onto a basis of
normalized shifted Legendre polynomials over the spatial domain [0, ell].

The ODE system is then solved using a leapfrog-type time integration scheme,
with modal coefficients stored and used to reconstruct the solution.

Key Features:
-------------
- Modal Galerkin projection in space using orthonormal Legendre basis functions
- Time stepping based on leapfrog integrator, using initial data at t = 0 and t = τ
- Support for analytical first derivatives (du₀, du₁, dv₀, dv₁) or numerical approximation
- Evaluation of second derivatives ⟨uᵢ″, φₘ⟩ and ⟨vᵢ″, φₘ⟩ via integration by parts
- Flexible quadrature interface for adaptive integration:
    - Supports 'hglq', 'glq', and SciPy's routines
- Nonlinear coupling through the squared norm of the spatial derivative ‖u₁′‖²
- Outputs include:
    - Modal coefficients for displacement and rotation
    - Condition numbers for Galerkin matrices (diagnostic information)
- Provides callable solution reconstruction and pointwise evaluation support

External dependencies:
- `utils.auxiliary` (aux): Basis functions, projection routines
- `utils.solver` (soln): Core time-stepping ODE solver

Usage:
------
Instantiate the class with beam parameters, initial and forcing data, then
use built-in methods to evaluate or reconstruct the solution at desired points.
"""
# --------------------------------------------------------------------------- #

import numpy as np  # Core numerical computing (arrays, linear algebra)

# Project-specific numerical routines
import utils.auxiliary as aux  # Galerkin operators, Legendre basis, projections
import utils.solver as soln    # Time integration and modal system solver


class TimoshenkoModelSolver:
    """
    Nonlinear PDE solver for the Timoshenko system using a Galerkin framework.
    """

    def __init__(
        self,
        ell: float, T: float,
        alpha: float, beta: float, gamma: float, delta: float,
        a1: float, a2: float,
        n: int, N: int,
        f1, f2,
        u0, u1, v0, v1,
        # Optional analytical derivatives
        du0=None, du1=None, dv0=None, dv1=None,
        # Optional solver parameters
        h: float = 1e-3, derivmeth: str = 'nd', tol: float = 1e-6, method: str = 'hglq',
        max_n: int = 50, max_depth: int = 20, n_points: int = 10
    ):
        """
        Initialize the problem configuration and solve immediately.

        Parameters
        ----------
        ell : float
            Length of the beam (domain [0, ell]).
        T : float
            Final simulation time.
        alpha, beta, gamma, delta : float
            System physical parameters.
        a1, a2 : float
            Coupling coefficients for the system.
        n : int
            Number of time steps.
        N : int
            Number of Galerkin modes.
        f1, f2 : callable
            External forcing functions (f₁(x, t), f₂(x, t)).
        u0, u1, v0, v1 : callable
            Initial condition functions at t = 0 and t = τ.
        du0, du1, dv0, dv1 : callable or None
            Optional analytical derivatives of the initial data.
        h : float
            Step size for numerical derivatives (if needed).
        derivmeth : str
            Method for numerical differentiation ('nd' or 'sfd').
        tol, method, max_n, max_depth, n_points : float | int
            Quadrature control parameters.
        """
        # Spatial and temporal setup
        self.ell = ell
        self.T = T
        self.n = n
        self.N = N
        self.tau = T / n
        self.t = np.linspace(0, T, n + 1)

        # Model parameters
        self.alpha, self.beta = alpha, beta
        self.gamma, self.delta = gamma, delta
        self.a1, self.a2 = a1, a2

        # Store initial condition functions
        self.u0, self.u1 = u0, u1
        self.v0, self.v1 = v0, v1
        self.u_initial = [u0, u1]
        self.v_initial = [v0, v1]

        # Derivatives: optional analytical expressions
        self.du = [du0, du1]
        self.dv = [dv0, dv1]

        # External sources
        self.f1 = f1
        self.f2 = f2

        # Quadrature and numerical settings
        self.h = h
        self.derivmeth = derivmeth
        self.tol = tol
        self.method = method
        self.max_n = max_n
        self.max_depth = max_depth
        self.n_points = n_points

        # Precompute modal solution on initialization
        self.tilde_u, self.tilde_v, self.cond_u, self.cond_v = self.solve_system()

    def solve_system(self):
        """
        solve_system: Solve the Galerkin-reduced system of ODEs for the Timoshenko model
        
        This method performs time integration on a modal ODE system derived from projecting
        the nonlinear Timoshenko beam PDEs onto a Legendre–Galerkin basis.
        
        Method Overview:
        ----------------
        - Projects initial conditions and external forcing onto modal basis functions
        - Supports optional analytical first derivatives (du, dv) for improved accuracy
        - Computes second derivative projections ⟨uᵢ″, φₘ⟩ and ⟨vᵢ″, φₘ⟩ via integration by parts
        - Evaluates nonlinear stiffness term qₖ = α + β * ‖uₖ′‖² at each time step
        - Solves the resulting system using preassembled modal matrices
        - Applies leapfrog-type time-stepping scheme for k ≥ 2
        - Tracks matrix condition numbers for numerical diagnostics
        
        Returns:
        --------
        tuple
            (tilde_u, tilde_v): Modal coefficients of displacement u(x, t) and rotation v(x, t)
            (cond_u, cond_v): Condition numbers of the Galerkin system matrices
        """
        
        return soln.solve_system(
            u_initial=self.u_initial,  # [u₀(x), u₁(x)]
            v_initial=self.v_initial,  # [v₀(x), v₁(x)]
            f1=self.f1,                # External force f₁(x, t)
            f2=self.f2,                # External force f₂(x, t)
            du=self.du,                # Optional: [du₀(x), du₁(x)]
            dv=self.dv,                # Optional: [dv₀(x), dv₁(x)]
            h=self.h,                  # Step size for numerical differentiation
            derivmeth=self.derivmeth,  # Derivative computation method ('nd' or 'sfd')
            tol=self.tol,              # Quadrature tolerance
            method=self.method,        # Quadrature method ('hglq', 'glq', 'scipy')
            max_n=self.max_n,          # Max points for adaptive quadrature
            max_depth=self.max_depth,  # Max recursion depth for quadrature
            n_points=self.n_points     # Fixed points for Gaussian quadrature
        )
    
    def galerkin_approx_solution_on_grid(
        self,
        solution_type: str,
        unif_prt_spc: int = None,
        x_val: float = None,
        k: int = None
    ) -> np.ndarray | float:
        """
        galerkin_approx_solution_on_grid: Evaluate Galerkin solution at spatial points or grid
        
        This method reconstructs the Galerkin-approximated solution using precomputed modal 
        coefficients. It allows evaluation at a specific spatial point or over a uniform 
        spatial grid, for a given time index.
        
        Parameters
        ----------
        solution_type : str
            Either 'u' for displacement or 'v' for rotation.
        unif_prt_spc : int, optional
            Number of uniform spatial intervals to generate grid points.
        x_val : float, optional
            Specific spatial coordinate at which to evaluate the solution.
        k : int, optional
            Time index (0 ≤ k ≤ n) for evaluation.
        
        Returns
        -------
        np.ndarray or float
            Reconstructed solution at all time steps on a spatial grid (array) or 
            single value at a specific point in space and time.
        """
        
        if solution_type not in {'u', 'v'}:
            raise ValueError("solution_type must be either 'u' or 'v'.")

        # Select modal data and initial values based on solution type
        coeffs = self.tilde_u if solution_type == 'u' else self.tilde_v
        init_0 = self.u0 if solution_type == 'u' else self.v0
        init_1 = self.u1 if solution_type == 'u' else self.v1

        # Setup spatial points
        if x_val is not None:
            if not (0 <= x_val <= self.ell):
                raise ValueError(f"x_val = {x_val} is outside domain [0, {self.ell}].")
            x = np.array([x_val])
        elif unif_prt_spc is not None:
            x = np.linspace(0, self.ell, unif_prt_spc + 1)
        else:
            raise ValueError("Specify either x_val or unif_prt_spc.")

        # Evaluate solution at t = 0, τ
        row0 = init_0(x)
        row1 = init_1(x)

        # Use Galerkin modes for t ≥ 2τ
        others = aux.galerkin_approx(self.ell, coeffs, x)
        all_results = np.vstack([row0, row1, others])  # Stack along time axis

        if k is not None:
            if not (0 <= k <= self.n):
                raise ValueError(f"k = {k} is out of bounds [0, {self.n}].")
            return all_results[k][0] if x_val is not None else all_results[k]

        return all_results

    def callable_compute_ansatz(
        self,
        solution_type: str,
        k: int = None,
        x_vals: float | int | list | np.ndarray = None
    ):
        """
        Return callable or evaluated Galerkin ansatz u(x,t_k) or v(x,t_k)
        
        Parameters
        ----------
        solution_type : str
            'u' (displacement) or 'v' (rotation)
        k : int, optional
            Time step index
        x_vals : float | list | np.ndarray, optional
            Evaluation points (optional)
        
        Returns
        -------
        Callable or np.ndarray
            Function u(x) or v(x), or array of values
        """
        
        if solution_type not in {'u', 'v'}:
            raise ValueError("solution_type must be 'u' or 'v'.")

        # Normalize and validate x input
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

        # Generate Galerkin basis functions φₘ(x)
        def generate_basis():
            return [
                (lambda m: (lambda x: aux.phi_m(m, self.ell, x)))(m + 1)
                for m in range(self.N)
            ]

        # Construct solution u(x,t_k) or v(x,t_k)
        def construct_function_at_k(k_idx: int):
            if k_idx == 0:
                return self.u0 if solution_type == 'u' else self.v0
            elif k_idx == 1:
                return self.u1 if solution_type == 'u' else self.v1

            coeffs = self.tilde_u[k_idx - 2] if solution_type == 'u' else self.tilde_v[k_idx - 2]
            basis = generate_basis()
            return lambda x: sum(c * phi(x) for c, phi in zip(coeffs, basis))

        # Return callable or evaluation at time k
        if k is not None:
            if not (0 <= k <= self.n):
                raise ValueError(f"k = {k} must be in range [0, {self.n}].")
            fn = construct_function_at_k(k)
            return fn if x_vals is None else fn(x_vals)

        # Return all time steps as callables or evaluated results
        functions = [construct_function_at_k(k_idx) for k_idx in range(self.n + 1)]
        return functions if x_vals is None else np.array([fn(x_vals) for fn in functions])