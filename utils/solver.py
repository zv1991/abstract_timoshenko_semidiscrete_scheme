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
    - Uses adaptive Gauss-Legendre integration with custom tolerance and resolution
- Nonlinear coupling through the squared norm of the spatial derivative ‖uᵢ′‖²
- Outputs include:
    - Modal coefficients for displacement and rotation
    - Condition numbers for Galerkin matrices (diagnostic information)
- Provides callable solution reconstruction and pointwise evaluation support

External dependencies:
- `utils.auxiliary` (aux): Basis functions, projection routines

"""
# --------------------------------------------------------------------------- #

import numpy as np  # Core numerical computing (arrays, linear algebra)
import time         # For timing the computation
from tqdm import tqdm  # Progress bar
import utils.auxiliary as aux  # Galerkin operators, Legendre basis, projections


class TimoshenkoModelSolver:
    """
    Nonlinear PDE solver for the Timoshenko system using a Galerkin framework.
    """

    # ----------------------------------------------------------------------- #
    #                         Class Initialization Method                     #
    # ----------------------------------------------------------------------- #
    def __init__(
        self,
        ell, T,                     # Domain length and final time
        alpha, beta, gamma, delta,  # PDE coefficients
        a1, a2,                     # Coupling constants
        n, N,           # Time discretization and number of Galerkin basis modes
        f1, f2,         # External forcing functions
        u0, u1, v0, v1, # Initial conditions
        du0=None, du1=None, dv0=None, dv1=None,  # Optional: known spatial derivatives of initial data
        h=1e-3,             # Finite difference step size (default: 1e-3)
        derivmeth='nd',     # Differentiation scheme ('nd' = numdifftools, 'sfd' = standard manually implemented)
        tol=1e-6,           # Integration tolerance for adaptive quadrature
        min_dx=1/128,       # Minimum interval width in adaptive integration
        n_gauss=5,          # Initial Gauss–Legendre points per subinterval
        max_gauss=50        # Maximum Gauss–Legendre nodes per interval
    ):
        # Spatial and temporal domain setup
        self.ell = ell               # Beam length (domain: [0, ell])
        self.T = T                   # Final simulation time
        self.n = n                   # Number of discrete time steps
        self.N = N                   # Number of Galerkin basis modes
        self.tau = T / n             # Time step size (τ)
        self.t = np.linspace(0, T, n + 1)  # Discretized time grid

        # Physical system parameters
        self.alpha, self.beta = alpha, beta         # Nonlinear stiffness and coupling terms
        self.gamma, self.delta = gamma, delta       # Damping and shear correction parameters
        self.a1, self.a2 = a1, a2                   # Coupling coefficients between displacement and rotation

        # Initial conditions (displacement and rotation)
        self.u0, self.u1 = u0, u1                   # Displacement at t = 0 and t = τ
        self.v0, self.v1 = v0, v1                   # Rotation at t = 0 and t = τ

        # Optional analytical derivatives of initial conditions
        self.du0, self.du1 = du0, du1               # First spatial derivatives of displacement at t=0 and t=τ
        self.dv0, self.dv1 = dv0, dv1               # First spatial derivatives of rotation at t=0 and t=τ

        # Forcing terms (external inputs)
        self.f1 = f1                                # Forcing for displacement equation
        self.f2 = f2                                # Forcing for rotation equation

        # Quadrature and differentiation configuration
        self.h = h                  # Step size for numerical differentiation (default: 1e-3)
        self.derivmeth = derivmeth  # Method for derivative approximation (default: 'nd' = numdifftools)
        self.tol = tol              # Absolute tolerance for adaptive quadrature (default: 1e-6)
        self.min_dx = min_dx        # Minimum allowed subinterval width (default: 1/128)
        self.n_gauss = n_gauss      # Initial number of Gauss–Legendre nodes (default: 5)
        self.max_gauss = max_gauss  # Max allowed Gauss–Legendre nodes adaptively (default: 50)
        
        # Constant used in the v-equation formulation in the Timoshenko system
        self.a0 = 4.0 / (2.0 + self.delta * self.tau**2)

        # Solve system and store modal coefficients and diagnostic data
        self.tilde_u, self.tilde_v, self.cond_u, self.cond_v, self.q_integr = self.solve_system()

    # ----------------------------------------------------------------------- #
    #                   Solve Modal Galerkin System Method                    #
    # ----------------------------------------------------------------------- #
    def solve_system(self):
        """
        Solve the Galerkin-reduced system of ODEs for the Timoshenko model.

        Returns
        -------
        tuple
            tilde_u, tilde_v : modal coefficient arrays for u(x,t) and v(x,t)
            cond_u, cond_v   : condition numbers of system matrices (stability diagnostic)
            q_integr         : nonlinear coefficients α + β ‖u′‖² over time
        """
        
        # Record start time
        start_time = time.time()
        # Notify user that computation is starting
        print("Computation has been started.")
        
        # Quadrature configuration passed to integration routines
        quad_kwargs = dict(
            tol=self.tol,
            min_dx=self.min_dx,
            n_gauss=self.n_gauss,
            max_gauss=self.max_gauss
        )

        # Project time-dependent forcing terms onto modal basis
        print("Projecting time-dependent forcing terms onto the modal basis.")
        f1_integr = aux.compute_time_dependent_integrals(self.f1, self.N, self.ell, self.t, **quad_kwargs)
        f2_integr = aux.compute_time_dependent_integrals(self.f2, self.N, self.ell, self.t, **quad_kwargs)

        # Project initial conditions and their first/second derivatives
        print("Projecting initial conditions and computing their first and second derivatives.")
        init_data = aux.compute_initial_integrals(
            [self.u0, self.u1], [self.v0, self.v1], self.N, self.ell,
            du=[self.du0, self.du1], dv=[self.dv0, self.dv1],
            h=self.h, derivmeth=self.derivmeth, **quad_kwargs
        )

        # Extract modal coefficients and derivative projections
        u0_integr, u1_integr = init_data['u_proj']
        v0_integr, v1_integr = init_data['v_proj']
        diff1u1 = init_data['diff1_u1']              # ⟨u₁′, φₘ⟩
        diff1v1 = init_data['diff1_v1']              # ⟨v₁′, φₘ⟩
        diff2u = init_data['diff2_u']                # ⟨uₖ″, φₘ⟩ over time
        diff2v = init_data['diff2_v']                # ⟨vₖ″, φₘ⟩ over time

        # Compute initial nonlinear coefficient q₁ = α + β * ‖u₁′‖²
        print("Computing the initial integral associated with the nonlinear term.")
        integral, _ = aux.integrate_derivative_form(
            f=self.u1 if self.du1 is None else None,
            df=self.du1 if self.du1 is not None else None,
            ell=self.ell,
            form='squared',
            m=None,
            h=self.h,
            derivmeth=self.derivmeth,
            **quad_kwargs
        )
        q_prev = self.alpha + self.beta * integral  # Initial nonlinearity coefficient

        # Allocate arrays for modal coefficients and matrix condition numbers
        tild_u = np.zeros((self.n - 1, self.N))      # Modal displacement coefficients
        tild_v = np.zeros((self.n - 1, self.N))      # Modal rotation coefficients
        cond_u = np.zeros(self.n - 1)                # Condition numbers for u-equation matrix
        cond_v = np.zeros(self.n - 1)                # Condition numbers for v-equation matrix
        q_integr = [None, q_prev]                    # Track qₖ = α + β * ‖uₖ′‖² for all steps

        # ----------------------------------------------------------
        # Time-stepping loop using leapfrog-type scheme
        # ----------------------------------------------------------
        
        for k in tqdm(range(self.n - 1), desc="Solving Timoshenko system", unit="step"):
            # Compute right-hand side (RHS) for linear systems at time step k
            if k == 0:
                # Conducting the first step: uses projected ICs at t=0, t=τ (special handling)
                b1 = (4 / self.ell**2) * (
                    self.tau**2 * f1_integr[k] + 2 * u1_integr
                    - self.a1 * self.tau**2 * diff1v1
                    - u0_integr + 0.5 * self.tau**2 * q_prev * diff2u[k]
                )
                b2 = (8 / (2 + self.delta * self.tau**2) / self.ell**2) * (
                    self.tau**2 * f2_integr[k] + 2 * v1_integr
                    + self.a2 * self.tau**2 * diff1u1
                    - (1 + 0.5 * self.tau**2 * self.delta) * v0_integr
                    + 0.5 * self.tau**2 * self.gamma * diff2v[k]
                )

            elif k == 1:
                # For the second step: uses Galerkin stencils from the previous step
                b1 = (4 / self.ell**2) * (
                    self.tau**2 * f1_integr[k]
                    + 0.5 * self.ell**2 * aux.galerkin_stencils(self.N, tild_u[k - 1])
                    - 0.5 * self.a1 * self.tau**2 * self.ell *
                      aux.galerkin_stencils(self.N, tild_v[k - 1], operator="first-order")
                    - u1_integr + 0.5 * self.tau**2 * q_prev * diff2u[k]
                )
                b2 = (8 / (2 + self.delta * self.tau**2) / self.ell**2) * (
                    self.tau**2 * f2_integr[k]
                    + 0.5 * self.ell**2 * aux.galerkin_stencils(self.N, tild_v[k - 1])
                    + 0.5 * self.a2 * self.tau**2 * self.ell *
                      aux.galerkin_stencils(self.N, tild_u[k - 1], operator="first-order")
                    - (1 + 0.5 * self.tau**2 * self.delta) * v1_integr
                    + 0.5 * self.tau**2 * self.gamma * diff2v[k]
                )

            else:
                # All later steps use fully recursive leapfrog stencils
                b1 = (
                    (4 * self.tau**2 / self.ell**2) * f1_integr[k]
                    + 2 * aux.galerkin_stencils(self.N, tild_u[k - 1])
                    - (2 * self.a1 * self.tau**2 / self.ell) *
                      aux.galerkin_stencils(self.N, tild_v[k - 1], operator="first-order")
                )
                b2 = (
                    (8 * self.tau**2 / (2 + self.delta * self.tau**2) / self.ell**2) * f2_integr[k]
                    + (4 / (2 + self.delta * self.tau**2)) *
                      aux.galerkin_stencils(self.N, tild_v[k - 1])
                    + (4 * self.a2 * self.tau**2 / (2 + self.delta * self.tau**2) / self.ell) *
                      aux.galerkin_stencils(self.N, tild_u[k - 1], operator="first-order")
                )

            # Compute condition numbers for diagnostic purposes
            cond_u[k] = aux.condition_number_associated_matrix(self.N, self.ell, 1, 0.5 * self.tau**2 * q_prev)
            cond_v[k] = aux.condition_number_associated_matrix(
                self.N, self.ell,
                1 + 0.5 * self.tau**2 * self.delta,
                0.5 * self.tau**2 * self.gamma
            )

            # Solve linear systems for modal coefficients
            tild_u[k] = aux.sys_soln(b1, self.N, 1, 0.5 * self.tau**2 * q_prev, self.ell)
            tild_v[k] = aux.sys_soln(b2, self.N,
                                     1 + 0.5 * self.tau**2 * self.delta,
                                     0.5 * self.tau**2 * self.gamma, self.ell)

            # Leapfrog update for k ≥ 2: subtract previous solution
            if k >= 2:
                tild_u[k] -= tild_u[k - 2]
                tild_v[k] -= tild_v[k - 2]

            # Update nonlinear coefficient qₖ for next step
            q_prev = self.alpha + self.beta * np.dot(tild_u[k], tild_u[k])
            q_integr.append(q_prev)
        
        elapsed_time = time.time() - start_time
        # Notify when computation is finished
        print(f"Computation has been completed in {elapsed_time:.2f} seconds.")
        
        # Return computed quantities
        return tild_u, tild_v, cond_u, cond_v, q_integr
    
    # ----------------------------------------------------------------------- #
    #                Evaluate Galerkin Solution on Grid or Point              #
    # ----------------------------------------------------------------------- #
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

    # ----------------------------------------------------------------------- #
    #                    Return Callable or Evaluated Ansatz                  #
    # ----------------------------------------------------------------------- #
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