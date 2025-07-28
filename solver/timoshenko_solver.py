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

# --------------------------------------------------------------------------- #
#                              Imported Modules                               #
# --------------------------------------------------------------------------- #

import numpy as np            # Core library for numerical operations and arrays
import time                   # Used for measuring elapsed computation time
from tqdm import tqdm         # Provides progress bar during time-stepping loop
import utils.auxiliary as aux # Custom module: contains Galerkin projection and solver utilities

# --------------------------------------------------------------------------- #
#                    Timoshenko Nonlinear Beam Solver Class                   #
# --------------------------------------------------------------------------- #

class TimoshenkoModelSolver:
    """
    Nonlinear PDE solver for the Timoshenko system using a Galerkin framework.
    Projects PDE system onto a Legendre basis and evolves in time with leapfrog integration.
    """

    # ----------------------------------------------------------------------- #
    #                         Class Initialization Method                     #
    # ----------------------------------------------------------------------- #
    def __init__(
        self,
        ell, T,                     # Spatial domain length and final time
        alpha, beta, gamma, delta,  # PDE coefficients
        a1, a2,                     # Coupling constants between u and v
        n, N,           # Number of time steps, number of basis functions
        f1, f2,         # Forcing terms (functions of space and time)
        u0, u1, v0, v1, # Initial displacement and rotation states
        du0=None, du1=None, dv0=None, dv1=None,  # Optional: known first spatial derivatives
        h=1e-3,                # Step size for finite difference approximation
        derivmeth='nd',        # Derivative computation method ('nd' or 'sfd')
        tol=1e-6,              # Tolerance for adaptive Gauss quadrature
        min_dx=1/128.0,        # Minimum subinterval size for adaptive quadrature
        n_gauss=5,             # Initial Gauss nodes per subinterval
        max_gauss=50,          # Max Gauss nodes allowed in adaptive quadrature
        
        # Boolean flag indicating whether analytical solutions are known (True) or not provided (False)
        known_solutions=False, # Default: analytical solutions are not provided
        
        # Boolean flag to enable Kahan–Babuška–Neumaier (KBN) compensated summation
        # for improved numerical precision in modal solution reconstruction.
        use_kahan_sum=False  # Default is False (standard summation).
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
        
        # Flag indicating availability of analytical solutions
        self.known_solutions = known_solutions  # Boolean flag indicating whether analytical solutions are known (True) or not provided (False)
        
        # Store flag controlling use of Kahan-Babuška-Neumaier summation in solution reconstruction.
        self.use_kahan_sum = use_kahan_sum  # Enables consistent access to this flag throughout the class methods
        
        # Precomputed constant used in the v-equation update
        self.a0 = 4.0 / (2.0 + self.delta * self.tau**2)

        # Solve system and store modal coefficients and diagnostic data
        self.tilde_u, self.tilde_v, self.cond_u, self.cond_v, self.q_integr = self.solve_system()

    # ----------------------------------------------------------------------- #
    #                   Solve Modal Galerkin System Method                    #
    # ----------------------------------------------------------------------- #
    def solve_system(self):
        """
        Solves the Galerkin-projected system of time-dependent ODEs using a
        leapfrog integrator and returns modal coefficients and diagnostics.

        Returns:
        --------
        tuple:
            - tilde_u (ndarray): Modal coefficients of displacement
            - tilde_v (ndarray): Modal coefficients of rotation
            - cond_u (ndarray): Condition numbers of u-matrix per time step
            - cond_v (ndarray): Condition numbers of v-matrix per time step
            - q_integr (list): Nonlinear coefficients over time steps
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

        # Project time-dependent forcing terms onto the modal basis
        print("Projecting time-dependent forcing terms onto the modal basis.")
        
        if self.known_solutions:
            print("Analytical solutions are provided; computing the right-hand side projections via integration by parts.")
            # -------------- Projection: RHS for u-equation (f₁) --------------
            """
            Given the strong form:
                f₁(x, t) = ∂²u/∂t² - (α + β ∫(∂u/∂x)² dx) ∂²u/∂x² + a₁ ∂v/∂x
        
            The weak form (after integration by parts) becomes:
                (f₁, φₘ) = (∂²u/∂t², φₘ)
                         + ((α + β ∫(∂u/∂x)² dx) ∂u/∂x - a₁ v, P̂ₘ)
        
            Let us define the following terms:
                1. self.f1[0] represents ∂²u/∂t²                    — acceleration term, projected against φₘ
                2. self.f1[1] represents (α + β ∫(∂u/∂x)² dx) ∂u/∂x — nonlinear term, projected against P̂ₘ
                3. self.f1[2] represents a₁ v                       — coupling term, projected against P̂ₘ
        
            Consequently, the combined nonlinear and coupling term is:
                nonlinear_and_coupling = (α + β ∫(∂u/∂x)² dx) ∂u/∂x - a₁ v
            """
            
            nonlinear_and_coupling = lambda x, t_val: (
                self.f1[1](x, t_val) - self.f1[2](x, t_val)
            )
            f1_integr = (
                aux.compute_time_dependent_integrals(
                    self.f1[0],  # ∂²u/∂t²
                    self.N,
                    self.ell,
                    self.t,
                    multiplier="galerkin_basis",
                    **quad_kwargs
                )
                +
                aux.compute_time_dependent_integrals(
                    nonlinear_and_coupling,  # Combined nonlinear and -a₁ v term
                    self.N,
                    self.ell,
                    self.t,
                    multiplier="norm_leg_poly",
                    **quad_kwargs
                )
            )
        
            # -------------- Projection: RHS for v-equation (f₂) --------------
            """
            Given the strong form:
                f₂(x, t) = ∂²v/∂t² - γ ∂²v/∂x² + δ v - a₂ ∂u/∂x
        
            The weak form (after integration by parts) becomes:
                (f₂, φₘ) = (∂²v/∂t² + δ v, φₘ) + (γ ∂v/∂x + a₂ u, P̂ₘ)
        
            We define the terms as follows:
                1. self.f2[0] represents ∂²v/∂t²    — acceleration term, projected against φₘ
                2. self.f2[1] represents γ ∂v/∂x    — stiffness term, projected against P̂ₘ
                3. self.f2[2] represents δ v(x, t)  — mass-like term, projected against φₘ
                4. self.f2[3] represents a₂ u(x, t) — coupling term, projected against P̂ₘ
            """
            
            f2_integr = aux.compute_time_dependent_integrals(
                lambda x, t: self.f2[0](x, t) + self.f2[2](x, t),  # ∂²v/∂t² + δ v
                self.N,
                self.ell,
                self.t,
                multiplier="galerkin_basis",
                **quad_kwargs
            )
        
            f2_integr += aux.compute_time_dependent_integrals(
                lambda x, t: self.f2[1](x, t) + self.f2[3](x, t),  # γ ∂v/∂x + a₂ u
                self.N,
                self.ell,
                self.t,
                multiplier="norm_leg_poly",
                **quad_kwargs
            )
        else:
            print("Analytical solutions are unavailable; computing the right-hand side projections directly from the given source terms.")
            """
            For f₁(x, t) and f₂(x, t), the following expressions are assumed:
        
                1. f₁(x, t) = ∂²u/∂t² - (α + β ∫(∂u/∂x)² dx) ∂²u/∂x² + a₁ ∂v/∂x
                2. f₂(x, t) = ∂²v/∂t² - γ ∂²v/∂x² + δ v - a₂ ∂u/∂x
        
            We compute the corresponding weak form projections:
                1. (f₁, φₘ) = (∂²u/∂t² - (α + β ∫(∂u/∂x)² dx) ∂²u/∂x² + a₁ ∂v/∂x, φₘ)
                2. (f₂, φₘ) = (∂²v/∂t² - γ ∂²v/∂x² + δ v - a₂ ∂u/∂x, φₘ)
            """
            f1_integr = aux.compute_time_dependent_integrals(
                self.f1, self.N, self.ell,
                self.t, multiplier="galerkin_basis", **quad_kwargs
            )
            f2_integr = aux.compute_time_dependent_integrals(
                self.f2, self.N, self.ell,
                self.t, multiplier="galerkin_basis", **quad_kwargs
            )
        
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
        
        # Precompute time-stepping constants
        const_u = 4.0 / self.ell**2
        const_v = 2.0 * self.a0 / self.ell**2
        const_rhs = self.tau**2 / self.ell
        
        for k in tqdm(range(self.n - 1), desc="Solving Timoshenko system", unit="step"):
            # Compute right-hand side (RHS) for linear systems at time step k
            if k == 0:
                # Conducting the first step: uses projected ICs at t=0, t=τ (special handling)
                b1 = const_u * (
                    self.tau**2 * (f1_integr[k] + 0.5 * q_prev * diff2u[k] - self.a1 * diff1v1)
                    + 2.0 * u1_integr - u0_integr
                )
                b2 = const_v * (
                    self.tau**2 * (f2_integr[k] + self.a2 * diff1u1 + 0.5 * (self.gamma * diff2v[k] - self.delta * v0_integr))
                    + 2.0 * v1_integr - v0_integr
                )
            elif k == 1:
                # For the second step: uses Galerkin stencils from the previous step
                b1 = const_u * (
                    self.tau**2 * (
                        f1_integr[k] + 0.5 * (q_prev * diff2u[k] - self.a1 * self.ell * aux.galerkin_stencils(self.N, tild_v[k - 1], operator="first-order"))
                    )
                    + 0.5 * self.ell**2 * aux.galerkin_stencils(self.N, tild_u[k - 1]) - u1_integr
                )
                b2 = const_v * (
                    self.tau**2 * (
                        f2_integr[k] + 0.5 * (self.a2 * self.ell * aux.galerkin_stencils(self.N, tild_u[k - 1], operator="first-order")
                                              + self.gamma * diff2v[k] - self.delta * v1_integr)
                    )
                    + 0.5 * self.ell**2 * aux.galerkin_stencils(self.N, tild_v[k - 1]) - v1_integr
                )
            else:
                # All later steps use fully recursive leapfrog stencils
                b1 = 2.0 * (
                    const_rhs * ((2.0 / self.ell) * f1_integr[k] - self.a1 * aux.galerkin_stencils(self.N, tild_v[k - 1], operator="first-order"))
                    + aux.galerkin_stencils(self.N, tild_u[k - 1])
                )
                b2 = self.a0 * (
                    const_rhs * ((2.0 / self.ell) * f2_integr[k] + self.a2 * aux.galerkin_stencils(self.N, tild_u[k - 1], operator="first-order"))
                    + aux.galerkin_stencils(self.N, tild_v[k - 1])
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
        
        # Notify when computation is finished
        print(f"Computation has been completed in {time.time() - start_time:.2f} seconds.")
        
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
        ----------------------------------------------------------------
        Return Callable or Evaluated Galerkin Approximate Solution u(x, t_k) or v(x, t_k)
        ----------------------------------------------------------------
        Generates callable function(s) representing the Galerkin approximate solution 
        for displacement (u) or rotation (v) at a specific time step (k).
        Optionally evaluates at specified spatial points (x_vals). Summation method is 
        controlled by the instance flag `self.use_kahan_sum`:
          - True:  use Kahan–Babuška–Neumaier (compensated) summation 
          - False: use standard Python summation (default) 
        
        Parameters
        ----------
        solution_type : str
            'u' for displacement or 'v' for rotation.
        k : int, optional
            Time step index (0 ≤ k ≤ n). If None, returns for all steps.
        x_vals : float | int | list | np.ndarray, optional
            Points at which to evaluate. If None, returns callables.
        
        Returns
        -------
        Callable or np.ndarray
            - If k and x_vals are None: list of callables, one per time step.
            - If k is specified and x_vals is None: single callable at step k.
            - If x_vals is specified: evaluated numpy array or scalar.
        """

        # ----------------------------------------
        # Input validation: Ensure solution_type is correct
        # ----------------------------------------
        if solution_type not in {'u', 'v'}:
            raise ValueError("solution_type must be 'u' or 'v'.")
    
        # ----------------------------------------
        # Helper: Normalize and validate x_vals input
        # ----------------------------------------
        def validate_and_convert_x_vals(x_input):
            """Ensure x_vals is converted to float or NumPy array."""
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
    
        # ----------------------------------------
        # Helper: Generate Galerkin basis functions φₘ(x)
        # ----------------------------------------
        def generate_basis():
            """
            Returns a list of φₘ(x) basis function callables.
            Each φₘ(x) depends on index m, the domain length ell, and position x.
            """
            return [
                (lambda m: (lambda x: aux.phi_m(m, self.ell, x)))(m + 1)
                for m in range(self.N)
            ]
    
        # ----------------------------------------
        # Helper: Construct ansatz u(x,t_k) or v(x,t_k) at time step k
        # ----------------------------------------
        def construct_function_at_k(k_idx: int):
            """
            Returns a callable Galerkin solution at time step k_idx.
            Applies initial condition for k=0 or k=1, or reconstructs
            solution from coefficients and basis for k≥2.
            """
            if k_idx == 0:
                return self.u0 if solution_type == 'u' else self.v0
            elif k_idx == 1:
                return self.u1 if solution_type == 'u' else self.v1
    
            # Select coefficients and generate basis functions
            coeffs = self.tilde_u[k_idx - 2] if solution_type == 'u' else self.tilde_v[k_idx - 2]
            basis = generate_basis()
    
            # Construct the ansatz function u(x) or v(x)
            def ansatz_function(x):
                # Handle vectorized evaluation
                if isinstance(x, np.ndarray):
                    return np.array([
                        aux.kahan_babuska_neumaier_sum([c * phi(xi) for c, phi in zip(coeffs, basis)])
                        if self.use_kahan_sum else
                        sum(c * phi(xi) for c, phi in zip(coeffs, basis))
                        for xi in x
                    ])
                else:
                    # Scalar evaluation
                    terms = [c * phi(x) for c, phi in zip(coeffs, basis)]
                    return aux.kahan_babuska_neumaier_sum(terms) if self.use_kahan_sum else sum(terms)
    
            return ansatz_function
    
        # ----------------------------------------
        # Main return logic
        # ----------------------------------------
        if k is not None:
            # Validate time step k
            if not isinstance(k, int) or not (0 <= k <= self.n):
                raise ValueError(f"k = {k} must be an integer in range [0, {self.n}].")
    
            # Build and return function or evaluated result at time step k
            fn = construct_function_at_k(k)
            return fn if x_vals is None else fn(x_vals)
    
        # If k is None, return functions or evaluated results at all time steps
        functions = [construct_function_at_k(k_idx) for k_idx in range(self.n + 1)]
        return functions if x_vals is None else np.array([fn(x_vals) for fn in functions])