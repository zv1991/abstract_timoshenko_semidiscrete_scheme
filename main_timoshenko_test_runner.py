# ======================================================
# MODULE IMPORTS
# ======================================================

# Utility functions for plotting, lambdification, quadrature projection,
# and error computation (e.g., L2 norms, symbolic-to-numeric tools).
import utils.auxiliary as aux


# ======================================================
# CONFIGURATION MODULES — PHYSICAL & NUMERICAL PARAMETERS
# ======================================================
# These configuration modules define simulation parameters for each test case.
# They include:
#   - Physical coefficients (α, β, γ, δ, a₁, a₂)
#   - Domain geometry (length ℓ, total time T)
#   - Discretization (number of basis functions N, time steps n, τ)
#   - Benchmark-specific oscillation or polynomial structure

import setting.config_test0 as cfg0  # cfg0: Testcase0 — Simple constant/linear solutions (debug baseline)
import setting.config_test1 as cfg1  # cfg1: Testcase1 — Smooth sinusoidal solution with moderate dynamics
import setting.config_test2 as cfg2  # cfg2: Testcase2 — Nonlinear system with sinusoidal forcing
import setting.config_test3 as cfg3  # cfg3: Testcase3 — Oscillatory benchmark using sinusoids in x and t
import setting.config_test4 as cfg4  # cfg4: Testcase4 — Variant of Testcase3 with tunable wave frequency
import setting.config_test5 as cfg5  # cfg5: Testcase5 — Legendre spatial basis with polynomial-in-time solutions


# ======================================================
# BENCHMARK SOLUTION CLASSES — SYMBOLIC PDE DEFINITIONS
# ======================================================
# These test case classes define symbolic expressions for:
#   - Displacement field u(x, t)
#   - Rotation field v(x, t)
#   - Source terms f₁(t, x), f₂(t, x)
#   - Initial conditions and spatial gradients
# Used for verifying numerical accuracy via method of manufactured solutions.

from tests.test0 import Testcase0  # Basic constant/polynomial field (sanity check)
from tests.test1 import Testcase1  # Mild trigonometric benchmark with temporal variation
from tests.test2 import Testcase2  # Sinusoidal solution with nonlinearity in u-equation
from tests.test3 import Testcase3  # Oscillatory test with spatial and temporal sine waves
from tests.test4 import Testcase4  # Testcase3 variant with parameterized spatial frequencies
from tests.test5 import Testcase5  # Analytical test using Legendre spatial modes and time polynomials


# ======================================================
# TIMOSHENKO SOLVER MODULE
# ======================================================
# Solves the nonlinear Timoshenko beam equations using:
#   - Spectral Galerkin discretization (Legendre basis)
#   - Explicit or semi-implicit time integration
from solver.timoshenko_solver import TimoshenkoModelSolver


# ======================================================
# TEST CASE DISPATCH DICTIONARY
# ======================================================
# Maps string test identifiers to corresponding configuration module
# and benchmark test class. Enables flexible switching between benchmarks.

test_selector = {
    "test0": lambda: (cfg0, Testcase0(cfg0)),
    "test1": lambda: (cfg1, Testcase1(cfg1)),
    "test2": lambda: (cfg2, Testcase2(cfg2)),
    "test3": lambda: (cfg3, Testcase3(cfg3)),
    "test4": lambda: (cfg4, Testcase4(cfg4)),
    "test5": lambda: (cfg5, Testcase5(cfg5)),
    # Add new test cases here as needed:
    # "test6": lambda: (cfg6, Testcase6(cfg6)),
}


# ======================================================
# SELECT TEST CASE TO RUN
# ======================================================
# Modify this to choose which test configuration to simulate.

test_name = "test1"  # Options: "test0", "test1", ..., "test5"

# Validate that the test name is supported
if test_name not in test_selector:
    raise ValueError(
        f"Unknown test name: '{test_name}'. "
        f"Available options: {list(test_selector.keys())}"
    )

# Retrieve the configuration and symbolic test case instance
cfg, test = test_selector[test_name]()  # e.g. "test2" → (cfg2, Testcase2(cfg2))


# ======================================================
# INITIAL AND BOUNDARY DATA EXTRACTION
# ======================================================
# These symbolic expressions define the right-hand side of the PDE and initial conditions.
# They are used to:
#   - Initialize the solver
#   - Project source terms and exact solutions onto spectral basis
#   - Evaluate solver accuracy

# Extracted symbolic functions:
# f1(t, x)     — source term for displacement u(x, t)
# f2(t, x)     — source term for rotation v(x, t)
# u0(x)        — displacement at initial time t = 0
# u1(x)        — displacement at next time step t = τ
# v0(x)        — rotation at initial time t = 0
# v1(x)        — rotation at next time step t = τ
# du0(x)       — spatial derivative ∂u/∂x at t = 0
# du1(x)       — spatial derivative ∂u/∂x at t = τ
# dv0(x)       — spatial derivative ∂v/∂x at t = 0
# dv1(x)       — spatial derivative ∂v/∂x at t = τ

f1, f2, u0, u1, v0, v1, du0, du1, dv0, dv1 = test.get_initial_data()

# ======================================================
# EXECUTE SOLVER PIPELINE (only if exact solutions exist)
# ======================================================

if test.known_solutions:

    # --------------------------------------------------
    # STEP 1: INSTANTIATE GALERKIN SOLVER
    # --------------------------------------------------

    # Construct solver with problem parameters and input functions
    solver = aux.named(  # Name-tag the solver (useful for logging/plot filenames)
        test.name,
        TimoshenkoModelSolver(
            ell=cfg.ell,         # Beam length (domain [0, ℓ])
            T=cfg.T,             # Final simulation time
            alpha=cfg.alpha,     # Elasticity parameter (α)
            beta=cfg.beta,       # Stiffness nonlinearity parameter (β)
            gamma=cfg.gamma,     # Rotational inertia (γ)
            delta=cfg.delta,     # Damping/restoring parameter (δ)
            a1=cfg.a1,           # Coupling from v to u
            a2=cfg.a2,           # Coupling from u to v
            n=cfg.n,             # Number of time steps
            N=cfg.N,             # Number of spatial basis functions
            f1=f1, f2=f2,        # External forcing terms (symbolic)
            u0=u0, u1=u1,        # Initial displacements at t=0 and t=τ
            v0=v0, v1=v1,        # Initial rotations at t=0 and t=τ
            du0=du0, du1=du1,    # Initial spatial derivatives of u
            dv0=dv0, dv1=dv1,    # Initial spatial derivatives of v
            h=1e-3,              # Finite difference step for derivatives (if needed)
            derivmeth='nd',      # Derivative method: 'nd' (NumDiff) or 'sfd' (Standard Finite Difference)
            tol=cfg.quad_kwargs['tol'],              # Gauss integration tolerance
            min_dx=cfg.quad_kwargs['min_dx'],        # Minimum subdivision size for integration
            n_gauss=cfg.quad_kwargs['n_gauss'],      # Initial number of Gauss nodes
            max_gauss=cfg.quad_kwargs['max_gauss'],  # Maximum number of Gauss nodes
            known_solutions=test.known_solutions     # Whether analytic solution is known (True/False)
        )
    )

    # Optional: Uncomment to run the simulation
    # solver.solve()


    # --------------------------------------------------
    # FUNCTION: select_solution_function
    # --------------------------------------------------
    # Purpose:
    #   Retrieve symbolic analytic solution function for displacement `u(x,t)` or rotation `v(x,t)`
    def select_solution_function(solution_type: str) -> callable:
        """
        Return exact symbolic solution function for given component.

        Parameters
        ----------
        solution_type : str
            'u' for displacement, 'v' for rotation.

        Returns
        -------
        callable
            The symbolic function u(x, t) or v(x, t).
        """
        if solution_type == 'u':
            return test.u
        elif solution_type == 'v':
            return test.v
        else:
            raise ValueError("`solution_type` must be either 'u' or 'v'.")


    # --------------------------------------------------
    # FUNCTION: compute_and_report_L2_errors
    # --------------------------------------------------
    # Purpose:
    #   Compute L2 norm errors for both u and v.
    #   Print error at each timestep and generate L2 error plots.

    def compute_and_report_L2_errors() -> dict:
        """
        Compute L2 error norms for displacement and rotation over all time steps.

        Returns
        -------
        dict
            Dictionary with error arrays and path to saved error plot.
        """
        L2_errors = {}

        for sol_type in ['u', 'v']:
            # Convert symbolic solution and solver approximation to callables
            exact_func = aux.callable_exact_solution(select_solution_function(sol_type), solver)
            approx_func = solver.callable_compute_ansatz(sol_type)

            # Compute L2 error across all time steps
            L2_errors[f"L2_error_{sol_type}"] = aux.compute_L2_error(
                exact_func,        # Exact benchmark solution at each time step
                approx_func,       # Numerical approximation from Galerkin solver
                solver.ell,        # Spatial domain length used for integration limits
                **cfg.quad_kwargs  # Inject quadrature parameters (tol, min_dx, etc.) from config
            )

        # Print errors step-by-step
        for sol_type in ['u', 'v']:
            print(f"\n--- L2 Error for solution '{sol_type}' ---")
            for k, err in enumerate(L2_errors[f"L2_error_{sol_type}"]):
                print(f"Time step {k:3d}: L2 error = {err:.6e}")

        # Plot combined L2 error graph
        plot_file = aux.plot_L2_errors_over_time(
            solver.t,
            L2_errors["L2_error_u"],
            L2_errors["L2_error_v"],
            solver
        )
        print(f"\nCombined error plot saved to: {plot_file}")

        return {
            "L2_error_u": L2_errors["L2_error_u"],
            "L2_error_v": L2_errors["L2_error_v"],
            "plot_file": plot_file
        }

    # Run L2 error diagnostics
    results = compute_and_report_L2_errors()


    # --------------------------------------------------
    # STEP 3: FINAL TIME LAYER COMPARISON PLOTS
    # --------------------------------------------------

    # Get final time step index
    time_layer = solver.n

    # Compare numerical vs exact solution at final time T for both u and v
    for sol_type in ['u', 'v']:
        path = aux.plot_exact_vs_approx_solution_at_time_k(
            exact_soln=select_solution_function(sol_type),
            approx_solver=solver,
            solution_type=sol_type,
            time_layer=time_layer,
            config=solver  # Pass solver for metadata
        )
        print(f"Saved comparison plot for '{sol_type}' at time layer {time_layer}: {path}")

    # Optional cleanup of used variables
    del f1, f2, path, results, sol_type, test_name, time_layer, test_selector