# ======================================================
# MODULE IMPORTS
# ======================================================

import utils.auxiliary as aux  # Utility tools: symbolic lambdification, error computation, visualization

# Configuration modules: contain model parameters for each benchmark test
import setting.config_test0 as cfg0  # Configuration for Testcase 0
import setting.config_test1 as cfg1  # Configuration for Testcase 1
import setting.config_test2 as cfg2  # Configuration for Testcase 2
import setting.config_test3 as cfg3  # Configuration for Testcase 3

# Benchmark test classes: provide symbolic solutions to validate the solver
from tests.test0 import Testcase0    # Basic analytic test case
from tests.test1 import Testcase1    # Slightly more advanced benchmark
from tests.test2 import Testcase2    # Trigonometric benchmark with nonlinear terms
from tests.test3 import Testcase3    # Oscillatory benchmark with spatial-temporal modes

# Numerical solver for Timoshenko beam equations (Galerkin spectral method)
from solver.timoshenko_solver import TimoshenkoModelSolver


# ======================================================
# METHOD: TEST CASE SELECTOR DICTIONARY
# ======================================================
# Maps string identifiers (e.g., "test0") to a tuple:
# (config object, benchmark test class instance)
# This supports easy switching between different test scenarios.

test_selector = {
    "test0": lambda: (cfg0, Testcase0(cfg0)),
    "test1": lambda: (cfg1, Testcase1(cfg1)),
    "test2": lambda: (cfg2, Testcase2(cfg2)),
    "test3": lambda: (cfg3, Testcase3(cfg3)),
    # Add new test entries here as needed
}


# ======================================================
# METHOD: SELECT AND VALIDATE TEST CASE
# ======================================================
# Choose and validate a test case by name

test_name = "test2"  # Change this to "test0", "test1", etc., to run a different benchmark

if test_name not in test_selector:
    raise ValueError(
        f"Unknown test name: {test_name}. "
        f"Available options: {list(test_selector.keys())}"
    )

# Load configuration and test instance dynamically
cfg, test = test_selector[test_name]()  # Example: ("test2") → (cfg2, Testcase2(cfg2))


# ======================================================
# METHOD: LOAD INITIAL CONDITIONS FROM TEST CASE
# ======================================================
# Extract symbolic initial and source terms for the PDE system
# These functions are needed to construct right-hand sides and apply initial/boundary conditions

# Returned expressions:
#   f1, f2   : Source terms for u-equation and v-equation
#   u0       : Displacement u(x, t=0)
#   u1       : Initial velocity ∂u/∂t at t=τ
#   v0       : Rotation v(x, t=0)
#   v1       : Initial angular velocity ∂v/∂t at t=τ
#   du0      : Spatial derivative ∂u/∂x at x in [0, ℓ] when t=0
#   du1      : Spatial derivative ∂u/∂x at x in [0, ℓ] when t=τ
#   dv0      : Spatial derivative ∂v/∂x at x in [0, ℓ] when t=0
#   dv1      : Spatial derivative ∂v/∂x at x in [0, ℓ] when t=τ

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
                exact_func, approx_func, solver.ell
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