# ======================================================
# MODULE IMPORTS
# ======================================================

# Utilities: provides helper functions for plotting, lambdification, and error metrics
import utils.auxiliary as aux

# Configuration modules: define model parameters and physical constants for each test scenario
import setting.config_test0 as cfg0  # Configuration settings for Testcase 0
import setting.config_test1 as cfg1  # Configuration settings for Testcase 1
import setting.config_test2 as cfg2  # Configuration settings for Testcase 2
import setting.config_test3 as cfg3  # Configuration settings for Testcase 3

# Test case definitions: symbolic benchmark problems with known analytical solutions
from tests.test0 import Testcase0    # Analytic test case with constant or simple functions
from tests.test1 import Testcase1    # Slightly more complex benchmark
from tests.test2 import Testcase2    # Trigonometric benchmark including nonlinear effects
from tests.test3 import Testcase3    # Oscillatory benchmark using sinusoids in space and time

# Solver for the Timoshenko beam PDE system using a Galerkin spectral method
from solver.timoshenko_solver import TimoshenkoModelSolver


# ======================================================
# FUNCTIONAL BLOCK: TEST CASE SELECTOR DICTIONARY
# ======================================================
# Maps a string identifier to a tuple of (configuration, testcase object).
# This structure allows easy test switching by key.

test_selector = {
    "test0": lambda: (cfg0, Testcase0(cfg0)),
    "test1": lambda: (cfg1, Testcase1(cfg1)),
    "test2": lambda: (cfg2, Testcase2(cfg2)),
    "test3": lambda: (cfg3, Testcase3(cfg3)),
    # Add additional test cases here in the format: "testX": lambda: (cfgX, TestcaseX(cfgX))
}


# ======================================================
# FUNCTIONAL BLOCK: SELECT AND VALIDATE TEST CASE
# ======================================================

# Specify the test case name to run (change as needed)
test_name = "test3"

# Ensure the test name is valid
if test_name not in test_selector:
    raise ValueError(
        f"Unknown test name: '{test_name}'. "
        f"Available options: {list(test_selector.keys())}"
    )

# Dynamically retrieve the config and test class instance
cfg, test = test_selector[test_name]()  # For "test2", returns (cfg2, Testcase2(cfg2))


# ======================================================
# FUNCTIONAL BLOCK: LOAD INITIAL & BOUNDARY DATA
# ======================================================
# Pull symbolic source terms and initial conditions for the Timoshenko system.
# These are used to construct the PDE system and verify numerical accuracy.

# Returned symbolic expressions:
#   f1(t, x)     : Source term for displacement equation (u)
#   f2(t, x)     : Source term for rotation equation (v)
#   u0(x)        : Initial displacement u(x, t=0)
#   u1(x)        : Displacement at t = τ (used for initialization)
#   v0(x)        : Initial rotation v(x, t=0)
#   v1(x)        : Rotation at t = τ (used for initialization)
#   du0(x)       : ∂u/∂x at t=0 (displacement gradient at initial time)
#   du1(x)       : ∂u/∂x at t=τ
#   dv0(x)       : ∂v/∂x at t=0 (rotation gradient at initial time)
#   dv1(x)       : ∂v/∂x at t=τ

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