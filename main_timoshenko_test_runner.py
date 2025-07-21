# ======================================================
# MODULE IMPORTS
# ======================================================

import utils.auxiliary as aux        # Utility functions for lambdification, L2 error computation, and plotting
import setting.config_test0 as cfg0  # Configuration parameters for Testcase 0
import setting.config_test1 as cfg1  # Configuration parameters for Testcase 1
from tests.test0 import Testcase0    # Predefined test case with known analytic solutions: test0
from tests.test1 import Testcase1    # Predefined test case with known analytic solutions: test1
from solver.timoshenko_solver import TimoshenkoModelSolver  # Galerkin solver for Timoshenko beam PDE system


# ======================================================
# TEST CASE SELECTOR
# ======================================================

# Dictionary mapping test identifiers to their respective configurations and test case objects
test_selector = {
    "test0": lambda: (cfg0, Testcase0(cfg0)),
    "test1": lambda: (cfg1, Testcase1(cfg1)),
    # Extend with additional test cases as needed
    # e.g., "test2": lambda: (cfg2, Testcase2(cfg2)),
}


# ======================================================
# SELECT AND INITIALIZE TEST CASE
# ======================================================

# Select a test case by its key
test_name = "test1"

# Validate test name and retrieve configuration and test case instance
if test_name not in test_selector:
    raise ValueError(f"Unknown test name: {test_name}. Available tests: {list(test_selector.keys())}")

# Retrieve config object and instantiated test case
cfg, test = test_selector[test_name]()  # This will load cfg0 and Testcase0(cfg0) for "test0"


# ======================================================
# INITIALIZE TEST CASE AND RETRIEVE INITIAL DATA
# ======================================================

# Symbolic source terms, initial and boundary conditions
f1, f2, u0, u1, v0, v1, du0, du1, dv0, dv1 = test.get_initial_data()


# ======================================================
# EXECUTE SOLVER PIPELINE (only if exact solutions exist)
# ======================================================

if test.known_solutions:

    # --------------------------------------------------
    # STEP 1: Instantiate Galerkin Solver for Timoshenko Beam
    # --------------------------------------------------

    # Construct the solver using test parameters and boundary/initial data
    solver = aux.named(  # Attaches a name to the solver instance for logging and plotting
        test.name,
        TimoshenkoModelSolver(
            ell=cfg.ell,         # Beam length
            T=cfg.T,             # Simulation duration
            alpha=cfg.alpha,     # Material parameter α: elasticity
            beta=cfg.beta,       # Material parameter β: stiffness nonlinearity
            gamma=cfg.gamma,     # Material parameter γ: rotational inertia
            delta=cfg.delta,     # Material parameter δ: damping/restoring
            a1=cfg.a1,           # Coupling coefficient from v to u
            a2=cfg.a2,           # Coupling coefficient from u to v
            n=cfg.n,             # Number of time steps
            N=cfg.N,             # Number of spatial basis functions
            f1=f1, f2=f2,        # Load/source terms
            u0=u0, u1=u1,        # Initial displacement at t=0 and t=τ
            v0=v0, v1=v1,        # Initial rotation at t=0 and t=τ
            du0=du0, du1=du1,    # Derivative of u (∂u/∂x) at t=0 and t=τ (initial condition)
            dv0=dv0, dv1=dv1     # Derivative of v (∂v/∂x) at t=0 and t=τ (initial condition)
        )
    )

    # Uncomment below to perform the actual numerical solution
    # solver.solve()


    # --------------------------------------------------
    # FUNCTION: select_solution_function
    # --------------------------------------------------
    # Purpose:
    #   Retrieve the exact symbolic solution function for displacement (u) or rotation (v)

    def select_solution_function(solution_type: str) -> callable:
        """
        Returns the exact symbolic solution for u(x, t) or v(x, t).

        Parameters
        ----------
        solution_type : str
            'u' for displacement or 'v' for rotation.

        Returns
        -------
        callable
            Symbolic solution function u(x, t) or v(x, t).
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
    #   Calculate and report the L2 error norms for u and v,
    #   display error per time step, and generate a diagnostic plot.

    def compute_and_report_L2_errors() -> dict:
        """
        Compute the L2 norm error over time for both solution components.
        Generates a combined error plot and prints per-step error diagnostics.

        Returns
        -------
        dict
            Contains L2 errors and path to plot file.
        """
        L2_errors = {}

        for sol_type in ['u', 'v']:
            # Create callable versions of the exact and numerical solutions
            exact_func = aux.callable_exact_solution(select_solution_function(sol_type), solver)
            approx_func = solver.callable_compute_ansatz(sol_type)

            # Compute L2 error across spatial domain for each time step
            L2_errors[f"L2_error_{sol_type}"] = aux.compute_L2_error(
                exact_func, approx_func, solver.ell
            )

        # Print L2 error per time step
        for sol_type in ['u', 'v']:
            print(f"\n--- L2 Error for solution '{sol_type}' ---")
            for k, err in enumerate(L2_errors[f"L2_error_{sol_type}"]):
                print(f"Time step {k:3d}: L2 error = {err:.6e}")

        # Plot L2 errors across all time steps
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

    # Execute L2 error computation and capture output
    results = compute_and_report_L2_errors()


    # --------------------------------------------------
    # STEP 3: Compare Final Time Layer with Exact Solution
    # --------------------------------------------------

    # Index of the final time step
    time_layer = solver.n

    # Generate comparison plots for u(x, T) and v(x, T)
    for sol_type in ['u', 'v']:
        path = aux.plot_exact_vs_approx_solution_at_time_k(
            exact_soln=select_solution_function(sol_type),
            approx_solver=solver,
            solution_type=sol_type,
            time_layer=time_layer,
            config=solver  # Used for metadata in plot
        )
        print(f"Saved comparison plot for '{sol_type}' at time layer {time_layer}: {path}")

    # Clean up temporary variables (optional in scripts)
    del path, results, sol_type, test_name, time_layer, test_selector