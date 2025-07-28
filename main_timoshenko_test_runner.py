# ======================================================
# MODULE IMPORTS
# ======================================================

import utils.auxiliary as aux        # Utility functions: lambdify solutions, compute L2 errors, plotting, etc.
import setting.config_test0 as cfg0  # Configuration: parameters specific to Testcase 0
import setting.config_test1 as cfg1  # Configuration: parameters specific to Testcase 1
import setting.config_test2 as cfg2  # Configuration: parameters specific to Testcase 2

from tests.test0 import Testcase0    # Analytic test case with known solution: Testcase0
from tests.test1 import Testcase1    # Analytic test case with known solution: Testcase1
from tests.test2 import Testcase2    # Analytic test case with known solution: Testcase2

from solver.timoshenko_solver import TimoshenkoModelSolver  # Galerkin method solver for Timoshenko beam PDE system


# ======================================================
# TEST CASE SELECTOR
# ======================================================

# Dictionary: maps test case name to a lambda function returning (config, testcase instance)
test_selector = {
    "test0": lambda: (cfg0, Testcase0(cfg0)),
    "test1": lambda: (cfg1, Testcase1(cfg1)),
    "test2": lambda: (cfg2, Testcase2(cfg2)),
    # Add further test cases here as needed
}


# ======================================================
# SELECT AND INITIALIZE TEST CASE
# ======================================================

# Choose the test case by name
test_name = "test2"

# Validate test name and extract config and testcase object
if test_name not in test_selector:
    raise ValueError(f"Unknown test name: {test_name}. Available tests: {list(test_selector.keys())}")

# Unpack configuration and testcase instance from selector
cfg, test = test_selector[test_name]()  # For "test0", returns cfg0 and Testcase0(cfg0)


# ======================================================
# INITIALIZE TEST CASE AND RETRIEVE INITIAL DATA
# ======================================================

# Obtain symbolic source terms, initial and boundary conditions from testcase
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