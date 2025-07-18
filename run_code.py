# ======================================================
# MODULE IMPORTS
# ======================================================

import utils.auxiliary as aux  # Utility functions: lambdify exact solution, L2 error norm, plotting utilities
import setting.config as cfg  # Configuration parameters: beam length, coefficients, mesh size, time steps
from tests.test_solns1 import Testcase1  # Symbolic test case using known analytical Timoshenko beam solution
from solver.timoshenko_solver import TimoshenkoModelSolver  # Galerkin solver implementation for Timoshenko beam


# ======================================================
# INITIALIZE TEST CASE AND RETRIEVE INITIAL DATA
# ======================================================

solns1 = Testcase1()  # Instantiate symbolic solution case (with known analytical formulas)

# Retrieve all necessary initial and boundary conditions from base class
f1, f2, u0, u1, v0, v1, du0, du1, dv0, dv1 = solns1.get_initial_data()


# ======================================================
# RUN SOLVER AND COMPARE IF EXACT SOLUTION IS KNOWN
# ======================================================
if solns1.known_solutions:

    # --------------------------------------------------
    # STEP 1: Instantiate Solver with Config Parameters
    # --------------------------------------------------
    solver = TimoshenkoModelSolver(
        ell=cfg.ell,          # Beam length
        T=cfg.T,              # Total time
        alpha=cfg.alpha, beta=cfg.beta,
        gamma=cfg.gamma, delta=cfg.delta,
        a1=cfg.a1, a2=cfg.a2, # Coupling coefficients
        n=cfg.n,              # Time discretization steps
        N=cfg.N,              # Spatial discretization steps (basis functions)
        f1=f1, f2=f2,         # Source terms for u and v
        u0=u0, u1=u1,         # Displacement at t = 0 and t = τ
        v0=v0, v1=v1,         # Rotation at t = 0 and t = τ
        du0=du0, du1=du1,     # ∂u/∂x at t = 0 and t = τ
        dv0=dv0, dv1=dv1      # ∂v/∂x at t = 0 and t = τ
    )

    # solver.solve()  # <- Solver execution would go here if implemented

    # --------------------------------------------------
    # FUNCTION: Select Symbolic Solution (u or v)
    # --------------------------------------------------
    def select_solution_function(solution_type: str) -> callable:
        """
        Returns the analytical solution function for a given type ('u' or 'v').

        Parameters
        ----------
        solution_type : str
            One of 'u' (displacement) or 'v' (rotation)

        Returns
        -------
        callable
            Exact analytical function u(x, t) or v(x, t)
        """
        if solution_type == 'u':
            return solns1.u
        elif solution_type == 'v':
            return solns1.v
        else:
            raise ValueError("`solution_type` must be 'u' or 'v'.")

    # --------------------------------------------------
    # FUNCTION: Compute and Report L2 Errors Over Time
    # --------------------------------------------------
    def compute_and_report_L2_errors() -> dict:
        """
        Computes L2 norm errors for u and v over all time steps and plots the results.

        Returns
        -------
        dict
            Dictionary containing L2 error lists and the path to the plot file.
        """
        L2_errors = {}

        # Loop over solution types ('u' and 'v')
        for sol_type in ['u', 'v']:
            exact_func = aux.callable_exact_solution(select_solution_function(sol_type), solver)
            approx_func = solver.callable_compute_ansatz(sol_type)

            L2_errors[f"L2_error_{sol_type}"] = aux.compute_L2_error(
                exact_func, approx_func, solver.ell
            )

        # Display L2 errors for each time step
        for sol_type in ['u', 'v']:
            print(f"\n--- L2 Error for solution '{sol_type}' ---")
            for k, err in enumerate(L2_errors[f"L2_error_{sol_type}"]):
                print(f"Time step {k:3d}: L2 error = {err:.6e}")

        # Plot L2 error evolution over time
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

    # Run L2 error computation and get results
    results = compute_and_report_L2_errors()

    # --------------------------------------------------
    # STEP 3: Plot Exact vs Approximate Solution at Final Time Layer
    # --------------------------------------------------
    time_layer = solver.n  # Final time step index

    for sol_type in ['u', 'v']:
        path = aux.plot_exact_vs_approx_solution_at_time_k(
            exact_soln=select_solution_function(sol_type),
            approx_solver=solver,
            solution_type=sol_type,
            time_layer=time_layer,
            config=solver
        )
        print(f"Saved comparison plot for '{sol_type}' at time layer {time_layer}: {path}")

    # Clean up temporary references
    del path, results, sol_type, time_layer