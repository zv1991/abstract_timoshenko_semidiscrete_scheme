# ======================================================
# MODULE IMPORTS
# ======================================================

import utils.auxiliary as aux  # Utility module: contains lambdification, L2 norm calculation, and plotting
import setting.config_test1 as cfg  # Global configuration file for beam and simulation parameters
from tests.test1 import Testcase1  # Symbolic test case with known analytical solution
from solver.timoshenko_solver import TimoshenkoModelSolver  # Galerkin solver implementation


# ======================================================
# INITIALIZE TEST CASE AND RETRIEVE INITIAL DATA
# ======================================================

# Instantiate the test case by passing the config (no implicit dependency on global cfg)
test1 = Testcase1(cfg)

# Retrieve required initial and boundary condition data: functions at t=0 and t=τ
f1, f2, u0, u1, v0, v1, du0, du1, dv0, dv1 = test1.get_initial_data()


# ======================================================
# RUN SOLVER AND POSTPROCESS IF EXACT SOLUTION IS KNOWN
# ======================================================

if test1.known_solutions:

    # --------------------------------------------------
    # STEP 1: Instantiate the Galerkin Solver
    # --------------------------------------------------
    solver = TimoshenkoModelSolver(
        ell=cfg.ell,       # Beam length
        T=cfg.T,           # Total simulation time
        alpha=cfg.alpha,   # Model coefficient α (elasticity term)
        beta=cfg.beta,     # Model coefficient β (nonlinear stiffness)
        gamma=cfg.gamma,   # Model coefficient γ (rotational inertia)
        delta=cfg.delta,   # Model coefficient δ (damping or restoring term)
        a1=cfg.a1,         # Coupling coefficient from v to u
        a2=cfg.a2,         # Coupling coefficient from u to v
        n=cfg.n,           # Number of time steps
        N=cfg.N,           # Number of spatial basis functions
        f1=f1, f2=f2,      # Right-hand side terms for u and v
        u0=u0, u1=u1,      # Displacement u at t=0 and t=τ
        v0=v0, v1=v1,      # Rotation v at t=0 and t=τ
        du0=du0, du1=du1,  # ∂u/∂x at t=0 and t=τ
        dv0=dv0, dv1=dv1   # ∂v/∂x at t=0 and t=τ
    )

    # Optionally invoke solver (commented for demonstration)
    # solver.solve()

    # --------------------------------------------------
    # METHOD: Select Symbolic Solution (u or v)
    # --------------------------------------------------
    def select_solution_function(solution_type: str) -> callable:
        """
        Returns the symbolic exact solution function u(x, t) or v(x, t).

        Parameters
        ----------
        solution_type : str
            One of 'u' (displacement) or 'v' (rotation)

        Returns
        -------
        callable
            Corresponding analytical solution function
        """
        if solution_type == 'u':
            return test1.u
        elif solution_type == 'v':
            return test1.v
        else:
            raise ValueError("`solution_type` must be 'u' or 'v'.")

    # --------------------------------------------------
    # METHOD: Compute and Report L2 Errors
    # --------------------------------------------------
    def compute_and_report_L2_errors() -> dict:
        """
        Computes the L2 error norm for both u and v over all time layers
        and generates plots for visual diagnostics.

        Returns
        -------
        dict
            Dictionary containing L2 error sequences and plot path.
        """
        L2_errors = {}

        for sol_type in ['u', 'v']:
            # Get callable functions: exact vs. approximate
            exact_func = aux.callable_exact_solution(select_solution_function(sol_type), solver)
            approx_func = solver.callable_compute_ansatz(sol_type)

            # Compute L2 error over time
            L2_errors[f"L2_error_{sol_type}"] = aux.compute_L2_error(
                exact_func, approx_func, solver.ell
            )

        # Display error evolution for both u and v
        for sol_type in ['u', 'v']:
            print(f"\n--- L2 Error for solution '{sol_type}' ---")
            for k, err in enumerate(L2_errors[f"L2_error_{sol_type}"]):
                print(f"Time step {k:3d}: L2 error = {err:.6e}")

        # Generate combined error plot
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

    # Run error analysis
    results = compute_and_report_L2_errors()

    # --------------------------------------------------
    # STEP 3: Plot Exact vs Approximate at Final Time
    # --------------------------------------------------
    time_layer = solver.n  # Final time index

    for sol_type in ['u', 'v']:
        path = aux.plot_exact_vs_approx_solution_at_time_k(
            exact_soln=select_solution_function(sol_type),
            approx_solver=solver,
            solution_type=sol_type,
            time_layer=time_layer,
            config=solver
        )
        print(f"Saved comparison plot for '{sol_type}' at time layer {time_layer}: {path}")

    # Optional cleanup of loop variables
    del path, results, sol_type, time_layer