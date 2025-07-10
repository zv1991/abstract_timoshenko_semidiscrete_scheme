# ===============================================================
# SCRIPT: Solve and Analyze Timoshenko Beam Model (Galerkin Method)
# ===============================================================
# Description:
#   - Loads benchmark data for nonlinear Timoshenko beam PDE
#   - Solves using Galerkin approximation
#   - Computes L2 error over time using exact solutions (if available)
#   - Otherwise, performs convergence study by comparing successive Galerkin approximations
# ===============================================================

# ---------------------------------------------------------------
# IMPORT REQUIRED MODULES AND CLASSES
# ---------------------------------------------------------------

import numpy as np  # Core numerical array operations

import utils.auxiliary as aux                      # Utility functions: L2 error, plotting, lambdification
import utils.config as cfg                         # Configuration constants: parameters, time grid, etc.
from utils.class_timoshenko_solns import TimoshenkoSolutions  # Provides benchmark initial/boundary data
from utils.class_timoshenko import TimoshenkoModelSolver      # Galerkin solver class for Timoshenko equations


# ---------------------------------------------------------------
# CONFIGURATION: Toggle for using exact solutions (if known)
# ---------------------------------------------------------------

known_solutions = True  # Set this to False to enable convergence testing

# ---------------------------------------------------------------
# STEP 1: LOAD INITIAL AND BOUNDARY DATA FROM SOLUTION CLASS
# ---------------------------------------------------------------

solns = TimoshenkoSolutions(known_solutions=known_solutions)

# Retrieve forcing functions, initial conditions, and spatial derivatives
f1, f2, u0, u1, v0, v1, du0, du1, dv0, dv1 = solns.get_initial_data()

# ---------------------------------------------------------------
# STEP 2: IF ANALYTICAL SOLUTIONS EXIST, PERFORM L2 ERROR ANALYSIS
# ---------------------------------------------------------------

if known_solutions:

    solver = TimoshenkoModelSolver(
        ell=cfg.ell, T=cfg.T,
        alpha=cfg.alpha, beta=cfg.beta,
        gamma=cfg.gamma, delta=cfg.delta,
        a1=cfg.a1, a2=cfg.a2,
        n=cfg.n, N=cfg.N,
        f1=f1, f2=f2,
        u0=u0, u1=u1,
        v0=v0, v1=v1,
        du0=du0, du1=du1,
        dv0=dv0, dv1=dv1
    )

    def select_solution_function(solution_type: str) -> callable:
        if solution_type == 'u':
            return solns.u
        elif solution_type == 'v':
            return solns.v
        else:
            raise ValueError("`solution_type` must be 'u' or 'v'.")

    def compute_and_report_L2_errors() -> dict:
        L2_errors = {}

        for sol_type in ['u', 'v']:
            exact_func = aux.callable_exact_solution(select_solution_function(sol_type))
            approx_func = solver.callable_compute_ansatz(sol_type)

            L2_errors[f"L2_error_{sol_type}"] = aux.compute_L2_error(
                exact_func, approx_func, solver.ell
            )

        for sol_type in ['u', 'v']:
            print(f"\n--- L2 Error for solution '{sol_type}' ---")
            for k, err in enumerate(L2_errors[f"L2_error_{sol_type}"]):
                print(f"Time step {k:3d}: L2 error = {err:.6e}")

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

    results = compute_and_report_L2_errors()

    time_layer = solver.n

    for sol_type in ['u', 'v']:
        path = aux.plot_exact_vs_approx_solution_at_time_k(
            exact_soln=select_solution_function(sol_type),
            approx_solver=solver,
            solution_type=sol_type,
            time_layer=time_layer,
            config=solver
        )
        print(f"Saved comparison plot for '{sol_type}' at time layer {time_layer}: {path}")

    del path, results, sol_type, time_layer

else:
    # ---------------------------------------------------------------
    # IF NO EXACT SOLUTION IS AVAILABLE, RUN CONVERGENCE ANALYSIS
    # ---------------------------------------------------------------
    print("No exact solutions available; running convergence analysis...")

    tol = 1e-6
    max_increment_n = 8
    max_galerkin_mode = 48

    n_base = cfg.n
    N_base = cfg.N

    for p in range(0, max_increment_n + 1):  # Time refinement
        n_updt = 2**p * n_base

        # Base solution at fixed spatial resolution
        solver_prev = TimoshenkoModelSolver(
            ell=cfg.ell, T=cfg.T,
            alpha=cfg.alpha, beta=cfg.beta,
            gamma=cfg.gamma, delta=cfg.delta,
            a1=cfg.a1, a2=cfg.a2,
            n=n_updt, N=N_base,
            f1=f1, f2=f2,
            u0=u0, u1=u1,
            v0=v0, v1=v1,
            du0=du0, du1=du1,
            dv0=dv0, dv1=dv1
        )

        for q in range(1, max_galerkin_mode + 1):  # Spatial refinement
            N_updt = N_base + q

            solver = TimoshenkoModelSolver(
                ell=cfg.ell, T=cfg.T,
                alpha=cfg.alpha, beta=cfg.beta,
                gamma=cfg.gamma, delta=cfg.delta,
                a1=cfg.a1, a2=cfg.a2,
                n=n_updt, N=N_updt,
                f1=f1, f2=f2,
                u0=u0, u1=u1,
                v0=v0, v1=v1,
                du0=du0, du1=du1,
                dv0=dv0, dv1=dv1
            )

            u_converged = True
            for k in range(n_updt):
                norm_u = aux.compute_L2_difference_norms_from_coeffs(
                    coeff_init=solver_prev.tilde_u,
                    coeff_next=solver.tilde_u,
                    time_layer=k
                )
                if norm_u > tol:
                    u_converged = False
                    break

            if u_converged:
                v_converged = True
                for k in range(n_updt):
                    norm_v = aux.compute_L2_difference_norms_from_coeffs(
                        coeff_init=solver_prev.tilde_v,
                        coeff_next=solver.tilde_v,
                        time_layer=k
                    )
                    if norm_v > tol:
                        v_converged = False
                        break
            else:
                v_converged = False

            if u_converged and v_converged:
                print(f"\n Converged at n = {n_updt}, N = {N_updt}")
                break  # Exit spatial loop

            solver_prev = solver  # Only if same n_updt

        if u_converged and v_converged:
            break  # Exit time loop

# ---------------------------------------------------------------
# FINAL CLEANUP: REMOVE FLAGS AND TEMPORARY OBJECTS
# ---------------------------------------------------------------
del known_solutions

u_vals = solver.callable_compute_ansatz(solution_type='u', k=solver.n, x_vals=1.0)
print(u_vals)
v_vals = solver.callable_compute_ansatz(solution_type='v', k=solver.n, x_vals=1.0)
print(v_vals)