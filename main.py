# ===============================================================
# SCRIPT: Solve and Analyze Timoshenko Beam Model (Galerkin Method)
# ===============================================================
# Description:
#   - Loads benchmark data for nonlinear Timoshenko beam PDE
#   - Solves using Galerkin approximation
#   - Computes L2 error over time using exact solutions (if available)
#   - Exports a LaTeX-styled plot of error curves (PDF)
# ===============================================================

# ---------------------------------------------------------------
# IMPORT REQUIRED MODULES AND CLASSES
# ---------------------------------------------------------------
import numpy as np  # Core numerical array operations and math support

import utils.auxiliary as aux  # Utility functions: error computation, callable converters, plotting
import utils.config as cfg  # Simulation parameters and model configuration
from utils.class_timoshenko_solns import TimoshenkoSolutions  # Exact analytical solution provider
from utils.class_timoshenko import TimoshenkoModelSolver  # Galerkin solver class for the Timoshenko beam

# ---------------------------------------------------------------
# CONFIGURATION FLAG: Use analytical benchmark solutions (if available)
# ---------------------------------------------------------------
known_solutions = False

# ---------------------------------------------------------------
# STEP 1: LOAD INITIAL AND BOUNDARY DATA FROM SOLUTION CLASS
# ---------------------------------------------------------------
solns = TimoshenkoSolutions(known_solutions=known_solutions)
f1, f2, u0, u1, v0, v1, du0, du1, dv0, dv1 = solns.get_initial_data()

# ---------------------------------------------------------------
# STEP 2: INITIALIZE THE GALERKIN SOLVER WITH MODEL PARAMETERS
# ---------------------------------------------------------------
solver = TimoshenkoModelSolver(
    ell=cfg.ell,          # Beam length
    T=cfg.T,              # Final time
    alpha=cfg.alpha,      # Damping coefficient for displacement
    beta=cfg.beta,        # Damping coefficient for rotation
    gamma=cfg.gamma,      # Rotational stiffness
    delta=cfg.delta,      # Rotational damping
    a1=cfg.a1,            # Coupling in u-equation (∂v/∂x)
    a2=cfg.a2,            # Coupling in v-equation (∂u/∂x)
    n=cfg.n,              # Number of time steps
    N=cfg.N,              # Number of Galerkin modes (spatial)
    f1=f1, f2=f2,         # External forcing terms
    u0=u0, u1=u1,         # Initial displacement and velocity
    v0=v0, v1=v1,         # Initial rotation and rotational velocity
    du0=du0, du1=du1,     # Initial ∂u/∂x and ∂u/∂t
    dv0=dv0, dv1=dv1      # Initial ∂v/∂x and ∂v/∂t
)

# ---------------------------------------------------------------
# STEP 3: IF ANALYTICAL SOLUTIONS EXIST, PERFORM L2 ERROR ANALYSIS
# ---------------------------------------------------------------
if known_solutions:

    def select_solution_function(solution_type: str) -> callable:
        """
        Return the analytical solution u(x, t) or v(x, t).
        
        Parameters
        ----------
        solution_type : str
            'u' for displacement, 'v' for rotation
        
        Returns
        -------
        callable
            Function of form (x, t) -> float
        """
        if solution_type == 'u':
            return solns.u
        elif solution_type == 'v':
            return solns.v
        else:
            raise ValueError("`solution_type` must be 'u' or 'v'.")

    def compute_and_report_L2_errors() -> dict:
        """
        Compute and print the L2 errors for numerical vs analytical solutions,
        and generate plots of error evolution.
        
        Returns
        -------
        dict
            Dictionary containing L2 error arrays and output plot path.
        """
        L2_errors = {}

        for sol_type in ['u', 'v']:
            exact_func = aux.callable_exact_solution(select_solution_function(sol_type))
            approx_func = solver.callable_compute_ansatz(sol_type)

            L2_errors[f"L2_error_{sol_type}"] = aux.compute_L2_error(
                exact_func, approx_func, cfg.ell
            )

        # Report errors
        for sol_type in ['u', 'v']:
            print(f"\n--- L2 Error for solution '{sol_type}' ---")
            for k, err in enumerate(L2_errors[f"L2_error_{sol_type}"]):
                print(f"Time step {k:3d}: L2 error = {err:.6e}")

        # Plot combined L2 error figure
        plot_file = aux.plot_L2_errors_over_time(
            cfg.t,
            L2_errors["L2_error_u"],
            L2_errors["L2_error_v"],
            cfg
        )
        print(f"\nCombined error plot saved to: {plot_file}")

        return {
            "L2_error_u": L2_errors["L2_error_u"],
            "L2_error_v": L2_errors["L2_error_v"],
            "plot_file": plot_file
        }

    # Execute error computation and plot results
    results = compute_and_report_L2_errors()

    # Plot numerical vs exact solutions at a selected time layer (e.g., time index 2)
    for sol_type in ['u', 'v']:
        path = aux.plot_exact_vs_approx_solution_at_time_k(
            exact_soln=select_solution_function(sol_type),
            approx_solver=solver,
            solution_type=sol_type,
            time_layer=2,
            config=cfg
        )
        print(f"Saved comparison plot for '{sol_type}' at time layer 2: {path}")

    # Cleanup
    del path, results, sol_type

else:
    # ---------------------------------------------------------------
    # IF NO EXACT SOLUTION IS KNOWN, SIMULATE WITHOUT ERROR ANALYSIS
    # ---------------------------------------------------------------
    print("No exact solutions available; skipping L2 error comparison.")

# ---------------------------------------------------------------
# FINAL CLEANUP: REMOVE FLAGS AND TEMPORARY OBJECTS
# ---------------------------------------------------------------
del known_solutions

tilde_u = solver.tilde_u

L2_norms = aux.compute_L2_norm_galerkin_approx(
    approx_solution_generator=solver.callable_compute_ansatz('u'),
    ell=cfg.ell
    )

L2_norms_all = aux.compute_L2_norm_from_galerkin_coeffs(
    coeff=solver.tilde_u,
    ell=cfg.ell
    )

err1 = abs(L2_norms[2] - L2_norms_all[0])
print(err1)
err2 = abs(L2_norms[3] - L2_norms_all[1])
print(err2)