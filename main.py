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
import numpy as np  # Core numerical array and math support
import utils.auxiliary as aux  # Utility functions: L2 error computation, callable wrappers, plotting
from utils.class_timoshenko_solns import TimoshenkoSolutions  # Class containing exact analytical solutions
import utils.config as cfg  # Simulation and model parameters
from utils.class_timoshenko import TimoshenkoModelSolver  # Galerkin method solver for Timoshenko system

# ---------------------------------------------------------------
# FLAG: Toggle use of known analytical benchmark solutions
# ---------------------------------------------------------------
known_solutions = True

# ---------------------------------------------------------------
# STEP 1: LOAD SOLUTION DATA (Exact or Benchmark Inputs)
# ---------------------------------------------------------------
solns = TimoshenkoSolutions(known_solutions=known_solutions)
f1, f2, u0, u1, v0, v1, du0, du1, dv0, dv1 = solns.get_initial_data()

# ---------------------------------------------------------------
# STEP 2: INITIALIZE THE GALERKIN SOLVER
# ---------------------------------------------------------------
solver = TimoshenkoModelSolver(
    ell=cfg.ell,          # Domain length
    T=cfg.T,              # Final simulation time
    alpha=cfg.alpha,      # Displacement damping coefficient
    beta=cfg.beta,        # Shear damping coefficient
    gamma=cfg.gamma,      # Rotational stiffness
    delta=cfg.delta,      # Rotational damping coefficient
    a1=cfg.a1,            # Coupling: ∂v/∂x in u-equation
    a2=cfg.a2,            # Coupling: ∂u/∂x in v-equation
    n=cfg.n,              # Number of time steps
    N=cfg.N,              # Number of spatial Galerkin basis modes
    f1=f1, f2=f2,         # External forcing functions
    u0=u0, u1=u1,         # Initial displacement and velocity
    v0=v0, v1=v1,         # Initial rotation and rotational velocity
    du0=du0, du1=du1,     # Initial spatial derivatives of displacement
    dv0=dv0, dv1=dv1      # Initial spatial derivatives of rotation
)

# ---------------------------------------------------------------
# IF ANALYTICAL SOLUTIONS ARE KNOWN: EVALUATE ERROR & PLOT
# ---------------------------------------------------------------
if known_solutions:

    def select_solns(solution_type: str) -> callable:
        """
        Return analytical solution u(x, t) or v(x, t) based on type.
        
        Parameters
        ----------
        solution_type : str
            'u' for displacement, 'v' for rotation
        
        Returns
        -------
        callable
            Function (x, t) -> exact solution value
        """
        if solution_type == 'u':
            return solns.u
        elif solution_type == 'v':
            return solns.v
        else:
            raise ValueError("`solution_type` must be 'u' or 'v'.")

    def compute_and_report_L2_errors(aux, solver, cfg, select_solns_func) -> dict:
        """
        Compute, print, and plot L2 errors over time for both solution components.
        
        Parameters
        ----------
        aux : module
            Utilities for exact/numerical solution evaluation and plotting
        solver : TimoshenkoModelSolver
            Solver containing numerical approximations
        cfg : module
            Configuration with time, space, and model parameters
        select_solns_func : function
            Returns exact solution function for a given type ('u' or 'v')
        
        Returns
        -------
        dict
            {
                "L2_error_u": np.ndarray of errors over time,
                "L2_error_v": np.ndarray of errors over time,
                "plot_file" : path to saved PDF plot
            }
        """
        L2_errors = {}

        for soln_type in ['u', 'v']:
            L2_errors[f"L2_error_{soln_type}"] = aux.compute_L2_error(
                aux.callable_exact_solution(func=select_solns_func(soln_type)),
                solver.callable_compute_ansatz(soln_type),
                cfg.ell
            )

        # Print time-resolved error data
        for soln_type in ['u', 'v']:
            print(f"\n--- L2 Error for solution '{soln_type}' ---")
            for k, err in enumerate(L2_errors[f"L2_error_{soln_type}"]):
                print(f"Time step {k:3d}: L2 error = {err:.6e}")

        # Plot combined figure
        plot_file = aux.plot_L2_errors_over_time(
            cfg.t, L2_errors["L2_error_u"], L2_errors["L2_error_v"], cfg
        )
        print(f"\nCombined error plot saved to: {plot_file}")

        return {
            "L2_error_u": L2_errors["L2_error_u"],
            "L2_error_v": L2_errors["L2_error_v"],
            "plot_file": plot_file
        }

    # Execute L2 error analysis
    results = compute_and_report_L2_errors(aux, solver, cfg, select_solns)
    
    for sol_type in ['u', 'v']:
        path = aux.plot_exact_vs_approx_solution_at_time_k(
            exact_soln=select_solns(sol_type),
            approx_solver=solver,
            solution_type=sol_type,
            time_layer=2,
            config=cfg
        )
        print(f"Saved plot for {sol_type}: {path}")

else:
    # ---------------------------------------------------------------
    # If no analytical solution exists, simulation still proceeds.
    # No L2 error comparison will be performed.
    # ---------------------------------------------------------------
    print("No exact solutions available; skipping L2 error comparison.")