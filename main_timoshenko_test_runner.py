# ======================================================
# MODULE IMPORTS
# ======================================================

# Utility functions: callable generation, L2 error computation, plotting, etc.
import utils.auxiliary as aux

# Configuration for different test cases
import setting.config_test0 as cfg0  # Mesh and physical parameters for Testcase 0
import setting.config_test1 as cfg1  # Mesh and physical parameters for Testcase 1

# Test cases with known analytical solutions
from tests.test0 import Testcase0    # Test case 0: used for validation against known solution
from tests.test1 import Testcase1    # Test case 1: another validation case with known solution

# Numerical solver for Timoshenko beam model using Galerkin method
from solver.timoshenko_solver import TimoshenkoModelSolver


# ======================================================
# RUN MULTIPLE TEST CASES USING CONFIGURATION-LINKED LOOP
# ======================================================

# Pair each configuration file with its corresponding test case
configs = [cfg0, cfg1]
tests = [Testcase0, Testcase1]

# List to collect solver instances (optional, for reuse or analysis)
solutions = []

# ======================================================
# MAIN LOOP: Execute Solver Pipeline for Each Test Case
# ======================================================

for cfg, TestClass in zip(configs, tests):
    
    # --------------------------------------------------
    # STEP 1: Initialize Test Case
    # --------------------------------------------------
    test = TestClass(cfg)  # Create test case instance with its config

    # Extract symbolic functions: source terms, initial/boundary conditions
    f1, f2, u0, u1, v0, v1, du0, du1, dv0, dv1 = test.get_initial_data()
    
    # Proceed only if exact symbolic solutions are known
    if test.known_solutions:
        
        # --------------------------------------------------
        # STEP 2: Instantiate Galerkin Solver for Timoshenko Beam
        # --------------------------------------------------
        solver = aux.named(  # Attach test name for better tracking/logging
            test.name,
            TimoshenkoModelSolver(
                ell=cfg.ell,         # Length of the beam
                T=cfg.T,             # Total simulation time
                alpha=cfg.alpha,     # Material property: elasticity
                beta=cfg.beta,       # Material property: nonlinear stiffness
                gamma=cfg.gamma,     # Material property: rotational inertia
                delta=cfg.delta,     # Material property: damping
                a1=cfg.a1,           # Coupling coefficient from v to u
                a2=cfg.a2,           # Coupling coefficient from u to v
                n=cfg.n,             # Number of time layers
                N=cfg.N,             # Number of Galerkin basis functions
                f1=f1, f2=f2,        # Right-hand side terms for u and v
                u0=u0, u1=u1,        # Displacement u at t=0 and t=τ
                v0=v0, v1=v1,        # Rotation v at t=0 and t=τ
                du0=du0, du1=du1,    # ∂u/∂x at t=0 and t=τ
                dv0=dv0, dv1=dv1     # ∂v/∂x at t=0 and t=τ
            )
        )

        # Save solver instance for future reference or plotting
        solutions.append(solver)

        # --------------------------------------------------
        # STEP 3 (Optional): Solve the PDE system numerically
        # --------------------------------------------------
        # solver.solve()  # Uncomment if you wish to run the simulation step

        # --------------------------------------------------
        # FUNCTION: select_solution_function
        # --------------------------------------------------
        def select_solution_function(solution_type: str) -> callable:
            """
            Select symbolic exact solution for displacement or rotation.

            Parameters
            ----------
            solution_type : str
                'u' for displacement, 'v' for rotation.

            Returns
            -------
            callable
                Corresponding symbolic function u(x, t) or v(x, t).
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
        def compute_and_report_L2_errors() -> dict:
            """
            Compute and report the L2 error norms for u and v over time.

            Returns
            -------
            dict
                Dictionary with L2 error time series for u and v, and plot path.
            """
            L2_errors = {}

            # Loop over both solution types
            for sol_type in ['u', 'v']:
                exact_func = aux.callable_exact_solution(
                    select_solution_function(sol_type), solver
                )
                approx_func = solver.callable_compute_ansatz(sol_type)

                # Compute L2 error across space for each time step
                L2_errors[f"L2_error_{sol_type}"] = aux.compute_L2_error(
                    exact_func, approx_func, solver.ell
                )

            # Print L2 error for each time step
            for sol_type in ['u', 'v']:
                print(f"\n--- L2 Error for solution '{sol_type}' ---")
                for k, err in enumerate(L2_errors[f"L2_error_{sol_type}"]):
                    print(f"Time step {k:3d}: L2 error = {err:.6e}")

            # Plot and save the error time series
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

        # Compute errors and generate report/plot
        results = compute_and_report_L2_errors()

        # --------------------------------------------------
        # STEP 4: Plot Numerical vs Exact Solution at Final Time
        # --------------------------------------------------
        time_layer = solver.n  # Final time index (corresponding to T)

        for sol_type in ['u', 'v']:
            # Save side-by-side plot of numerical and exact solution
            path = aux.plot_exact_vs_approx_solution_at_time_k(
                exact_soln=select_solution_function(sol_type),
                approx_solver=solver,
                solution_type=sol_type,
                time_layer=time_layer,
                config=solver
            )
            print(f"Saved comparison plot for '{sol_type}' at time layer {time_layer}: {path}")

        # Log the completion of the current test case
        print(f"\n Completed all computations for test case: {test.name}\n{'-'*60}")
        
        # Clear unused vars to avoid conflicts or leaks
        del path, results, sol_type, time_layer

# ======================================================
# GLOBAL COMPLETION MESSAGE
# ======================================================

print("\n All test cases have been computed successfully!")
print(f"Total cases executed: {len(solutions)}")
print("Stored solver instances:", [solver.name for solver in solutions])
print("=====================================================\n")
