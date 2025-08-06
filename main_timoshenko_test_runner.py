# ======================================================
# MODULE: main.py (or driver.py)
# PURPOSE: Driver script to load a benchmark test case, configure
# the solver, and prepare initial/boundary symbolic expressions.
# ======================================================


# ======================================================
# MODULE IMPORTS
# ======================================================

# Auxiliary module with helper functions:
# - Projection of symbolic expressions onto basis functions
# - Norm computations (e.g., L2 error)
# - Lambdification and visualization utilities
import utils.auxiliary as aux

# Dynamic registry for test cases:
# Maps string keys (e.g. "test3") to their associated configuration and Testcase class
from testcase_registry.registry import get_testcase


# ======================================================
# SELECT TEST CASE TO LOAD
# ======================================================

# Set the name of the test case to run.
# Options: 'test0', 'test1', ..., 'test6' depending on availability.
test_name = "test7"

# Retrieve both the configuration module (cfg) and symbolic benchmark instance (test)
# - cfg: holds physical and numerical simulation parameters
# - test: contains symbolic PDE definitions for u(x,t), v(x,t), and source terms
cfg, test = get_testcase(test_name)


# ======================================================
# TIMOSHENKO SOLVER MODULE
# ======================================================
# Import the solver that numerically solves the Timoshenko beam equations:
# - Uses Legendre spectral Galerkin method in space
# - Time stepping scheme may be explicit or semi-implicit depending on model

from solver.timoshenko_solver import TimoshenkoModelSolver


# ======================================================
# INITIAL AND BOUNDARY DATA EXTRACTION
# ======================================================
# Symbolic expressions derived from the selected Testcase are extracted here.
# These functions define:
# - Forcing terms (f1, f2)
# - Initial conditions at t = 0 and t = τ
# - Spatial derivatives at both time steps (for computing nonlinear terms)

# Unpack symbolic components from the test object:
# - f1(t,x), f2(t,x): source terms for u and v equations
# - u0(x), u1(x): initial and first-step values of displacement u
# - v0(x), v1(x): initial and first-step values of rotation v
# - du0(x), du1(x): ∂u/∂x at t = 0 and t = τ
# - dv0(x), dv1(x): ∂v/∂x at t = 0 and t = τ

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
            # h=1e-3,              # Finite difference step for derivatives (if needed)
            # derivmeth='nd',      # Derivative method: 'nd' (NumDiff) or 'sfd' (Standard Finite Difference)
            # tol=cfg.quad_kwargs['tol'],              # Gauss integration tolerance
            # min_dx=cfg.quad_kwargs['min_dx'],        # Minimum subdivision size for integration
            # n_gauss=cfg.quad_kwargs['n_gauss'],      # Initial number of Gauss nodes
            # max_gauss=cfg.quad_kwargs['max_gauss'],  # Maximum number of Gauss nodes
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
                exact_func,        # Exact benchmark solution at each time step
                approx_func,       # Numerical approximation from Galerkin solver
                solver.ell,        # Spatial domain length used for integration limits
                **cfg.quad_kwargs  # Inject quadrature parameters (tol, min_dx, etc.) from config
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
    del f1, f2, path, results, sol_type, test_name, time_layer
else:
    # ---------------------------------------------------------------
    # IF NO EXACT SOLUTION IS AVAILABLE, RUN CONVERGENCE ANALYSIS
    # ---------------------------------------------------------------
    print("No exact solutions available; running convergence analysis...")

    tol = 1e-2
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
                    config=solver_prev,
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
                        config=solver_prev,
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