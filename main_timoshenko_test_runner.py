# ======================================================
# MODULE: main_timoshenko_test_runner.py
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
import utils.auxiliary as aux  # Local utilities: projection, norms, plotting, callable wrappers

# Dynamic registry for test cases:
# Maps string keys (e.g., "test3") to their associated configuration and Testcase class
from testcase_registry.registry import get_testcase  # Dispatcher: returns (cfg_module, testcase_instance)


# ======================================================
# SELECT TEST CASE TO LOAD
# ======================================================

# Set the name of the test case to run.
# Options: 'test0', 'test1', ..., 'test9' depending on availability.
test_name = "test10"  # <- change this to switch benchmarks

# Retrieve both the configuration module (cfg) and symbolic benchmark instance (test)
# - cfg: holds physical and numerical simulation parameters
# - test: contains symbolic PDE definitions for u(x,t), v(x,t), and source terms
cfg, test = get_testcase(test_name)  # Calls registry; lazy-instantiates the testcase


# ======================================================
# TIMOSHENKO SOLVER MODULE
# ======================================================
# Import the solver that numerically solves the Timoshenko beam equations:
# - Uses Legendre spectral Galerkin method in space
# - Time stepping scheme may be explicit or semi-implicit depending on the model
from solver.timoshenko_solver import TimoshenkoModelSolver  # Core numerical solver  <-- fixed import


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
f1, f2, u0, u1, v0, v1, du0, du1, dv0, dv1 = test.get_initial_data()  # Vectorized callables expected


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
    # === Method Title: select_solution_function — choose exact solution u or v
    def select_solution_function(solution_type: str) -> callable:
        """
        Return the exact symbolic solution function for the given component.

        Parameters
        ----------
        solution_type : str
            'u' for displacement, 'v' for rotation.

        Returns
        -------
        callable
            The symbolic function u(x, t) or v(x, t).

        Raises
        ------
        ValueError
            If solution_type is neither 'u' nor 'v'.
        """
        if solution_type == 'u':          # Branch for displacement solution
            return test.u                 # Exact u(x,t) from testcase
        elif solution_type == 'v':        # Branch for rotation solution
            return test.v                 # Exact v(x,t) from testcase
        else:
            # Defensive programming: explicit, informative error
            raise ValueError("`solution_type` must be either 'u' or 'v'.")


    # --------------------------------------------------
    # FUNCTION: compute_and_report_L2_errors
    # --------------------------------------------------
    # Purpose:
    #   Compute L2 norm errors for both u and v.
    #   Print error at each timestep and generate L2 error plots.
    # === Method Title: compute_and_report_L2_errors — compute + plot L2 errors over time
    def compute_and_report_L2_errors() -> dict:
        """
        Compute L2 error norms for displacement and rotation over all time steps.

        Returns
        -------
        dict
            Dictionary with error arrays and path to saved error plot:
            {
                "L2_error_u": np.ndarray,
                "L2_error_v": np.ndarray,
                "plot_file": str
            }
        """
        L2_errors = {}                       # Accumulator for both components
        sol_types = ('u', 'v')               # Small constant tuple to avoid magic strings

        for sol_type in sol_types:
            # Convert symbolic solution and solver approximation to callables
            exact_func = aux.callable_exact_solution(  # Wrap exact u/v for (x,t) evaluation on solver grid
                select_solution_function(sol_type),
                solver
            )
            approx_func = solver.callable_compute_ansatz(sol_type)  # Numerical approximation evaluator

            # Compute L2 error across all time steps (vector of length n+1 or n)
            L2_errors[f"L2_error_{sol_type}"] = aux.compute_L2_error(
                exact_func,        # Exact benchmark solution at each time step
                approx_func,       # Numerical approximation from Galerkin solver
                solver.ell,        # Spatial domain length used for integration limits
                **cfg.quad_kwargs  # Inject quadrature parameters (tol, min_dx, etc.) from config
            )

        # Print errors step-by-step for both components
        for sol_type in sol_types:
            print(f"\n--- L2 Error for solution '{sol_type}' ---")
            for k, err in enumerate(L2_errors[f"L2_error_{sol_type}"]):
                print(f"Time step {k:3d}: L2 error = {err:.6e}")  # Fixed-width index, scientific notation

        # Plot combined L2 error graph (returns output filepath)
        plot_file = aux.plot_L2_errors_over_time(
            solver.t,                        # Time grid from solver
            L2_errors["L2_error_u"],         # Error curve for u
            L2_errors["L2_error_v"],         # Error curve for v
            solver                           # Pass solver for metadata (naming, etc.)
        )
        print(f"\nCombined error plot saved to: {plot_file}")

        # Standardized return payload for programmatic use
        return {
            "L2_error_u": L2_errors["L2_error_u"],
            "L2_error_v": L2_errors["L2_error_v"],
            "plot_file": plot_file
        }

    # Run L2 error diagnostics
    results = compute_and_report_L2_errors()  # Dict of arrays + plot path


    # --------------------------------------------------
    # STEP 3: FINAL TIME LAYER COMPARISON PLOTS
    # --------------------------------------------------

    # Get final time step index (solver.n is the number of steps; final layer is n)
    time_layer = solver.n

    # Compare numerical vs exact solution at final time T for both u and v
    for sol_type in ('u', 'v'):
        path = aux.plot_exact_vs_approx_solution_at_time_k(
            exact_soln=select_solution_function(sol_type),  # exact u or v
            approx_solver=solver,                           # solver holding approximation
            solution_type=sol_type,                         # 'u' or 'v'
            time_layer=time_layer,                          # final index
            config=solver                                   # Pass solver for metadata
        )
        print(f"Saved comparison plot for '{sol_type}' at time layer {time_layer}: {path}")
    
    # Optional cleanup of used variables (helps keep REPL clean if importing this script)
    del f1, f2, path, results, sol_type, test_name, time_layer

else:
    # ===============================================================
    # TITLE: Convergence Analysis (No Exact Solution Available)
    # ===============================================================
    print("No exact solutions available; running convergence analysis...")

    # ---------------------------------------------------------------
    # TITLE: Tunable thresholds and refinement limits
    # ---------------------------------------------------------------
    tol = 1e-7                # L2-difference threshold for declaring convergence
    max_increment_n = 8        # Max time-refinement doublings: n -> 2^k * n_base
    max_galerkin_mode = 50     # Try up to 50 extra spatial modes (N_base+1 ... N_base+48)

    # Base resolutions drawn from selected configuration
    n_base = cfg.n             # Time steps at baseline resolution
    N_base = cfg.N             # Spatial modes at baseline resolution

    # Hard caps to keep runtime bounded
    n_limit = 4096             # Max allowed time steps
    N_limit = 45               # Max allowed spatial modes

    converged = False          # Global flag toggled on success

    # Accumulate rows; each row corresponds to a single tolerance violation
    # Columns: component ('u' or 'v'), n, N, layer (time index), norm (as string, .6e), tol (as string, .6e)
    failure_rows = []

    # ===============================================================
    # TITLE: Time refinement loop (progressively doubles n up to n_limit)
    # ===============================================================
    for n_incr in range(0, max_increment_n + 1):
        n_updt = (2 ** n_incr) * n_base   # Updated time resolution for this pass
        if n_updt > n_limit:
            break                         # Practical cap reached: stop refining time

        # -----------------------------------------------------------
        # TITLE: Baseline solution at fixed spatial resolution (N_base)
        # Purpose: compare refined-N solutions against this baseline
        # -----------------------------------------------------------
        solver_prev = aux.named(
            test.name,
            TimoshenkoModelSolver(
                ell=cfg.ell, T=cfg.T,
                alpha=cfg.alpha, beta=cfg.beta,
                gamma=cfg.gamma, delta=cfg.delta,
                a1=cfg.a1, a2=cfg.a2,
                n=n_updt, N=N_base,
                f1=f1, f2=f2,
                u0=u0, u1=u1,
                v0=v0, v1=v1,
                du0=du0, du1=du1,
                dv0=dv0, dv1=dv1,
                known_solutions=test.known_solutions
            )
        )

        # ===========================================================
        # TITLE: Spatial refinement loop (increment N up to N_limit)
        # ===========================================================
        for galerkin_mode in range(1, max_galerkin_mode + 1):
            N_updt = N_base + galerkin_mode
            if N_updt > N_limit:
                break  # Practical cap reached: stop refining space

            # Progress feedback for long runs
            print(f"Testing n = {n_updt}, N = {N_updt}.")

            # Current solver at (n_updt, N_updt) to compare with baseline (N_base)
            solver = aux.named(
                test.name,
                TimoshenkoModelSolver(
                    ell=cfg.ell, T=cfg.T,
                    alpha=cfg.alpha, beta=cfg.beta,
                    gamma=cfg.gamma, delta=cfg.delta,
                    a1=cfg.a1, a2=cfg.a2,
                    n=n_updt, N=N_updt,
                    f1=f1, f2=f2,
                    u0=u0, u1=u1,
                    v0=v0, v1=v1,
                    du0=du0, du1=du1,
                    dv0=dv0, dv1=dv1,
                    known_solutions=test.known_solutions
                )
            )

            # -------------------------------------------------------
            # TITLE: Check convergence for u-coefficients across time
            # -------------------------------------------------------
            u_converged = True
            for k in range(2, n_updt):  # Skip first two layers to avoid start-up transients
                norm_u = aux.compute_L2_difference_norms_from_coeffs(
                    coeff_init=solver_prev.tilde_u,  # baseline coefficients at N_base
                    coeff_next=solver.tilde_u,       # refined   coefficients at N_updt
                    config=solver_prev,              # grid/weights context for L2 difference
                    time_layer=k
                )
                if norm_u > tol:
                    # Console diagnostic (detailed and consistent formatting)
                    print(
                        f"[u] n={n_updt}, N={N_updt}, layer={k}: "
                        f"L2 diff norm = {norm_u:.6e} > tol = {tol:.6e}"
                    )
                    # Record failure for optional CSV logging on overall failure
                    failure_rows.append([
                        "u", n_updt, N_updt, k,
                        f"{norm_u:.6e}",   # store formatted norm (scientific, 6 decimals)
                        f"{tol:.6e}"       # store formatted tol  (scientific, 6 decimals)
                    ])
                    u_converged = False
                    break  # Keep behavior: stop at first u-failure for this (n_updt, N_updt)

            # -------------------------------------------------------
            # TITLE: Check convergence for v-coefficients (if u passed)
            # -------------------------------------------------------
            if u_converged:
                v_converged = True
                for k in range(2, n_updt):
                    norm_v = aux.compute_L2_difference_norms_from_coeffs(
                        coeff_init=solver_prev.tilde_v,
                        coeff_next=solver.tilde_v,
                        config=solver_prev,
                        time_layer=k
                    )
                    if norm_v > tol:
                        print(
                            f"[v] n={n_updt}, N={N_updt}, layer={k}: "
                            f"L2 diff norm = {norm_v:.6e} > tol = {tol:.6e}"
                        )
                        failure_rows.append([
                            "v", n_updt, N_updt, k,
                            f"{norm_v:.6e}",
                            f"{tol:.6e}"
                        ])
                        v_converged = False
                        break
            else:
                v_converged = False

            # -------------------------------------------------------
            # TITLE: Success path — both components converged
            # -------------------------------------------------------
            if u_converged and v_converged:
                print(f"\n Converged at n = {n_updt}, N = {N_updt}")
                converged = True

                # Snapshot plots of approximate solutions (optional diagnostics)
                print("Plotting approximate solution snapshots")
                for sol_type in ("u", "v"):
                    aux.plot_approx_solution_at_time_k(
                        approx_solver=solver,
                        solution_type=sol_type,
                        config=solver
                    )
                break  # Exit spatial loop early on success

            # Promote current solver to baseline for next N iteration
            solver_prev = solver

        # If success at this n_updt, stop time refinement
        if converged:
            break

    # ---------------------------------------------------------------
    # TITLE: Final reporting and CSV logging if convergence fails
    # ---------------------------------------------------------------
    if not converged:
        print(f"\n Convergence was not reached within the given limits: n ≤ {n_limit} and N ≤ {N_limit}.")

        # Prepare CSV path only on total failure (avoid clutter on success)
        from pathlib import Path       # Cross-platform, safe filesystem paths
        from datetime import datetime  # Timestamp for unique filenames
        import csv                     # Lightweight CSV writer (stdlib)

        run_name = getattr(test, "name", "run")                  # Use testcase name if available
        base_dir = Path("plots") / run_name / "convergence_logs" # Tidy output area
        base_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Filename embeds tested limits and timestamp for clarity and uniqueness
        csv_path = base_dir / f"convergence_failures_n{n_limit}_N{N_limit}_{timestamp}.csv"

        if failure_rows:
            with csv_path.open(mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["component", "n", "N", "layer", "norm", "tol"])
                writer.writerows(failure_rows)  # rows already carry formatted norm/tol
            print(f"\nSaved convergence failures to: {csv_path}")
        else:
            print("\nNo convergence failures to log (no rows written).")