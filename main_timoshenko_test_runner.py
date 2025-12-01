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
# Options: 'test0', 'test1', ..., 'test13' depending on availability.
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
    # This branch is executed when no exact (analytical) solution is available
    # for the current test case. We perform a numerical convergence study
    # instead, refining time steps and spatial modes.
    print("No exact solutions available; running convergence analysis...")

    # ---------------------------------------------------------------
    # IMPORTS (LOCAL TO THIS BRANCH)
    # ---------------------------------------------------------------
    from pathlib import Path       # Path: cross-platform filesystem path helper (joins, mkdir, etc.)
    from datetime import datetime  # datetime: provides current time for timestamped filenames
    import csv                     # csv: standard library CSV reader/writer used to log convergence data

    # ---------------------------------------------------------------
    # TITLE: Tunable thresholds and refinement limits
    # ---------------------------------------------------------------
    tol = 1e-4                  # L2-difference tolerance: convergence is accepted when all norms ≤ tol
    max_increment_n = 8         # Max exponent for time refinement: n_updt = 2**n_incr * n_base
    max_galerkin_mode = 50      # Max number of extra spatial modes beyond N_base to test (1..50)

    # Base resolutions drawn from the selected configuration object `cfg`
    n_base = cfg.n              # Baseline number of time steps from configuration
    N_base = cfg.N              # Baseline number of spatial modes (Galerkin modes) from configuration

    # Hard caps to keep overall runtime and memory usage bounded
    n_limit = 4096              # Absolute upper limit on time steps (safety bound)
    N_limit = 45                # Absolute upper limit on spatial modes (safety bound)

    converged = False           # Global flag: set to True if both u and v converge for some (n, N)

    # Track the specific (n, N) pair where convergence is first detected
    best_n = None               # Records converged number of time steps, if found
    best_N = None               # Records converged number of spatial modes, if found

    # ===============================================================
    # CSV accumulation:
    #   1) Detailed norms at each time layer (csv_rows)
    #   2) Per-(n, N) *failure* events (event_rows)
    #
    # csv_rows columns (main CSV):
    #   component ('u' or 'v')
    #   n          (time steps)
    #   N          (spatial modes)
    #   layer      (time index, or "max" for per-case maximum)
    #   norm       (as string, .6e)
    #   tol        (as string, .6e)
    #   is_max     (0 for regular rows, 1 for "max" summary rows)
    #
    # event_rows columns (SECOND CSV, written only if convergence fails overall):
    #   component  ('u' or 'v')
    #   n          (time steps)
    #   N          (spatial modes)
    #   layer      (time index where the convergence test first fails)
    #   norm       (as string, .6e)
    #   status     ('fail' — always failure in this file)
    # ===============================================================
    csv_rows: list[list[object]] = []   # Detailed per-layer norms for main CSV
    event_rows: list[list[object]] = [] # Per-(n, N) failure events for secondary CSV

    # Pre-format tolerance once (used repeatedly in CSV rows)
    tol_str = f"{tol:.6e}"      # String representation of tolerance, written into CSV for traceability

    # ===============================================================
    # TITLE: Time refinement loop (progressively doubles n up to n_limit)
    # ===============================================================
    # We start from n_base time steps and progressively refine the time grid
    # by powers of 2: n_updt = 2**n_incr * n_base, until we hit n_limit or converge.
    for n_incr in range(max_increment_n + 1):
        n_updt = (2 ** n_incr) * n_base   # Updated time resolution for this pass

        if n_updt > n_limit:             # Stop refinement if we exceed the hard upper bound
            break                        # Practical cap reached: stop refining time

        # -----------------------------------------------------------
        # TITLE: Baseline solution at fixed spatial resolution (N_base)
        # Purpose: compare refined-N solutions against this baseline
        # -----------------------------------------------------------
        # aux.named(...): helper that wraps/labels the solver instance (e.g. for logging, caching).
        # TimoshenkoModelSolver: core PDE solver for the Timoshenko beam model.
        solver_prev = aux.named(
            test.name,                   # Name/label of the current test case
            TimoshenkoModelSolver(
                ell=cfg.ell, T=cfg.T,    # Domain length and final time from config
                alpha=cfg.alpha, beta=cfg.beta,
                gamma=cfg.gamma, delta=cfg.delta,
                a1=cfg.a1, a2=cfg.a2,    # Physical/material parameters
                n=n_updt, N=N_base,      # Time steps (refined) and baseline spatial modes
                f1=f1, f2=f2,            # Forcing terms for u and v
                u0=u0, u1=u1,            # Initial/boundary data for u
                v0=v0, v1=v1,            # Initial/boundary data for v
                du0=du0, du1=du1,        # Initial/boundary data for ∂u/∂t
                dv0=dv0, dv1=dv1,        # Initial/boundary data for ∂v/∂t
                known_solutions=test.known_solutions  # Optional exact solutions (for diagnostics)
            )
        )

        # ===========================================================
        # TITLE: Spatial refinement loop (increment N up to N_limit)
        # ===========================================================
        # For each fixed time resolution n_updt, increase the number of
        # spatial modes N one by one, starting from N_base + 1,
        # checking convergence against the baseline solver_prev.
        for galerkin_mode in range(1, max_galerkin_mode + 1):
            N_updt = N_base + galerkin_mode  # New number of spatial modes

            if N_updt > N_limit:             # Respect hard cap on spatial modes
                break                        # Practical cap reached: stop refining space

            # Progress feedback for potentially long runs
            print(f"Testing n = {n_updt}, N = {N_updt}.")

            # Current solver at (n_updt, N_updt) to compare with baseline at (n_updt, N_base)
            solver = aux.named(
                test.name,               # Same test name label for the refined solver
                TimoshenkoModelSolver(
                    ell=cfg.ell, T=cfg.T,
                    alpha=cfg.alpha, beta=cfg.beta,
                    gamma=cfg.gamma, delta=cfg.delta,
                    a1=cfg.a1, a2=cfg.a2,
                    n=n_updt, N=N_updt,  # Same time steps as baseline, but more spatial modes
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
            #       (and log all norms + per-case max)
            # -------------------------------------------------------
            u_converged = True           # Will be flipped to False if any time layer violates tolerance
            max_norm_u_case = 0.0        # Max L2 difference for u over time layers for this (n, N)
            max_norm_u_layer = None      # Time layer index where max_norm_u_case occurs

            # Iterate over time layers; skip first two layers to avoid initialization transients
            for k in range(2, n_updt):
                # aux.compute_L2_difference_norms_from_coeffs:
                #   Computes L2 norm of difference between two coefficient sets at a given time layer.
                norm_u = aux.compute_L2_difference_norms_from_coeffs(
                    coeff_init=solver_prev.tilde_u,  # u-coefficients for baseline (N_base)
                    coeff_next=solver.tilde_u,       # u-coefficients for refined (N_updt)
                    config=solver_prev,              # Use baseline solver for grid/weights context
                    time_layer=k                     # Time layer index to compare
                )

                # Update per-case maximum norm for u at this (n_updt, N_updt)
                if norm_u > max_norm_u_case:
                    max_norm_u_case = norm_u
                    max_norm_u_layer = k           # Store layer where the maximum is seen

                # Append detailed u data for this time layer to CSV rows
                csv_rows.append([
                    "u",                             # component identifier
                    n_updt,                          # time steps used
                    N_updt,                          # spatial modes used
                    k,                               # time layer index
                    f"{norm_u:.6e}",                 # L2 norm for u at this layer (scientific notation)
                    tol_str,                         # tolerance value as string
                    0                                # is_max = 0 (regular per-layer row)
                ])

                if norm_u > tol:
                    # Console diagnostic: explains why u failed convergence at this (n, N, k)
                    print(
                        f"[u] n={n_updt}, N={N_updt}, layer={k}: "
                        f"L2 diff norm = {norm_u:.6e} > tol = {tol:.6e}"
                    )
                    u_converged = False              # Mark u as non-converged for this (n, N) pair

                    # Record failure event for this (n_updt, N_updt) in event_rows
                    event_rows.append([
                        "u",                         # component
                        n_updt,                      # time steps
                        N_updt,                      # spatial modes
                        k,                           # layer where failure occurred
                        f"{norm_u:.6e}",             # failing norm value
                        "fail"                       # status: failure
                    ])

                    # Stop checking further time layers for u when first failure occurs
                    break

            # NOTE:
            # If the loop above never ran (very small n_updt), max_norm_u_layer could remain None
            # and max_norm_u_case would stay 0.0; we simply log that as-is.

            # After finishing or breaking the u-loop, log the per-case maximum norm for u
            csv_rows.append([
                "u",
                n_updt,
                N_updt,
                "max",                               # layer = "max" marks this as a summary row
                f"{max_norm_u_case:.6e}",           # maximum L2 norm across checked layers
                tol_str,
                1                                    # is_max = 1 indicates a per-case maximum summary
            ])

            # -------------------------------------------------------
            # TITLE: Check convergence for v-coefficients (if u passed)
            #       (and log all norms + per-case max)
            # -------------------------------------------------------
            if u_converged:
                # Only test v convergence if u has already converged for this (n, N)
                v_converged = True                  # Will be flipped on first violation
                max_norm_v_case = 0.0               # Maximum L2 difference for v over time layers
                max_norm_v_layer = None             # Time layer where max_norm_v_case occurs

                for k in range(2, n_updt):
                    # Compute L2 norm of difference between v-coefficients (baseline vs refined)
                    norm_v = aux.compute_L2_difference_norms_from_coeffs(
                        coeff_init=solver_prev.tilde_v,  # v-coefficients for baseline (N_base)
                        coeff_next=solver.tilde_v,       # v-coefficients for refined (N_updt)
                        config=solver_prev,              # same grid/weights as baseline
                        time_layer=k                     # time layer index being compared
                    )

                    # Update per-case maximum norm for v
                    if norm_v > max_norm_v_case:
                        max_norm_v_case = norm_v
                        max_norm_v_layer = k           # Store layer where maximum is seen

                    # Append detailed v data for this time layer to CSV rows
                    csv_rows.append([
                        "v",                             # component identifier
                        n_updt,
                        N_updt,
                        k,
                        f"{norm_v:.6e}",                 # L2 norm for v at this layer
                        tol_str,
                        0                                # is_max = 0 (regular row)
                    ])

                    if norm_v > tol:
                        # Console diagnostic for v divergence at this (n, N, k)
                        print(
                            f"[v] n={n_updt}, N={N_updt}, layer={k}: "
                            f"L2 diff norm = {norm_v:.6e} > tol = {tol:.6e}"
                        )
                        v_converged = False             # Mark v as non-converged

                        # Record failure event for this (n_updt, N_updt) in event_rows
                        event_rows.append([
                            "v",                         # component
                            n_updt,                      # time steps
                            N_updt,                      # spatial modes
                            k,                           # layer where failure occurred
                            f"{norm_v:.6e}",             # failing norm value
                            "fail"                       # status: failure
                        ])

                        # Stop checking further time layers for v when first failure occurs
                        break

                # After finishing or breaking the v-loop, log per-case maximum for v
                csv_rows.append([
                    "v",
                    n_updt,
                    N_updt,
                    "max",
                    f"{max_norm_v_case:.6e}",          # maximum L2 norm for v across layers
                    tol_str,
                    1                                   # is_max = 1 (summary row)
                ])
            else:
                # If u did not converge, v is considered non-converged as well for this (n, N)
                v_converged = False

            # -------------------------------------------------------
            # TITLE: Success path — both components converged
            # -------------------------------------------------------
            if u_converged and v_converged:
                # We declare overall convergence when both u and v meet the tolerance
                print(f"\n Converged at n = {n_updt}, N = {N_updt}")
                converged = True                      # Set global convergence flag
                best_n = n_updt                      # Record converged time resolution
                best_N = N_updt                      # Record converged spatial resolution

                # NOTE: By requirement, we DO NOT record success events in event_rows,
                # and we WILL NOT create an additional failures CSV when convergence
                # succeeds overall.

                # Snapshot plots of approximate solutions for diagnostics/visual confirmation
                print("Plotting approximate solution snapshots")
                for sol_type in ("u", "v"):
                    # aux.plot_approx_solution_at_time_k:
                    #   Creates a plot of the approximate solution for component 'u' or 'v'
                    #   at selected time layers using the solver's internal state.
                    aux.plot_approx_solution_at_time_k(
                        approx_solver=solver,         # Use converged solver instance
                        solution_type=sol_type,       # "u" or "v" component
                        config=solver                 # Solver also acts as its own plotting config
                    )

                break  # Exit spatial refinement loop early on success

            # Promote current refined solver to be the new baseline for the next N increment.
            # This lets us compare successive spatial refinements without recomputing
            # from N_base each time, which can be cheaper.
            solver_prev = solver

        # If convergence has been achieved at this time resolution, stop refining time.
        if converged:
            break

    # ---------------------------------------------------------------
    # TITLE: Final reporting and CSV logging (success or failure)
    # ---------------------------------------------------------------
    # At this point, either we have converged for some (n, N), or we hit limits
    # without convergence. In all cases, we write out csv_rows for post-analysis.
    # Additionally, we write event_rows as a second CSV ONLY IF convergence fails
    # overall, as per the requirement.

    # Human-readable status + console message
    if not converged:
        # Convergence did not succeed within the configured bounds
        print(
            f"\n Convergence was not reached within the given limits: "
            f"n ≤ {n_limit} and N ≤ {N_limit}."
        )
        status_tag = "failure"   # Used as a label in the output filenames
        # For failure, we record the limits in the filename as representative values
        n_rep = n_limit
        N_rep = N_limit
    else:
        # Convergence was achieved; report success
        print("\n Convergence achieved within the specified limits.")
        status_tag = "success"   # Label for success scenario
        # Use the actual converged (n, N) pair if recorded, otherwise fall back to limits
        n_rep = best_n if best_n is not None else n_limit
        N_rep = best_N if best_N is not None else N_limit

    # Base path for storing convergence log CSV files
    run_name = getattr(test, "name", "run")            # Use test.name if present, else default to "run"
    base_dir = Path("plots") / run_name / "convergence_logs"  # Directory for convergence logs
    base_dir.mkdir(parents=True, exist_ok=True)        # Ensure directory hierarchy exists

    # Generate a timestamp string to make filenames unique and sortable
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---------------------------------------------------------------
    # TITLE: Main CSV file with all per-layer norms (csv_rows)
    # ---------------------------------------------------------------
    # Filename includes convergence status, representative (n, N), and timestamp
    csv_path = base_dir / f"convergence_{status_tag}_n{n_rep}_N{N_rep}_{timestamp}.csv"

    # Only write the main CSV file if there are rows collected
    if csv_rows:
        # Open the CSV file in write mode with UTF-8 encoding
        with csv_path.open(mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)  # Create a CSV writer object
            # Write header row describing the columns
            writer.writerow(["component", "n", "N", "layer", "norm", "tol", "is_max"])
            # Write all accumulated data rows
            writer.writerows(csv_rows)
        print(f"\nSaved convergence data (including per-case maxima) to: {csv_path}")
    else:
        # If no rows were recorded, inform the user; nothing is written to disk
        print("\nNo convergence data to log (no rows written).")

    # ---------------------------------------------------------------
    # TITLE: Additional CSV with per-(n, N) failure events (event_rows)
    # ---------------------------------------------------------------
    # Requirement:
    #   - If the convergence test FAILS overall (converged == False),
    #     create an additional CSV containing, for each (n_updt, N_updt)
    #     where a failure occurred, the time layer k, the norm value, and
    #     the component ('u' or 'v').
    #   - If convergence SUCCEEDS overall (converged == True), DO NOT
    #     create this additional CSV.
    if not converged:
        if event_rows:
            # Only in failure case, with at least one recorded failure, do we write the events CSV.
            events_csv_path = base_dir / (
                f"convergence_failures_n{n_rep}_N{N_rep}_{timestamp}.csv"
            )

            with events_csv_path.open(mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Header for the failures-events CSV
                writer.writerow(["component", "n", "N", "layer", "norm", "status"])
                writer.writerows(event_rows)
            print(f"Saved per-(n, N) convergence failures to: {events_csv_path}")
        else:
            # This would be unusual (no failure events but converged == False),
            # but we handle it gracefully.
            print(
                "Convergence failed but no individual failure events were recorded; "
                "no additional failures CSV written."
            )
    else:
        # Convergence succeeded overall; by design, no additional failures CSV is created.
        print("Convergence succeeded; no additional failures CSV file was created.")
