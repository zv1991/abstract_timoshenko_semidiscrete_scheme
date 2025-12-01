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
    # for the current test case. We perform a numerical convergence study,
    # refining both time steps and spatial modes until convergence is detected
    # (or configured limits are reached).
    print("No exact solutions available; running convergence analysis...")

    # ---------------------------------------------------------------
    # IMPORTS (LOCAL TO THIS BRANCH)
    # ---------------------------------------------------------------
    from pathlib import Path       # Path: cross-platform filesystem path helper (joining, creating dirs, etc.)
    from datetime import datetime  # datetime: used to generate timestamp-based, unique filenames
    import csv                     # csv: standard library CSV writer for logging convergence metrics

    # ---------------------------------------------------------------
    # HELPER FUNCTION: component-wise convergence check
    # ---------------------------------------------------------------
    def _check_component_convergence(
        component: str,
        solver_prev,
        solver,
        n_updt: int,
        N_updt: int,
        tol: float,
        tol_str: str,
        csv_rows: list[list[object]],
        event_rows: list[list[object]],
        aux_module,
    ) -> tuple[bool, float]:
        """
        Check convergence for a single component ('u' or 'v') across time layers
        for a given (n_updt, N_updt), log per-layer norms to csv_rows, and
        record the first failure (if any) into event_rows.

        Parameters
        ----------
        component : str
            Either 'u' or 'v'; selects which coefficient set to compare.
        solver_prev :
            Baseline solver instance at (n_updt, N_base), used as reference.
        solver :
            Refined solver instance at (n_updt, N_updt) to compare against baseline.
        n_updt : int
            Number of time steps used in this refinement pass.
        N_updt : int
            Number of spatial modes used in this refinement pass.
        tol : float
            L2 tolerance; norms above this are considered non-convergent.
        tol_str : str
            Preformatted string version of `tol` for CSV logging.
        csv_rows : list[list[object]]
            Accumulator for main CSV: detailed norms and per-(component, n, N) maxima.
        event_rows : list[list[object]]
            Accumulator for failure-event CSV (only written if global convergence fails).
        aux_module :
            Module/namespace providing numerical helpers, e.g.
            `compute_L2_difference_norms_from_coeffs`.

        Returns
        -------
        converged : bool
            True if the component converged at all checked time layers; False otherwise.
        max_norm : float
            Maximum L2 difference observed across the checked time layers.
        """
        # Select the appropriate coefficient arrays based on the component name.
        if component == "u":
            coeff_prev = solver_prev.tilde_u
            coeff_next = solver.tilde_u
        elif component == "v":
            coeff_prev = solver_prev.tilde_v
            coeff_next = solver.tilde_v
        else:
            # Defensive guard: only 'u' and 'v' are supported components.
            raise ValueError(f"Unknown component: {component!r}. Expected 'u' or 'v'.")

        converged = True   # Assume convergence until we observe a violation.
        max_norm = 0.0     # Tracks maximum norm for this (component, n_updt, N_updt).

        # Iterate over time layers; skip first two layers to avoid initialization transients.
        for k in range(2, n_updt):
            # Compute L2 norm of the difference between baseline and refined coefficients.
            norm_val = aux_module.compute_L2_difference_norms_from_coeffs(
                coeff_init=coeff_prev,   # Baseline coefficients.
                coeff_next=coeff_next,   # Refined coefficients.
                config=solver_prev,      # Grid/weights context taken from baseline solver.
                time_layer=k             # Current time layer index.
            )

            # Track the maximum norm observed across layers for this component.
            if norm_val > max_norm:
                max_norm = norm_val

            # Append detailed per-layer data to main CSV rows.
            csv_rows.append([
                component,               # 'u' or 'v'.
                n_updt,                  # Time steps.
                N_updt,                  # Spatial modes.
                k,                       # Time layer index.
                f"{norm_val:.6e}",       # L2 norm in scientific notation.
                tol_str,                 # Tolerance as a string for consistency.
                0                        # is_max = 0 → regular per-layer entry.
            ])

            # If this layer violates the tolerance, record a failure event and stop checking.
            if norm_val > tol:
                print(
                    f"[{component}] n={n_updt}, N={N_updt}, layer={k}: "
                    f"L2 diff norm = {norm_val:.6e} > tol = {tol:.6e}"
                )
                converged = False

                # Record the first failure event for this (component, n_updt, N_updt).
                event_rows.append([
                    component,           # Component that failed ('u' or 'v').
                    n_updt,              # Time steps.
                    N_updt,              # Spatial modes.
                    k,                   # First failing layer.
                    f"{norm_val:.6e}",   # Failing norm value.
                    "fail"               # Status flag (failure).
                ])

                break                   # Stop checking further layers for this component.

        # After we finish (or break), append a "max" summary row for this component.
        csv_rows.append([
            component,
            n_updt,
            N_updt,
            "max",                      # Special marker: per-(component, n, N) maximum.
            f"{max_norm:.6e}",          # Largest L2 norm seen for this component.
            tol_str,
            1                           # is_max = 1 → summary row.
        ])

        return converged, max_norm

    # ---------------------------------------------------------------
    # TITLE: Tunable thresholds and refinement limits
    # ---------------------------------------------------------------
    tol = 1e-4                  # L2-difference tolerance: convergence is accepted if all norms ≤ tol.
    max_increment_n = 8         # Max exponent for time refinement: n_updt = 2**n_incr * n_base.
    max_galerkin_mode = 50      # Max number of additional spatial modes beyond N_base to test.

    # Base resolutions drawn from the selected configuration object `cfg`.
    n_base = cfg.n              # Baseline number of time steps from configuration.
    N_base = cfg.N              # Baseline number of spatial modes (Galerkin modes) from configuration.

    # Hard caps to keep overall runtime and memory usage bounded.
    n_limit = 2048              # Absolute upper limit on time steps (safety bound).
    N_limit = 17                # Absolute upper limit on spatial modes (safety bound).

    converged = False           # Global flag: True if both u and v converge for some (n, N).

    # Track the specific (n, N) pair where convergence is first detected.
    best_n = None               # Records converged number of time steps, if found.
    best_N = None               # Records converged number of spatial modes, if found.

    # ===============================================================
    # CSV accumulation structures
    #   1) csv_rows   → detailed norms at each time layer
    #   2) event_rows → per-(n, N) component failure events
    # ===============================================================
    csv_rows: list[list[object]] = []    # Detailed per-layer norms for main CSV.
    event_rows: list[list[object]] = []  # Per-(n, N) failure events for secondary CSV.

    # Pre-format tolerance once (used repeatedly in CSV rows for consistency).
    tol_str = f"{tol:.6e}"               # String representation of tolerance for CSV output.

    # ===============================================================
    # TITLE: Time refinement loop (progressively doubles n up to n_limit)
    # ===============================================================
    for n_incr in range(max_increment_n + 1):
        n_updt = (2 ** n_incr) * n_base   # Updated time resolution for this pass.

        # Stop refinement if we exceed the hard upper bound on time steps.
        if n_updt > n_limit:
            break

        # -----------------------------------------------------------
        # TITLE: Baseline solution at fixed spatial resolution (N_base)
        # Purpose: compare refined-N solutions against this baseline.
        # -----------------------------------------------------------
        solver_prev = aux.named(
            test.name,                   # Name/label of the current test case.
            TimoshenkoModelSolver(
                ell=cfg.ell, T=cfg.T,    # Domain length and final time from config.
                alpha=cfg.alpha, beta=cfg.beta,
                gamma=cfg.gamma, delta=cfg.delta,
                a1=cfg.a1, a2=cfg.a2,    # Physical/material parameters.
                n=n_updt, N=N_base,      # Time steps (refined) and baseline spatial modes.
                f1=f1, f2=f2,            # Forcing terms for u and v.
                u0=u0, u1=u1,            # Initial/boundary data for u.
                v0=v0, v1=v1,            # Initial/boundary data for v.
                du0=du0, du1=du1,        # Initial/boundary data for ∂u/∂t.
                dv0=dv0, dv1=dv1,        # Initial/boundary data for ∂v/∂t.
                known_solutions=test.known_solutions  # Optional exact solutions (diagnostics only).
            )
        )

        # ===========================================================
        # TITLE: Spatial refinement loop (increment N up to N_limit)
        # ===========================================================
        for galerkin_mode in range(1, max_galerkin_mode + 1):
            N_updt = N_base + galerkin_mode  # New number of spatial modes being tested.

            # If we exceed the spatial mode cap, stop refining space for this n_updt.
            if N_updt > N_limit:
                break

            # Progress feedback for potentially long runs.
            print(f"Testing n = {n_updt}, N = {N_updt}.")

            # Current solver at (n_updt, N_updt) to compare with baseline at (n_updt, N_base).
            solver = aux.named(
                test.name,               # Same test name label for the refined solver.
                TimoshenkoModelSolver(
                    ell=cfg.ell, T=cfg.T,
                    alpha=cfg.alpha, beta=cfg.beta,
                    gamma=cfg.gamma, delta=cfg.delta,
                    a1=cfg.a1, a2=cfg.a2,
                    n=n_updt, N=N_updt,  # Same time steps as baseline, but more spatial modes.
                    f1=f1, f2=f2,
                    u0=u0, u1=u1,
                    v0=v0, v1=v1,
                    du0=du0, du1=du1,
                    dv0=dv0, dv1=dv1,
                    known_solutions=test.known_solutions
                )
            )

            # -------------------------------------------------------
            # TITLE: Convergence check for component u
            # -------------------------------------------------------
            u_converged, max_norm_u = _check_component_convergence(
                component="u",
                solver_prev=solver_prev,
                solver=solver,
                n_updt=n_updt,
                N_updt=N_updt,
                tol=tol,
                tol_str=tol_str,
                csv_rows=csv_rows,
                event_rows=event_rows,
                aux_module=aux
            )

            # -------------------------------------------------------
            # TITLE: Convergence check for component v (only if u passed)
            # -------------------------------------------------------
            if u_converged:
                v_converged, max_norm_v = _check_component_convergence(
                    component="v",
                    solver_prev=solver_prev,
                    solver=solver,
                    n_updt=n_updt,
                    N_updt=N_updt,
                    tol=tol,
                    tol_str=tol_str,
                    csv_rows=csv_rows,
                    event_rows=event_rows,
                    aux_module=aux
                )
            else:
                # If u did not converge, v is considered non-converged as well.
                v_converged = False
                max_norm_v = 0.0  # Defined for completeness; not used further.

            # -------------------------------------------------------
            # TITLE: Success path — both components converged
            # -------------------------------------------------------
            if u_converged and v_converged:
                print(f"\n Converged at n = {n_updt}, N = {N_updt}")
                converged = True          # Set global convergence flag.
                best_n = n_updt          # Record converged time resolution.
                best_N = N_updt          # Record converged spatial resolution.

                # Requirement: do NOT record success events in event_rows and
                # only save the main CSV in the global success case (handled later).

                # Snapshot plots of approximate solutions for diagnostics / visual check.
                print("Plotting approximate solution snapshots")
                for sol_type in ("u", "v"):
                    aux.plot_approx_solution_at_time_k(
                        approx_solver=solver,   # Use converged solver instance.
                        solution_type=sol_type, # "u" or "v" component.
                        config=solver           # Solver also acts as its own plotting config.
                    )

                break  # Exit spatial refinement loop early on success.

            # Promote current refined solver to be the new baseline for the next N increment.
            solver_prev = solver

        # If convergence has been achieved at this time resolution, stop refining time.
        if converged:
            break

    # ---------------------------------------------------------------
    # TITLE: Final reporting and CSV logging (success or failure)
    # ---------------------------------------------------------------
    if not converged:
        # No (n, N) pair satisfied the convergence criterion within limits.
        print(
            f"\n Convergence was not reached within the given limits: "
            f"n ≤ {n_limit} and N ≤ {N_limit}."
        )
        status_tag = "failure"   # Label used in filenames for the failure case.
        n_rep = n_limit          # Representative n for the failure log filename.
        N_rep = N_limit          # Representative N for the failure log filename.
    else:
        # At least one (n, N) pair satisfied the convergence criterion.
        print("\n Convergence achieved within the specified limits.")
        status_tag = "success"   # Label used in filenames for the success case.
        n_rep = best_n if best_n is not None else n_limit
        N_rep = best_N if best_N is not None else N_limit

    # Base path for storing convergence log CSV files.
    run_name = getattr(test, "name", "run")              # Use test.name if present; else default to "run".
    base_dir = Path("plots") / run_name / "convergence_logs"  # Directory for convergence logs.
    base_dir.mkdir(parents=True, exist_ok=True)          # Ensure directory hierarchy exists.

    # Timestamp used to make filenames unique and sortable.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---------------------------------------------------------------
    # TITLE: Main CSV file with all per-layer norms (csv_rows)
    # ---------------------------------------------------------------
    # Requirement:
    #   - Save this file ONLY if convergence succeeds (global `converged` is True).
    #   - If convergence fails, DO NOT create this main CSV.
    if converged:
        if csv_rows:
            csv_path = base_dir / f"convergence_{status_tag}_n{n_rep}_N{N_rep}_{timestamp}.csv"
            with csv_path.open(mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["component", "n", "N", "layer", "norm", "tol", "is_max"])
                writer.writerows(csv_rows)
            print(f"\nConvergence succeeded. Saved convergence data to: {csv_path}")
        else:
            # Convergence succeeded but no norms were recorded — unusual but handled gracefully.
            print("\nConvergence succeeded, but no convergence data was produced.")
    else:
        # Failure case: main CSV is intentionally NOT written.
        print("\nConvergence failed — main convergence CSV NOT written (success-only).")

    # ---------------------------------------------------------------
    # TITLE: Additional CSV with per-(n, N) failure events (event_rows)
    # ---------------------------------------------------------------
    # Only write this file when global convergence FAILED, and only if
    # there is at least one recorded failure event.
    if not converged:
        if event_rows:
            # NOTE: This filename is kept exactly as requested.
            events_csv_path = base_dir / (
                f"convergence_failures_n{n_rep}_N{N_rep}_{timestamp}.csv"
            )
            with events_csv_path.open(mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["component", "n", "N", "layer", "norm", "status"])
                writer.writerows(event_rows)
            print(f"Saved per-(n, N) convergence failures to: {events_csv_path}")
        else:
            # Global failure without any recorded per-(n, N) failure events.
            print(
                "Convergence failed, but no individual failure events were recorded; "
                "no additional failures CSV written."
            )
    else:
        # Convergence succeeded globally → no failure CSV is created by design.
        print("Convergence succeeded; no additional failures CSV file was created.")
