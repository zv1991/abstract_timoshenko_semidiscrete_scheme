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

def compute_L2_norm(
    approx_solution_generator,
    ell: float,
    k: int = None,
    tol: float = 1e-6,
    method: str = "hglq"
):
    """
    Compute the L2 norm of the approximate solution ũ(x) over [0, ell].

    Parameters
    ----------
    approx_solution_generator : callable or list of callables
        Approximated solution(s). Each function maps x -> ũ_k(x).
    
    ell : float
        Length of the spatial domain [0, ell].

    k : int, optional
        Time index. If provided, compute the L2 norm only at that time step.

    tol : float, optional
        Tolerance for numerical integration.

    method : str, optional
        Integration method to use. Options:
            - 'glq'   : Gauss-Legendre quadrature
            - 'hglq'  : Hierarchical GLQ
            - 'scipy' : SciPy adaptive quadrature

    Returns
    -------
    float or list of floats
        L2 norm at time step k, or list of norms across all time steps.
    """

    # Normalize input to list
    if callable(approx_solution_generator):
        approx_solution_generator = [approx_solution_generator]

    def compute_norm_at_k(k_idx):
        """
        Compute L2 norm at specific time index.
        """
        approx_fn = approx_solution_generator[k_idx]

        def squared_fn(x):
            return approx_fn(x)**2

        integral, _, _ = aux.unified_adaptive_quadrature(
            squared_fn, ell=ell, tol=tol, method=method
        )

        return np.sqrt(integral)

    if k is not None:
        if not (0 <= k < len(approx_solution_generator)):
            raise ValueError(f"Time index k = {k} is out of bounds.")
        return compute_norm_at_k(k)

    return [compute_norm_at_k(i) for i in range(len(approx_solution_generator))]

def L2_integral_matrix_approach(
    N: int,
    coeff: np.ndarray,
    ell: float,
    time_layer: int = None
) -> float | list[float]:
    """
    Compute the L2 norm(s) of a Galerkin-approximated solution using matrix-vector form:
    (ell / 2) * sqrt(cᵗ H c), where H is applied via `galerkin_stencils()`.

    Assumes `coeff` has shape (num_time_layers, N), where time index k = 2 maps to row 0.

    Parameters
    ----------
    N : int
        Number of Galerkin basis functions (number of columns in coeff).
    
    coeff : np.ndarray
        Coefficient matrix of shape (n-2, N), corresponding to time layers k = 2, ..., n-1.

    ell : float
        Length of the spatial domain.

    time_layer : int, optional
        Specific time step k (must satisfy k ≥ 2). If None, computes for all k.

    Returns
    -------
    float or list of floats
        Single L2 norm if `time_layer` is specified; list of norms for all valid k otherwise.
    
    Raises
    ------
    IndexError
        If time_layer < 2 or exceeds available time steps.
    """
    if coeff.shape[1] != N:
        raise ValueError(f"Expected coeff shape (*, {N}), got {coeff.shape}")

    num_layers = coeff.shape[0]  # corresponds to time steps k = 2, ..., n-1

    def compute_single(k_index: int) -> float:
        """
        Compute L2 norm for the coefficient vector at row k_index.
        Maps to time_layer = k_index + 2.
        """
        c_k = coeff[k_index, :]
        H_c = aux.galerkin_stencils(N=N, v=c_k, operator="identity")
        l2_squared = np.dot(c_k, H_c)
        return (ell / 2.0) * np.sqrt(l2_squared)

    if time_layer is not None:
        if time_layer < 2:
            raise IndexError(f"Invalid time_layer = {time_layer}. Must be ≥ 2.")
        k_index = time_layer - 2
        if k_index >= num_layers:
            raise IndexError(f"time_layer = {time_layer} is out of bounds for coeff shape {coeff.shape}")
        return compute_single(k_index)

    # Return for all valid time steps (k = 2 to n-1)
    return [compute_single(k_idx) for k_idx in range(num_layers)]

tilde_u = solver.tilde_u

L2_norms = compute_L2_norm(approx_solution_generator=solver.callable_compute_ansatz('u'), ell=cfg.ell)

L2_norms_all = L2_integral_matrix_approach(N=cfg.N, coeff=solver.tilde_u, ell=cfg.ell)

err1 = abs(L2_norms[2] - L2_norms_all[0])
err2 = abs(L2_norms[3] - L2_norms_all[1])