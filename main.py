# ===============================================================
# SCRIPT: Solve and Analyze Timoshenko Beam Model (Galerkin)
# ---------------------------------------------------------------
# Description:
#   - Loads benchmark data for the nonlinear Timoshenko beam PDE
#   - Solves using a Galerkin method
#   - Computes L2 error over time
#   - Generates a LaTeX-styled error plot (saved as PDF)
# ===============================================================

# ---------------------------------------------------------------
# IMPORT REQUIRED MODULES AND CLASSES
# ---------------------------------------------------------------

# Utility functions: L2 norm computation, LaTeX-styled plotting
import utils.auxiliary as aux

# Provides exact symbolic benchmark data (forces, initial conditions, solution)
from utils.class_timoshenko_solns import TimoshenkoSolutions

# Simulation configuration (domain, time, damping, discretization settings)
import utils.config as cfg

# Galerkin solver implementation for the nonlinear Timoshenko beam equations
from utils.class_timoshenko import TimoshenkoModelSolver


# -------------------------------------
# Create an instance of the class
# -------------------------------------

# Case 1: Using known analytical solutions
benchmark = TimoshenkoSolutions(known_solutions=True)

# OR

# Case 2: Using approximated Taylor-based initial data
# benchmark = TimoshenkoSolutions(known_solutions=False)


# -------------------------------------
# Retrieve all data components
# -------------------------------------

f1, f2, u0, u1, v0, v1, du0, du1, dv0, dv1 = benchmark.get_initial_data()


# ---------------------------------------------------------------
# STEP 2: INITIALIZE THE GALERKIN SOLVER
# ---------------------------------------------------------------

# Instantiate the solver using configuration values
solver = TimoshenkoModelSolver(
    ell=cfg.ell,        # Beam length (domain: x ∈ [0, ell])
    T=cfg.T,            # Total simulation time
    alpha=cfg.alpha,    # Damping in displacement equation
    beta=cfg.beta,      # Viscous damping coefficient
    gamma=cfg.gamma,    # Rotational stiffness
    delta=cfg.delta,    # Damping in rotational equation
    a1=cfg.a1,          # Coupling term: ∂v/∂x affects u
    a2=cfg.a2,          # Coupling term: ∂u/∂x affects v
    n=cfg.n,            # Number of time steps
    N=cfg.N,            # Number of Galerkin spatial modes
    f1=f1, f2=f2,       # External forces
    u0=u0, u1=u1,       # Displacement at at t=0, τ
    v0=v0, v1=v1,       # Rotation at at t=0, τ
    du0=du0, du1=du1,   # ∂u/∂x at t=0, τ
    dv0=dv0, dv1=dv1    # ∂v/∂x at t=0, τ
    
)


# # ---------------------------------------------------------------
# # STEP 3: COMPUTE L2 ERROR BETWEEN NUMERICAL AND EXACT SOLUTION
# # ---------------------------------------------------------------

# # Compute L2 norm error between:
# #   - Exact solution u(x, t)
# #   - Galerkin approximation u_h(x, t)
# L2_error = aux.compute_L2_error(
#     benchmark.callable_exact_solution('u'),     # Reference solution
#     solver.callable_compute_ansatz('u'),        # Numerical solution
#     cfg.ell                                     # Integration over spatial domain [0, ell]
# )


# # ---------------------------------------------------------------
# # STEP 4: PRINT L2 ERROR (TIME-DEPENDENT)
# # ---------------------------------------------------------------

# # Time-dependent error array: print error at each time step
# for k in range(cfg.n + 1):
#     print(f"Time step {k:3d}: L2 error = {L2_error[k]:.6e}")


# # ---------------------------------------------------------------
# # STEP 5: PLOT L2 ERROR OVER TIME AND EXPORT TO PDF
# # ---------------------------------------------------------------

# # Plot and export error as a LaTeX-styled publication-quality PDF
# pdf_file = aux.plot_L2_error_over_time(cfg.t, L2_error, cfg)

# # Confirm the output path to user
# print(f"Plot successfully saved to: {pdf_file}")

cond_u = solver.cond_u
cond_v = solver.cond_v

tild_u = solver.tilde_u
tild_v = solver.tilde_v