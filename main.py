# # ========================================================
# # IMPORT MODULES
# # ========================================================

# # Load simulation parameters like beam length, total time, coefficients, and resolution
# import utils.config as cfg

# # L²-norm error calculator for solution comparison
# from utils.auxiliary import compute_L2_error as L2_error

# # Provides symbolic initial conditions, source terms, and exact solution evaluation tools
# from utils.initial_data import (
#     get_initial_data,
#     exact_solution,
#     compute_exact_solution
# )

# # Galerkin-based solver for the nonlinear Timoshenko beam PDE system
# from utils.class_timoshenko import TimoshenkoModelSolver


# # ========================================================
# # GENERATE SYMBOLIC INITIAL CONDITIONS AND FORCINGS
# # ========================================================

# # Retrieve symbolic expressions for:
# # - f1, f2 : source terms for u and v PDEs
# # - u0, v0 : initial conditions u(x, 0), v(x, 0)
# # - u1, v1 : second-order Taylor approximations at t = τ
# f1, f2, u0, u1, v0, v1 = get_initial_data()


# # ========================================================
# # INITIALIZE THE GALERKIN SOLVER
# # ========================================================

# # Instantiate solver with parameters and initial data
# solver = TimoshenkoModelSolver(
#     ell=cfg.ell,         # Beam length
#     T=cfg.T,             # Final time
#     alpha=cfg.alpha,     # Memory kernel coefficient
#     beta=cfg.beta,       # Viscous damping coefficient for u
#     gamma=cfg.gamma,     # Rotational stiffness for v
#     delta=cfg.delta,     # Damping coefficient for v
#     a1=cfg.a1,           # Coupling term (v into u)
#     a2=cfg.a2,           # Coupling term (u into v)
#     n=cfg.n,             # Number of time steps
#     N=cfg.N,             # Number of Galerkin modes
#     f1=f1, f2=f2,        # Source functions
#     u0=u0, u1=u1,        # Initial and Taylor-approximated states for u
#     v0=v0, v1=v1         # Initial and Taylor-approximated states for v
# )


# # ========================================================
# # EVALUATE SOLUTIONS FOR COMPARISON
# # ========================================================

# # Evaluate Galerkin solution for u(x, t) at 5 points (4 partitions) in space
# gal_approx_u = solver.galerkin_approx_solution(
#     solution_type='u',
#     unif_prt_spc=4
# )

# # Evaluate exact symbolic solution u(x, t) at the same spatial grid
# exact_u = exact_solution(
#     solution_type='u',
#     unif_prt_spc=4
# )


# # ========================================================
# # ACCESS CONDITION NUMBER (STABILITY CHECK)
# # ========================================================

# # Retrieve matrix condition numbers (used to detect potential ill-conditioning)
# cond_u = solver.cond_u


# # ========================================================
# # COMPUTE L2 ERROR ACROSS TIME STEPS
# # ========================================================

# # List of exact solution functions: [u(x, t0), u(x, t1), ..., u(x, tn)]
# u_exact_list = compute_exact_solution('u')

# # List of Galerkin solution functions for the same time steps
# u_galerkin_list = solver.compute_ansatz('u')

# # Compute L² error between exact and Galerkin solutions at all time steps
# errors = L2_error(
#     exact_solution_generator=u_exact_list,
#     approx_solution_generator=u_galerkin_list,
#     ell=cfg.ell
# )

# # Print L² error per time step
# for i, error in enumerate(errors):
#     print(f"L2 error at time step {i}: {error}")

# ----------------------------------------
# Import required modules and classes
# ----------------------------------------

from utils.timoshenko_benchmark import TimoshenkoBenchmark  # Benchmark generator for symbolic solution, source terms, and initial conditions

# Load physical and simulation configuration parameters
# Includes beam length, time grid, coefficients, number of modes/steps
import utils.config as cfg

# Galerkin-based solver for the nonlinear Timoshenko beam PDE system
from utils.class_timoshenko import TimoshenkoModelSolver


# ----------------------------------------
# Generate symbolic benchmark data
# ----------------------------------------

# Create a benchmark instance to obtain source terms and exact initial data
benchmark = TimoshenkoBenchmark()

# Retrieve symbolic source terms and initial states for displacement (u) and rotation (v)
f1, f2, u0, u1, v0, v1 = benchmark.get_initial_data()


# ========================================================
# INITIALIZE THE GALERKIN SOLVER
# ========================================================

# Instantiate the solver with configuration values and initial conditions
solver = TimoshenkoModelSolver(
    ell=cfg.ell,         # Beam length (domain spatial extent)
    T=cfg.T,             # Total simulation time
    alpha=cfg.alpha,     # Memory kernel coefficient for the displacement equation
    beta=cfg.beta,       # Viscous damping coefficient applied to u
    gamma=cfg.gamma,     # Rotational stiffness (coefficient in v-equation)
    delta=cfg.delta,     # Damping coefficient in v-equation
    a1=cfg.a1,           # Coupling term: ∂v/∂x → u-equation
    a2=cfg.a2,           # Coupling term: ∂u/∂x → v-equation
    n=cfg.n,             # Number of time steps in the simulation
    N=cfg.N,             # Number of Galerkin basis functions/modes
    f1=f1, f2=f2,        # Time-dependent source functions for u and v equations
    u0=u0, u1=u1,        # Initial displacement and its second-order Taylor approximation at t=τ
    v0=v0, v1=v1         # Initial rotation and its second-order Taylor approximation at t=τ
)