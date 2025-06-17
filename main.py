# ========================================================
# IMPORT MODULES
# ========================================================

# Load configuration parameters (e.g., beam length, final time, PDE coefficients, resolution settings)
import utils.config as cfg

# Retrieve symbolic initial and boundary data, as well as exact analytical solutions
from utils.initial_data import get_initial_data, exact_solution

# Class implementing Galerkin projection solver for the nonlinear Timoshenko beam model
from utils.class_timoshenko import TimoshenkoModelSolver


# ========================================================
# GENERATE SYMBOLIC INITIAL CONDITIONS AND FORCINGS
# ========================================================

# Extract:
# - f1, f2: symbolic source terms for the PDE system
# - u0, v0: initial displacements u(x,0), v(x,0)
# - u1, v1: approximations at t=τ via Taylor expansion
f1, f2, u0, u1, v0, v1 = get_initial_data()


# ========================================================
# INITIALIZE THE GALERKIN SOLVER
# ========================================================

# Instantiate the Timoshenko solver with model parameters and initial data
solver = TimoshenkoModelSolver(
    ell=cfg.ell,         # Beam length (spatial domain: [0, ell])
    T=cfg.T,             # Final time of simulation
    alpha=cfg.alpha,     # Memory coefficient (elastic effect)
    beta=cfg.beta,       # Viscous damping coefficient (u-equation)
    gamma=cfg.gamma,     # Rotational stiffness (v-equation)
    delta=cfg.delta,     # Damping coefficient (v-equation)
    a1=cfg.a1,           # Coupling: influence of v on u
    a2=cfg.a2,           # Coupling: influence of u on v
    n=cfg.n,             # Number of time steps (temporal resolution)
    N=cfg.N,             # Number of Galerkin modes (spatial resolution)
    f1=f1,               # Source term for u(x, t)
    f2=f2,               # Source term for v(x, t)
    u0=u0, u1=u1,        # Initial state and time-approximated state for u
    v0=v0, v1=v1         # Initial state and time-approximated state for v
)


# ========================================================
# EVALUATE GALERKIN AND EXACT SOLUTIONS
# ========================================================

# Evaluate Galerkin approximation for u(x, t) at 5 equally spaced points (4 partitions)
gal_approx_u = solver.galerkin_approx_solution(solution_type='u', unif_prt_spc=4)

# Evaluate exact analytical solution at the same grid
exact_u = exact_solution(solution_type='u', unif_prt_spc=4)


# ========================================================
# ACCESS CONDITION NUMBER (STABILITY METRIC)
# ========================================================

# Retrieve condition numbers of system matrix used in Galerkin solver for u
# This helps detect numerical instability (e.g., if condition number is too high)
cond_u = solver.cond_u