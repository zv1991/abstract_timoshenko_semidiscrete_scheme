# =========================
# IMPORT MODULES
# =========================

# Load configuration parameters such as domain length, final time, and physical coefficients
import utils.config as cfg

# Provides symbolic source terms and initial displacement/velocity profiles
from utils.initial_data import get_initial_data, exact_solution

# Implements the Galerkin method to solve Timoshenko beam PDEs
from utils.class_timoshenko import TimoshenkoModelSolver


# =========================
# GENERATE INITIAL DATA
# =========================

# Retrieve symbolic source terms (f1, f2) and initial displacement/velocity for u and v
f1, f2, u0, u1, v0, v1 = get_initial_data()


# =========================
# INITIALIZE SOLVER
# =========================

# Instantiate the solver object with physical parameters, resolution settings, and initial data
solver = TimoshenkoModelSolver(
    ell=cfg.ell,                 # Spatial domain length (beam length)
    T=cfg.T,                     # Final simulation time
    alpha=cfg.alpha,             # Elastic memory coefficient for u
    beta=cfg.beta,               # Viscoelastic damping coefficient for u
    gamma=cfg.gamma,             # Stiffness coefficient for v
    delta=cfg.delta,             # Damping coefficient for v
    a1=cfg.a1,                   # Coupling from v to u
    a2=cfg.a2,                   # Coupling from u to v
    n=cfg.n,                     # Time discretization resolution
    N=cfg.N,                     # Spatial discretization resolution (number of Galerkin modes)
    f1=f1,                       # Forcing term in u-equation
    f2=f2,                       # Forcing term in v-equation
    u0=u0,                       # Initial displacement: u(x, 0)
    u1=u1,                       # Initial approximation: u(x, τ)
    v0=v0,                       # Initial displacement: v(x, 0)
    v1=v1                        # Initial approximation: v(x, τ)
)


# =========================
# SOLVE AND RECONSTRUCT GALERKIN SOLUTION
# =========================

# Compute approximate Galerkin solution u(x, t) at selected spatial slices
# `unif_prt_spc` specifies how many uniformly spaced spatial points to print
gal_approx_u = solver.galerkin_approx_solution('u', unif_prt_spc=5, k=2)

# Extract condition number of the Galerkin system matrix
# A large condition number may indicate numerical instability
cond_u = solver.cond_u