# ----------------------------------------
# Import required modules and classes
# ----------------------------------------

from utils.class_timoshenko_benchmark import TimoshenkoBenchmark  # Benchmark generator for symbolic solution, source terms, and initial conditions

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