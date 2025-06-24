# ---------------------------------------------------------------
# IMPORT REQUIRED MODULES AND CLASSES
# ---------------------------------------------------------------

# Compute the L2 norm error between the exact and approximate solution
from utils.auxiliary import compute_L2_error as L2_norm

# Provides symbolic benchmark data for the Timoshenko beam model
from utils.class_timoshenko_benchmark import TimoshenkoBenchmark

# Configuration file containing physical constants and solver settings
import utils.config as cfg

# Galerkin solver implementation for the nonlinear Timoshenko beam PDE system
from utils.class_timoshenko import TimoshenkoModelSolver

# For plotting the L2 error over time
import matplotlib.pyplot as plt


# ---------------------------------------------------------------
# GENERATE SYMBOLIC BENCHMARK DATA
# ---------------------------------------------------------------

# Instantiate benchmark object to obtain reference solutions and initial data
benchmark = TimoshenkoBenchmark()

# Retrieve:
# - f1, f2: external force source terms
# - u0, u1: initial displacement and its time derivative
# - v0, v1: initial rotation and its time derivative
f1, f2, u0, u1, v0, v1 = benchmark.get_initial_data()


# ---------------------------------------------------------------
# INITIALIZE AND CONFIGURE THE GALERKIN SOLVER
# ---------------------------------------------------------------

# Create an instance of the Galerkin solver for the nonlinear Timoshenko beam equations
solver = TimoshenkoModelSolver(
    ell=cfg.ell,         # Length of the beam (spatial domain)
    T=cfg.T,             # Total simulation time
    alpha=cfg.alpha,     # Memory damping coefficient (displacement equation)
    beta=cfg.beta,       # Viscous damping coefficient
    gamma=cfg.gamma,     # Rotational stiffness
    delta=cfg.delta,     # Damping in the rotational equation
    a1=cfg.a1,           # Coupling term: ∂v/∂x → displacement
    a2=cfg.a2,           # Coupling term: ∂u/∂x → rotation
    n=cfg.n,             # Number of time steps
    N=cfg.N,             # Number of spatial Galerkin modes
    f1=f1, f2=f2,        # External forces
    u0=u0, u1=u1,        # Displacement initial conditions
    v0=v0, v1=v1         # Rotation initial conditions
)


# ---------------------------------------------------------------
# COMPUTE APPROXIMATE SOLUTION AND L2 ERROR
# ---------------------------------------------------------------

# Calculate L2 error between the exact solution u(x, t) and its Galerkin approximation
L2_error = L2_norm(
    benchmark.callable_exact_solution('u'),    # Callable exact solution u(x, t)
    solver.callable_compute_ansatz('u'),       # Callable Galerkin approximation
    cfg.ell                                    # Domain length for integration
)


# ---------------------------------------------------------------
# DISPLAY L2 ERROR OVER TIME OR MODES
# ---------------------------------------------------------------

# Check whether L2_error is a sequence (time series) or a scalar
if hasattr(L2_error, '__len__'):
    # L2 error is time-dependent; print error for each step
    for k, err in enumerate(L2_error):
        print(f"For k = {k}, the L2 error = {err:.6e}")
else:
    # L2 error is a single scalar value
    print(f"The L2 error = {L2_error:.6e}")


# ---------------------------------------------------------------
# PLOT L2 ERROR OVER TIME
# ---------------------------------------------------------------

plt.figure(figsize=(8, 4))
plt.plot(cfg.t, L2_error, marker='o', linestyle='-', label='L2 Error')
plt.title("L2 Error Over Time")
plt.xlabel("Time")
plt.ylabel("L2 Error")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()