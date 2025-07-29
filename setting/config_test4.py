# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # NumPy: array manipulation, numerical operations, linspace, trigonometry


# ======================================================
# SIMULATION DOMAIN CONFIGURATION
# ======================================================
# Defines the temporal and spatial domain for the Timoshenko beam simulation.

T = 2.0     # Total simulation time → defines time interval [0, T]
ell = 2.0   # Beam length → defines spatial interval [0, ℓ]


# ======================================================
# MODEL PHYSICAL PARAMETERS
# ======================================================
# Physical constants governing the nonlinear Timoshenko beam system.
# These affect elasticity, damping, and rotational coupling in the PDEs.

alpha = 1.0   # Linear stiffness in displacement equation u(x, t)
beta  = 1.0   # Nonlinear coefficient scaling energy from strain (∂u/∂x)^2
gamma = 1.0   # Linear stiffness in rotation equation v(x, t)
delta = 1.0   # Damping coefficient in v(x, t) — energy dissipation
a1    = 1.0   # Coupling term: ∂v/∂x contribution in u-equation
a2    = 1.0   # Coupling term: ∂u/∂x contribution in v-equation


# ======================================================
# ANALYTICAL SOLUTION PARAMETERS
# ======================================================
# Frequencies for benchmark analytical solutions used to validate numerical solvers.

lam_u = 14  # Number of spatial oscillations in displacement u(x, t)
lam_v = 14  # Number of spatial oscillations in rotation v(x, t)


# ======================================================
# TEMPORAL DISCRETIZATION PARAMETERS
# ======================================================
# Defines time grid and time step size for numerical integration over [0, T].

n = 512                         # Number of time subintervals in [0, T]
t = np.linspace(0, T, n + 1)    # Time grid with (n+1) points: includes both endpoints
tau = T / n                     # Time step size: τ = T / n


# ======================================================
# SPECTRAL METHOD PARAMETERS
# ======================================================
# Spectral Galerkin projection settings using Legendre polynomial basis.

N = 35  # Number of Legendre polynomial modes used for spatial projection


# ======================================================
# QUADRATURE SETTINGS FOR SPATIAL INTEGRATION
# ======================================================
# Configuration for adaptive Gauss–Legendre quadrature to project symbolic functions.

quad_kwargs = {
    'tol': 1e-6,           # Absolute tolerance for adaptive integration convergence
    'min_dx': 1 / 128.0,   # Minimum allowed subinterval width before stopping refinement
    'n_gauss': 5,          # Initial number of Gauss–Legendre nodes per subinterval
    'max_gauss': 50        # Maximum nodes allowed to avoid over-refinement
}