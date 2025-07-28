# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # NumPy is used for numerical operations like array creation and time discretization


# ======================================================
# DOMAIN PARAMETERS
# ======================================================
# Defines the spatial and temporal extent of the simulation domain.

T = 1.0     # Total simulation time — sets the time domain [0, T]
ell = 2.0   # Beam length — defines the spatial domain [0, ell]


# ======================================================
# EQUATION COEFFICIENTS
# ======================================================
# Physical and structural parameters governing the Timoshenko beam model:
# These coefficients influence stiffness, damping, and coupling behavior.

alpha = 1.0   # Material stiffness in the displacement equation (u) — affects elastic recovery
beta  = 1.0   # Nonlinear damping coefficient for displacement (u) — scaled by gradient magnitude
gamma = 1.0   # Linear rotational stiffness — determines restoring torque in rotation (v)
delta = 1.0   # Damping coefficient for rotational motion (v) — energy dissipation
a1    = 1.0   # Coupling coefficient: ∂v/∂x in the displacement (u) equation
a2    = 1.0   # Coupling coefficient: ∂u/∂x in the rotation (v) equation


# ======================================================
# OSCILLATION PARAMETERS FOR BENCHMARK SOLUTIONS
# ======================================================
# Parameters that define the spatial frequency of the test/trial functions used
# to validate or benchmark numerical accuracy.

lam_u = 5  # Number of spatial oscillations in displacement u(x, t)
lam_v = 5  # Number of spatial oscillations in rotation v(x, t)


# ======================================================
# TEMPORAL DISCRETIZATION
# ======================================================
# Time-stepping configuration for the numerical solver.

n = 128                        # Number of uniform time intervals in the range [0, T]
t = np.linspace(0, T, n + 1)   # Array of time nodes: [t₀, t₁, ..., tₙ]; total n+1 points
tau = T / n                   # Time step size τ = T / n


# ======================================================
# SPECTRAL METHOD CONFIGURATION
# ======================================================
# Configuration for the Galerkin method using Legendre polynomials.
# Used for projecting PDEs onto a finite-dimensional function space.

N = 15  # Number of Legendre polynomial basis functions used in the Galerkin projection


# ======================================================
# QUADRATURE CONFIGURATION
# ======================================================
# Parameters controlling the adaptive Gauss–Legendre integration scheme used
# for accurate spatial integration during spectral projection.

quad_kwargs = {
    'tol': 1e-6,           # Absolute tolerance for integral convergence — higher accuracy
    'min_dx': 1 / 128.0,   # Minimum width of a subinterval to avoid over-refinement
    'n_gauss': 5,          # Initial number of Gauss–Legendre points per subinterval
    'max_gauss': 50        # Upper bound on Gauss points — prevents excessive refinement
}