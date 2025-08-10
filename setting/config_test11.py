# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # NumPy is used for numerical operations, array manipulations, and time discretization
# Tip: Keeping imports minimal here ensures this configuration module loads quickly.
#      Other modules can import these constants without pulling in heavy dependencies.


# ======================================================
# DOMAIN PARAMETERS
# ======================================================
# Define the spatial and temporal domain limits for the simulation.
# These are global constants so other modules (solver, tests, plotting) can
# `from config import T, ell, ...` and remain in sync.

T = 1.0     # Total simulation time; the simulation runs over the time interval [0, T]
ell = 2.0   # Length of the beam; defines the spatial domain [0, ell]
# Note: Many basis/quad routines expect domains starting at 0; keep [0, ell] consistent project-wide.


# ======================================================
# EQUATION COEFFICIENTS
# ======================================================
# Physical parameters of the Timoshenko beam model used in PDEs:
# Affect stiffness, damping, and interdependence between displacement and rotation.
# Guidance:
# - Keep units consistent with the discretized PDEs (especially if mixing SI with nondimensional forms).
# - If you switch to a nondimensional system, document the scaling used.

alpha = 1.0   # Material stiffness coefficient for displacement u(x,t) — governs elasticity
beta  = 1.0   # Nonlinear damping coefficient in u-equation — introduces gradient-based dissipation
gamma = 1.0   # Rotational stiffness in v-equation — determines restoring torque behavior
delta = 0.0   # Linear damping coefficient in v-equation — introduces energy loss
a1    = 1.0   # Coupling coefficient from ∂v/∂x in u-equation — rotational influence on displacement
a2    = 1.0   # Coupling coefficient from ∂u/∂x in v-equation — displacement influence on rotation
# Sanity note: In many studies alpha, gamma > 0 for well-posedness; check your solver’s assumptions.


# ======================================================
# OSCILLATION PARAMETERS FOR BENCHMARK SOLUTIONS
# ======================================================
# Parameters for constructing analytical/benchmark initial data — typically used
# to verify convergence and numerical correctness via oscillatory functions.
# These are NOT the PDE coefficients; they shape initial conditions only.

lam_u = 5  # Number of spatial oscillations in initial displacement u(x, 0)
lam_v = 5  # Number of spatial oscillations in initial rotation v(x, 0)

A_u = 0.25  # Amplitude of the sine function for u(x, 0)
A_v = 0.25  # Amplitude of the sine function for v(x, 0)


# ======================================================
# TEMPORAL DISCRETIZATION
# ======================================================
# Define time discretization for time-stepping algorithms (e.g., Runge–Kutta, backward Euler).
# We use a uniform time grid here for simplicity; adaptive schemes can still use `t` as
# nominal output times while stepping internally with variable dt.

n = 128                       # Number of time steps (uniformly spaced) over [0, T]
t = np.linspace(0, T, n + 1)  # Discretized time array from t₀ = 0 to tₙ = T; (n+1) points total
tau = T / n                  # Time step size (τ), derived from total time and step count
# Check: tau * n == T (within FP tolerance). Choosing n as a power of two can help with FFT-based post-processing.


# ======================================================
# SPECTRAL METHOD CONFIGURATION
# ======================================================
# Defines the size of the approximation space used in a Galerkin spectral projection.
# If you use Legendre polynomials on [0, ell], ensure your basis normalization matches
# your quadrature rules to preserve orthogonality.

N = 20  # Number of Legendre polynomial basis functions (modal resolution in space)
# Recommendation: Start small to validate correctness, then increase N to study convergence.


# ======================================================
# QUADRATURE CONFIGURATION
# ======================================================
# Parameters for adaptive Gauss–Legendre quadrature,
# which ensures accurate numerical integration for mass/stiffness matrices and load terms.
# Tuning tips:
# - Lower 'tol' for higher accuracy (more work).
# - 'min_dx' prevents pathological subdivision.
# - 'n_gauss' is the initial per-subdomain Gauss points; 'max_gauss' caps refinement.

quad_kwargs = {
    'tol': 1e-6,           # Absolute error tolerance for adaptive integration
    'min_dx': 1 / 128.0,   # Smallest allowed subinterval width — prevents over-refinement
    'n_gauss': 5,          # Initial number of Gauss points per integration subdomain
    'max_gauss': 50        # Maximum allowable number of Gauss points — acts as a refinement cap
}
# Implementation note: Log quadrature stats (subdivisions, points used) in the integrator to tune these.
# Practical hint: If you see oscillatory integrands with higher lam_*, increasing n_gauss or max_gauss helps.