# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # NumPy is used for numerical operations, array manipulations, and time discretization
# Tip: Keeping imports minimal here ensures this configuration module loads quickly.
#      Other modules can import these constants without pulling in heavy dependencies.


# ======================================================
# DOMAIN PARAMETERS
# ======================================================
# Title: Time and Space Domain Configuration
# Description: Define the spatial and temporal bounds for the PDE simulation.

T = 1.0     # Total simulation time; simulation runs over the interval [0, T]
ell = 2.0   # Length of the beam (spatial domain); domain is [0, ell]
# Note: Keep the spatial domain [0, ell] consistent across basis/quad routines.


# ======================================================
# EQUATION COEFFICIENTS
# ======================================================
# Title: Physical Parameters for Timoshenko Beam Model
# Description: Governs dynamics in displacement and rotation PDEs.

alpha = 1.0   # Stiffness coefficient for displacement u(x, t)
beta  = 1.0   # Nonlinear damping coefficient for u-equation
gamma = 1.0   # Rotational stiffness coefficient for v-equation
delta = 1.0   # Linear damping coefficient for v-equation
a1    = 1.0   # Coupling: ∂v/∂x influence in u-equation
a2    = 1.0   # Coupling: ∂u/∂x influence in v-equation
# Sanity check: Usually alpha > 0 and gamma > 0 for stability and well-posedness.


# ======================================================
# OSCILLATION PARAMETERS FOR BENCHMARK SOLUTIONS
# ======================================================
# Title: Parameters for Initial Oscillatory Conditions
# Description: Used to construct initial displacement/rotation for convergence tests.

lam = 5  # Number of spatial oscillations in u(x, 0) and v(x, 0)

# Calculate temporal oscillation frequency for time-dependent sine solutions
lam1 = T / np.pi * np.sqrt(gamma * (lam * np.pi / ell)**2 + delta)

A = 1.0  # Amplitude of sine functions for initial displacement/rotation
# Practical use: Use these for method of manufactured solutions (MMS) testing.


# ======================================================
# TEMPORAL DISCRETIZATION
# ======================================================
# Title: Time Grid for Time-Stepping Algorithms
# Description: Create a uniform time grid for solvers such as RK or backward Euler.

n = 2048                      # Number of time steps (uniform)
t = np.linspace(0, T, n + 1)  # Discrete time array from 0 to T (inclusive)
tau = T / n                   # Time step size (τ); ensures t spans [0, T] uniformly


# ======================================================
# SPECTRAL METHOD CONFIGURATION
# ======================================================
# Title: Modal Resolution in Spectral Galerkin Methods
# Description: Specifies number of basis functions (e.g., Legendre) for projection.

N = 20  # Number of Legendre polynomial basis functions
# Tip: Start with small N for validation; increase to study convergence behavior.


# ======================================================
# QUADRATURE CONFIGURATION
# ======================================================
# Title: Adaptive Gauss–Legendre Quadrature Parameters
# Description: Configure quadrature used in numerical integration of PDE components.

quad_kwargs = {
    'tol': 1e-6,           # Absolute error tolerance for adaptive integration
    'min_dx': 1 / 128.0,   # Minimum subinterval width (avoids over-refinement)
    'n_gauss': 5,          # Initial Gauss points per integration subdomain
    'max_gauss': 50        # Maximum allowed Gauss points during refinement
}
# Integration Tip: For highly oscillatory functions (larger lam, lam1), increase n_gauss and max_gauss.
# Debugging Tip: Log quadrature diagnostics (e.g., total subintervals, max refinement depth) to optimize accuracy-performance tradeoff.