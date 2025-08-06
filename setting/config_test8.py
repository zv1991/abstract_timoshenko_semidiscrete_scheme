# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # NumPy is used for numerical operations, array manipulations, and time discretization


# ======================================================
# DOMAIN PARAMETERS
# ======================================================
# Define the spatial and temporal domain limits for the simulation

T = 1.0     # Total simulation time; the simulation runs over the time interval [0, T]
ell = 2.0   # Length of the beam; defines the spatial domain [0, ell]


# ======================================================
# EQUATION COEFFICIENTS
# ======================================================
# Physical parameters of the Timoshenko beam model used in PDEs:
# Affect stiffness, damping, and interdependence between displacement and rotation.

alpha = 1.0   # Material stiffness coefficient for displacement u(x,t) — governs elasticity
beta  = 1.0   # Nonlinear damping coefficient in u-equation — introduces gradient-based dissipation
gamma = 1.0   # Rotational stiffness in v-equation — determines restoring torque behavior
delta = 1.0   # Linear damping coefficient in v-equation — introduces energy loss
a1    = 1.0   # Coupling coefficient from ∂v/∂x in u-equation — rotational influence on displacement
a2    = 1.0   # Coupling coefficient from ∂u/∂x in v-equation — displacement influence on rotation


# ======================================================
# OSCILLATION PARAMETERS FOR BENCHMARK SOLUTIONS
# ======================================================
# Parameters for constructing analytical or benchmark solutions — typically used
# to verify convergence and numerical correctness via oscillatory Gaussian functions.

lam_u = 15  # Number of spatial oscillations in initial displacement u(x, 0)
lam_v = 15  # Number of spatial oscillations in initial rotation v(x, 0)

A_u = 1.0  # Amplitude of the Gaussian profile for u(x, 0)
A_v = 1.0  # Amplitude of the Gaussian profile for v(x, 0)

c_u = 2.0  # Gaussian width parameter for u(x, 0); lower values = narrower peak
c_v = 2.0  # Gaussian width parameter for v(x, 0); lower values = narrower peak


# ======================================================
# TEMPORAL DISCRETIZATION
# ======================================================
# Define time discretization for time-stepping algorithms like Runge-Kutta or backward Euler.

n = 128                       # Number of time steps (uniformly spaced) over [0, T]
t = np.linspace(0, T, n + 1)  # Discretized time array from t₀ = 0 to tₙ = T; (n+1) points total
tau = T / n                  # Time step size (τ), derived from total time and step count


# ======================================================
# SPECTRAL METHOD CONFIGURATION
# ======================================================
# Defines the size of the approximation space used in Galerkin spectral projection.

N = 20  # Number of Legendre polynomial basis functions (modal resolution in space)


# ======================================================
# QUADRATURE CONFIGURATION
# ======================================================
# Parameters for adaptive Gauss–Legendre quadrature,
# which ensures accurate numerical integration in the spectral method.

quad_kwargs = {
    'tol': 1e-6,           # Absolute error tolerance for adaptive integration
    'min_dx': 1 / 128.0,   # Smallest allowed subinterval width — prevents over-refinement
    'n_gauss': 5,          # Initial number of Gauss points per integration subdomain
    'max_gauss': 50        # Maximum allowable number of Gauss points — acts as a refinement cap
}