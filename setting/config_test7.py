# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # NumPy is used for numerical computations, such as array operations, discretization, etc.


# ======================================================
# DOMAIN PARAMETERS
# ======================================================
# Define spatial and temporal boundaries for simulation.

T = 1.0      # Total simulation time; time interval: [0, T]
ell = 2.0    # Length of the beam; spatial interval: [0, ell]


# ======================================================
# EQUATION COEFFICIENTS
# ======================================================
# Physical coefficients for the Timoshenko beam model.
# These affect stiffness, damping, and coupling between translation and rotation.

alpha = 10.0  # Stiffness coefficient for translational displacement u(x, t)
beta  = 0.25  # Nonlinear damping coefficient in u-equation (dissipation based on gradients)
gamma = 10.0  # Rotational stiffness coefficient in v-equation (restoring torque)
delta = 1.0  # Linear damping coefficient in v-equation (energy loss through damping)
a1    = 1.0  # Coupling coefficient: ∂v/∂x in u-equation (rotation affects displacement)
a2    = 1.0  # Coupling coefficient: ∂u/∂x in v-equation (displacement affects rotation)


# ======================================================
# OSCILLATION PARAMETERS FOR BENCHMARK SOLUTIONS
# ======================================================
# Parameters for constructing analytical or test solutions,
# usually Gaussian modulated sine/cosine waves.

# Spatial oscillation parameters
lam_u = 19   # Number of spatial oscillations for u(x, t)
lam_v = 19   # Number of spatial oscillations for v(x, t)

# Temporal oscillation parameters
lam1_u = 2   # Number of temporal oscillations for u(x, t)
lam1_v = 2   # Number of temporal oscillations for v(x, t)

# Amplitude parameters
A_u = 1.0   # Amplitude of Gaussian profile for u(x, t)
A_v = 1.0   # Amplitude of Gaussian profile for v(x, t)

# Width of Gaussian envelope (smaller = narrower)
c_u = 1.0    # Gaussian width parameter for u(x, t)
c_v = 1.0    # Gaussian width parameter for v(x, t)


# ======================================================
# TEMPORAL DISCRETIZATION
# ======================================================
# Define time grid for numerical integration over [0, T]

n = 256                        # Number of time intervals
t = np.linspace(0, T, n + 1)   # Discretized time points (n+1 total, including endpoints)
tau = T / n                   # Time step size Δt (uniform steps)


# ======================================================
# SPECTRAL METHOD CONFIGURATION
# ======================================================
# Configuration for Galerkin spectral projection using Legendre polynomials.

N = 45  # Number of basis functions (Legendre polynomials) used for spatial approximation


# ======================================================
# QUADRATURE CONFIGURATION
# ======================================================
# Settings for adaptive Gauss–Legendre quadrature used to compute integrals
# accurately in the spectral method (e.g., mass, stiffness matrices).

quad_kwargs = {
    'tol': 1e-6,          # Absolute error tolerance for adaptive quadrature
    'min_dx': 1 / 128.0,  # Minimum subinterval width (avoids excessive subdivision)
    'n_gauss': 5,         # Initial number of Gauss points per subinterval
    'max_gauss': 50       # Maximum number of Gauss points allowed (prevents overload)
}