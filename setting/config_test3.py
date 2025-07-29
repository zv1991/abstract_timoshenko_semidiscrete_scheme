# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # NumPy is used for numerical operations like array creation, discretization, and math functions


# ======================================================
# DOMAIN PARAMETERS
# ======================================================
# These parameters define the physical domain for simulation:
# T: total simulation time (temporal domain)
# ell: length of the beam (spatial domain)

T = 1.0     # Total simulation time → defines time interval [0, T]
ell = 2.0   # Beam length → defines spatial interval [0, ell]


# ======================================================
# EQUATION COEFFICIENTS
# ======================================================
# Coefficients controlling material behavior and system dynamics in the Timoshenko model.

alpha = 1.0   # Elastic stiffness in displacement equation u(x, t)
beta  = 1.0   # Nonlinear damping in u(x, t), scales with strain energy (∂u/∂x)^2
gamma = 1.0   # Rotational stiffness in rotation equation v(x, t)
delta = 1.0   # Damping in v(x, t) — dissipates rotational energy
a1    = 1.0   # Coupling term: ∂v/∂x feeds into u-equation (shear interaction)
a2    = 1.0   # Coupling term: ∂u/∂x feeds into v-equation (shear interaction)


# ======================================================
# OSCILLATION PARAMETERS FOR BENCHMARK SOLUTIONS
# ======================================================
# These parameters define how many spatial oscillations appear in the benchmark
# solutions u(x, t) and v(x, t). Higher values create finer wave structures.

lam_u = 14  # Spatial frequency (number of sine wave peaks) in displacement field u
lam_v = 14  # Spatial frequency in rotation field v


# ======================================================
# TEMPORAL DISCRETIZATION
# ======================================================
# Setup for uniform time discretization used in numerical integration/solving.

n = 256                        # Number of uniform time intervals in [0, T]
t = np.linspace(0, T, n + 1)   # Time grid: [t₀, t₁, ..., tₙ]; includes both endpoints
tau = T / n                    # Time step size: τ = (T - 0) / n


# ======================================================
# SPECTRAL METHOD CONFIGURATION
# ======================================================
# Settings for the Galerkin spectral method using Legendre polynomials as basis functions.

N = 35  # Number of Legendre polynomial modes (basis functions) in spatial projection


# ======================================================
# QUADRATURE CONFIGURATION
# ======================================================
# Controls adaptive Gauss–Legendre quadrature used in numerical integration
# (e.g., projection of symbolic functions onto basis functions).

quad_kwargs = {
    'tol': 1e-6,           # Absolute tolerance for quadrature convergence
    'min_dx': 1 / 128.0,   # Minimum interval size before subdivision halts
    'n_gauss': 5,          # Initial number of Gauss–Legendre points per subinterval
    'max_gauss': 50        # Upper limit to avoid excessive refinement in quadrature
}