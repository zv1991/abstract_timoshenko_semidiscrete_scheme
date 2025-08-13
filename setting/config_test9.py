"""
Timoshenko beam benchmark – configuration (comments-only refinement)
-------------------------------------------------------------------
Purpose:
- Define physical domain, model coefficients, discretization, and quadrature
  controls used by the solver/assembly code elsewhere.

Notes:
- This file contains *only constants*; no functions/methods are defined.
- All executable assignments remain unchanged; improvements are purely comments/structure.
"""

# ======================================================
# MODULE IMPORTS  (Title: Dependencies)
# ======================================================
# External libraries required by this configuration module.
# Keep imports minimal to reduce startup time and avoid unnecessary dependency chains.

import numpy as np  # NumPy is used for numerical operations like array creation, discretization, and math functions


# ======================================================
# DOMAIN PARAMETERS  (Title: Physical domain extents)
# ======================================================
# These parameters define the physical domain for simulation:
# T: total simulation time (temporal domain)
# ell: length of the beam (spatial domain)
#
# TIP: Keep these strictly positive to avoid degenerate grids or undefined steps.

T = 1.0     # Total simulation time → defines time interval [0, T]
ell = 2.0   # Beam length → defines spatial interval [0, ell]


# ======================================================
# EQUATION COEFFICIENTS  (Title: Model constants)
# ======================================================
# Coefficients controlling material behavior and system dynamics in the Timoshenko model.
# Typical usage: passed into assembly/solver routines to form operators and source terms.

alpha = 1.0   # Elastic stiffness in displacement equation u(x, t)
beta  = 1.0   # Nonlinear damping in u(x, t), scales with strain energy (∂u/∂x)^2
gamma = 1.0   # Rotational stiffness in rotation equation v(x, t)
delta = 1.0   # Damping in v(x, t) — dissipates rotational energy
a1    = 1.0   # Coupling term: ∂v/∂x feeds into u-equation (shear interaction)
a2    = 1.0   # Coupling term: ∂u/∂x feeds into v-equation (shear interaction)


# ======================================================
# OSCILLATION PARAMETERS FOR BENCHMARK SOLUTIONS  (Title: Analytic frequencies)
# ======================================================
# These parameters define how many spatial oscillations appear in the benchmark
# solutions u(x, t) and v(x, t). Higher values create finer wave structures.
# If you increase these, consider increasing N below to avoid spatial under-resolution.

lam_u = 14  # Spatial frequency (number of sine wave peaks) in displacement field u
lam_v = 14  # Spatial frequency in rotation field v


# ======================================================
# TEMPORAL DISCRETIZATION  (Title: Uniform time grid)
# ======================================================
# Setup for uniform time discretization used in numerical integration/solving.
# n   : number of sub-intervals; larger n → smaller τ (time step) and higher temporal resolution
# t   : array of n+1 samples spanning [0, T]; includes both endpoints
# tau : time step size; avoid n = 0 to prevent division by zero

n = 256                        # Number of uniform time intervals in [0, T]
t = np.linspace(0, T, n + 1)   # Time grid: [t₀, t₁, ..., tₙ]; includes both endpoints
tau = T / n                    # Time step size: τ = (T - 0) / n


# ======================================================
# SPECTRAL METHOD CONFIGURATION  (Title: Spatial approximation)
# ======================================================
# Settings for the Galerkin spectral method using Legendre polynomials as basis functions.
# Increasing N typically improves accuracy but raises computational cost (e.g., matrix sizes).

N = 45  # Number of Legendre polynomial modes (basis functions) in spatial projection


# ======================================================
# QUADRATURE CONFIGURATION  (Title: Adaptive Gauss–Legendre controls)
# ======================================================
# Controls adaptive Gauss–Legendre quadrature used in numerical integration
# (e.g., projection of symbolic functions onto basis functions).
# Lower 'tol' increases accuracy but may trigger more refinement up to 'max_gauss'.

quad_kwargs = {
    'tol': 1e-6,           # Absolute tolerance for quadrature convergence
    'min_dx': 1 / 128.0,   # Minimum interval size before subdivision halts
    'n_gauss': 5,          # Initial number of Gauss–Legendre points per subinterval
    'max_gauss': 50        # Upper limit to avoid excessive refinement in quadrature
}