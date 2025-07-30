# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # NumPy: numerical computing library for array manipulation,
                    # mesh generation, and evaluating mathematical expressions.


# ======================================================
# SIMULATION DOMAIN CONFIGURATION
# ======================================================
# Defines the spatial and temporal extent of the domain for solving the PDEs.

T = 2.0     # Final simulation time → time domain is [0, T]
ell = 2.0   # Physical length of the beam → spatial domain is [0, ell]


# ======================================================
# MODEL PHYSICAL PARAMETERS
# ======================================================
# These constants define material and system behavior in the nonlinear
# Timoshenko beam model. They appear in the PDE system and affect its dynamics.

alpha = 1.0   # Damping coefficient in the displacement (u) equation (memory effect)
beta  = 1.0   # Nonlinear stiffness coefficient — scales contribution of (∂u/∂x)²
gamma = 1.0   # Stiffness in the rotational (v) equation — affects bending
delta = 1.0   # Damping in the rotational (v) equation — energy dissipation
a1    = 1.0   # Coupling from ∂v/∂x to u-equation (shear deformation contribution)
a2    = 1.0   # Coupling from ∂u/∂x to v-equation (shear deformation feedback)


# ======================================================
# ANALYTICAL (MANUFACTURED) SOLUTION PARAMETERS
# ======================================================
# Parameters used to construct symbolic benchmark solutions for verification.
# These are separable functions of space and time, designed to satisfy the PDE system exactly.

lam_u = 14  # Number of sine wave oscillations in spatial profile of u(x, t)
lam_v = 14  # Number of sine wave oscillations in spatial profile of v(x, t)

pow_coeff_u = 1 / 8.0  # Exponent for time-dependent growth/decay in u(x, t)
pow_coeff_v = 1 / 8.0  # Exponent for time-dependent growth/decay in v(x, t)

mult_coeff_u = 1 / 8.0  # Amplitude multiplier on time dependence for u(x, t)
mult_coeff_v = 1 / 8.0  # Amplitude multiplier on time dependence for v(x, t)


# ======================================================
# TEMPORAL DISCRETIZATION PARAMETERS
# ======================================================
# Define the resolution of the time grid used for stepping forward in time.

n = 512                         # Number of uniform time steps
t = np.linspace(0, T, n + 1)    # Discrete time points: t₀ = 0, tₙ = T
tau = T / n                     # Time step size τ (used in solvers for advancing state)


# ======================================================
# SPECTRAL GALERKIN METHOD PARAMETERS
# ======================================================
# Sets spatial resolution for the Galerkin spectral method.
# The solution is projected onto a basis of Legendre polynomials of degree < N.

N = 35  # Number of Legendre polynomial basis functions for spatial approximation


# ======================================================
# NUMERICAL INTEGRATION SETTINGS FOR PROJECTION
# ======================================================
# These parameters configure the adaptive Gauss–Legendre quadrature method
# used for evaluating integrals during projection (e.g., stiffness/mass matrices, forcing terms).

quad_kwargs = {
    'tol': 1e-6,           # Absolute error tolerance for stopping criterion in adaptive integration
    'min_dx': 1 / 128.0,   # Minimum subinterval length to avoid infinite subdivision
    'n_gauss': 5,          # Initial number of Gauss nodes per interval (usually 3–5)
    'max_gauss': 50        # Maximum number of Gauss points allowed (prevents runaway refinement)
}