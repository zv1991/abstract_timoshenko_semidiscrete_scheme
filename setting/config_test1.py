# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # NumPy is used for efficient numerical computations including
                    # array operations, time discretization, and function evaluations.


# ======================================================
# DOMAIN PARAMETERS
# ======================================================
# These parameters define the simulation's spatial and temporal domains.

T = 1.0     # Total simulation duration — defines time interval [0, T]
ell = 2.0   # Length of the beam — defines spatial domain [0, ell]


# ======================================================
# EQUATION COEFFICIENTS
# ======================================================
# Constants defining the material behavior and damping characteristics
# of the nonlinear Timoshenko beam model.

alpha = 1.0   # Memory-type damping in the displacement (u) equation
beta  = 1.0   # Nonlinear damping for the displacement field (u)
gamma = 1.0   # Stiffness coefficient in the rotation equation (v)
delta = 1.0   # Linear damping in the rotation (v) equation
a1    = 1.0   # Coupling term in u-equation involving ∂v/∂x
a2    = 1.0   # Coupling term in v-equation involving ∂u/∂x


# ======================================================
# LEGENDRE POLYNOMIAL CONFIGURATION FOR BENCHMARK SOLUTION
# ======================================================
# These values are used to construct a known analytical solution (manufactured solution)
# via shifted Legendre polynomials of specified degrees.

m_u = 2  # Degree of Legendre polynomial for displacement u(x, t)
m_v = 2  # Degree of Legendre polynomial for rotation    v(x, t)

# Use the maximum degree to determine minimum necessary spectral resolution
degree_max = max(m_u, m_v)  # Ensures enough basis functions to resolve both fields


# ======================================================
# TEMPORAL DISCRETIZATION
# ======================================================
# Discretize the time domain [0, T] using uniform intervals.

n = 5                          # Number of time steps; affects time resolution
t = np.linspace(0, T, n + 1)   # Time grid points array: t₀, t₁, ..., tₙ
tau = T / n                    # Uniform time step size τ


# ======================================================
# SPECTRAL METHOD CONFIGURATION
# ======================================================
# Configure the spatial discretization using a Legendre-Galerkin method.

# If degree_max is 1 (minimal), use N=2 to avoid poor resolution
N_default = 2 if degree_max == 1 else degree_max

N = N_default  # Number of Legendre polynomial basis functions for spatial discretization;
               # higher N increases spatial accuracy but also computational load
