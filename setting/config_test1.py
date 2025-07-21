# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # NumPy is used for numerical operations like array creation and time discretization


# ======================================================
# DOMAIN PARAMETERS
# ======================================================

T = 1.0     # Total simulation time — defines the time interval [0, T]
ell = 2.0   # Length of the beam — defines the spatial domain [0, ell]


# ======================================================
# EQUATION COEFFICIENTS
# ======================================================

# These constants define the physical and material properties
# used in the nonlinear Timoshenko beam equations.

alpha = 1.0   # Stiffness factor for displacement u (related to damping/memory effect)
beta  = 1.0   # Nonlinear damping coefficient in the u-equation
gamma = 1.0   # Linear stiffness for the rotation equation (v component)
delta = 1.0   # Damping coefficient in the v-equation (controls rotational energy dissipation)
a1    = 1.0   # Coupling term: gradient of v contributes to the u-equation
a2    = 1.0   # Coupling term: gradient of u contributes to the v-equation


# ======================================================
# LEGENDRE POLYNOMIAL CONFIGURATION FOR BENCHMARK SOLUTION
# ======================================================

# These represent the degree of Legendre polynomials used to construct
# analytical benchmark solutions for u(x, t) and v(x, t), respectively.

m_u = 2  # Degree of shifted Legendre polynomial used for the displacement field u(x, t)
m_v = 2  # Degree of shifted Legendre polynomial used for the rotation field v(x, t)


# ======================================================
# TEMPORAL DISCRETIZATION
# ======================================================

n = 5                          # Number of uniform time steps in [0, T]
t = np.linspace(0, T, n + 1)   # Array of time points: [t₀, t₁, ..., tₙ]
tau = T / n                    # Time step size τ = T / n


# ======================================================
# SPECTRAL METHOD CONFIGURATION
# ======================================================

# Number of spectral basis functions (Legendre polynomials) used in the Galerkin method.
# A higher value improves spatial accuracy at the cost of computational complexity.

N = 2  # Number of Legendre basis functions for Galerkin projection