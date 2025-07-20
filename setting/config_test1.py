# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # NumPy is used for numerical operations, including array generation and time discretization


# ======================================================
# DOMAIN PARAMETERS
# ======================================================

T = 1.0      # Total simulation time — defines the time interval [0, T]
ell = 2.0    # Length of the beam — defines the spatial domain [0, ell]


# ======================================================
# EQUATION COEFFICIENTS
# ======================================================

# These constants define the physical behavior of the nonlinear Timoshenko beam model:
# They govern damping, stiffness, and the coupling between displacement (u) and rotation (v).

alpha = 1.0   # Material stiffness factor for u-equation (memory-type damping)
beta  = 1.0   # Nonlinear damping factor for the displacement u
gamma = 1.0   # Linear stiffness coefficient for the rotation v
delta = 1.0   # Damping coefficient in the v-equation
a1    = 1.0   # Coupling: ∂v/∂x appears in u-equation
a2    = 1.0   # Coupling: ∂u/∂x appears in v-equation


# ======================================================
# TEMPORAL DISCRETIZATION
# ======================================================

n = 128                       # Number of time steps (subintervals of [0, T])
t = np.linspace(0, T, n + 1)  # Discrete time points t₀, t₁, ..., tₙ ∈ [0, T]
tau = T / n                   # Time step size τ = T / n


# ======================================================
# SPECTRAL METHOD CONFIGURATION
# ======================================================

N = 3  # Number of Legendre basis functions used in Galerkin projection
       # Controls spatial resolution: higher N = better approximation, more cost