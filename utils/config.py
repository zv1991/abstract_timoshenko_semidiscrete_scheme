import numpy as np  # NumPy is used for creating time grids and handling numerical arrays


# ---------------------------------------------------------------------------
# DOMAIN PARAMETERS
# ---------------------------------------------------------------------------

T = 1.0       # Total simulation time, defining the interval [0, T] for the temporal domain
ell = 2.0     # Length of the spatial domain, defining the beam extent [0, ell]


# ---------------------------------------------------------------------------
# EQUATION COEFFICIENTS
# ---------------------------------------------------------------------------

# Physical parameters for the nonlinear Timoshenko beam model.
# These coefficients govern energy dissipation, stiffness, and displacement-rotation coupling.

alpha = 1.0   # Damping due to memory effects in the displacement equation
beta  = 1.0   # Viscous damping in the displacement equation (u-equation)
gamma = 1.0   # Rotational stiffness in the rotation equation (v-equation)
delta = 1.0   # Damping in the rotation equation
a1    = 1.0   # Coupling: gradient of rotation (∂v/∂x) influences displacement
a2    = 1.0   # Coupling: gradient of displacement (∂u/∂x) influences rotation


# ---------------------------------------------------------------------------
# TEMPORAL DISCRETIZATION
# ---------------------------------------------------------------------------

n = 200                      # Number of uniform time intervals (steps)
t = np.linspace(0, T, n + 1) # Time grid with (n+1) equally spaced points from 0 to T
tau = T / n                  # Time step size τ = T / n


# ---------------------------------------------------------------------------
# SPECTRAL METHOD CONFIGURATION
# ---------------------------------------------------------------------------

N = 10  # Number of Legendre polynomial basis functions for the Galerkin method
        # Determines the spectral resolution in space (higher N → higher accuracy)