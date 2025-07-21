# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # NumPy is used for numerical operations like array creation and time discretization


# ======================================================
# DOMAIN PARAMETERS
# ======================================================

T = 1.0     # Total simulation time — sets the time domain [0, T]
ell = 2.0   # Beam length — defines the spatial domain [0, ell]


# ======================================================
# EQUATION COEFFICIENTS
# ======================================================

# These physical parameters define material behavior and coupling in the Timoshenko beam system.
# They influence wave propagation, damping, and interaction between displacement and rotation.

alpha = 1.0   # Material stiffness in the u-equation (memory effect / elastic response)
beta  = 1.0   # Nonlinear damping coefficient for u (dependent on gradient magnitude)
gamma = 1.0   # Linear stiffness in the v-equation (rotational restoring force)
delta = 1.0   # Damping in the v-equation (controls dissipation of rotational energy)
a1    = 1.0   # Coupling coefficient: ∂v/∂x appears in u-equation
a2    = 1.0   # Coupling coefficient: ∂u/∂x appears in v-equation


# ======================================================
# OSCILLATION PARAMETERS FOR BENCHMARK SOLUTIONS
# ======================================================

# These define the number of oscillations in the trigonometric test solutions u(x, t), v(x, t).
# Higher values create more complex test shapes and can stress numerical accuracy.

lam_u = 5  # Number of spatial oscillations in displacement u(x, t)
lam_v = 5  # Number of spatial oscillations in rotation v(x, t)


# ======================================================
# TEMPORAL DISCRETIZATION
# ======================================================

n = 128                        # Number of uniform time intervals in [0, T]
t = np.linspace(0, T, n + 1)   # Array of n+1 time points: [t₀, t₁, ..., tₙ]
tau = T / n                    # Time step size τ = T / n


# ======================================================
# SPECTRAL METHOD CONFIGURATION
# ======================================================

# Number of Legendre polynomials used as spatial basis functions in the Galerkin method.
# Increasing N improves spatial resolution but increases computational cost.

N = 15  # Number of basis functions (modes) used in the Galerkin projection