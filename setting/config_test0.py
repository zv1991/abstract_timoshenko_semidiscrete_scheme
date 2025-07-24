# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # NumPy: Used for numerical operations, especially efficient array handling.
                    # Essential for domain discretization, time grid creation, and polynomial operations.

# ======================================================
# DOMAIN PARAMETERS
# ======================================================
# These parameters define the spatial and temporal domains over which
# the Timoshenko beam model will be simulated.

T = 1.0      # Final simulation time; defines the upper time boundary [0, T]
ell = 2.0    # Length of the beam; defines the spatial domain [0, ℓ]

# ======================================================
# EQUATION COEFFICIENTS
# ======================================================
# Coefficients representing physical properties and damping effects
# in the nonlinear Timoshenko beam system.

alpha = 1.0  # Memory-type damping in the displacement equation (u)
beta  = 1.0  # Nonlinear damping on displacement (u)
gamma = 1.0  # Rotational stiffness coefficient (v equation)
delta = 1.0  # Damping coefficient in rotational equation (v)
a1    = 1.0  # Coupling: ∂v/∂x contribution in u-equation
a2    = 1.0  # Coupling: ∂u/∂x contribution in v-equation

# ======================================================
# BENCHMARK SOLUTION POLYNOMIAL DEGREES
# ======================================================
# Defines degrees of manufactured polynomial solutions for u(x) and v(x)
# used in method validation (e.g., convergence tests).

m1_u, m2_u = 10, 5  # u(x) ~ x^m1_u * (ℓ - x)^m2_u
m1_v, m2_v = 5, 10  # v(x) ~ x^m1_v * (ℓ - x)^m2_v

# Determine the maximum total polynomial degree
degree_max = max(m1_u + m2_u, m1_v + m2_v)  # Ensures sufficient resolution for basis functions

# ======================================================
# TEMPORAL DISCRETIZATION
# ======================================================
# Constructs a uniform grid in the time domain [0, T] with n intervals.

n = 128                       # Number of time intervals (i.e., steps)
t = np.linspace(0, T, n + 1)  # Time grid: n+1 points including endpoints
tau = T / n                   # Time step size (Δt = T / n)

# ======================================================
# SPECTRAL METHOD CONFIGURATION
# ======================================================
# The Galerkin spectral method is used for spatial discretization.
# This section selects the number of Legendre basis functions.

# -----------------------------
# FUNCTION: set_N
# -----------------------------
def set_N(input_N: int = None) -> int:
    """
    Determine the number of Legendre basis functions (N) to be used
    in the Galerkin approximation of the Timoshenko beam model.

    Parameters:
        input_N (int, optional): Optional user-defined number of basis functions.

    Returns:
        int: Number of basis functions selected for the simulation.

    Raises:
        TypeError: If input_N is provided but not an integer.
        ValueError: If input_N is non-positive.
    """

    # User-specified override with input validation
    if input_N is not None:
        if not isinstance(input_N, int):
            raise TypeError("input_N must be an integer.")
        if input_N <= 0:
            raise ValueError("input_N must be a positive integer.")
        return input_N

    # Default rule based on benchmark polynomial degrees
    return 2 if degree_max == 1 else degree_max

# Select number of Legendre basis functions
N = set_N()  # Determines spatial resolution in the spectral Galerkin solver.
             # Increasing N improves accuracy at the cost of computation.