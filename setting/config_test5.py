# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # NumPy enables efficient numerical computations: array creation,
                    # mesh generation, time stepping, and math evaluations.


# ======================================================
# DOMAIN PARAMETERS
# ======================================================
# Defines the spatial and temporal boundaries of the simulation domain.

T = 1.0     # Total simulation time — defines time interval [0, T]
ell = 2.0   # Length of the beam — defines spatial domain [0, ell]


# ======================================================
# EQUATION COEFFICIENTS
# ======================================================
# Coefficients governing material properties and dynamic behavior
# in the nonlinear Timoshenko beam equations.

alpha = 1.0   # Damping coefficient in displacement equation (u)
beta  = 1.0   # Nonlinear stiffness multiplier in u-equation
gamma = 1.0   # Elastic stiffness in rotation equation (v)
delta = 1.0   # Damping coefficient in v-equation
a1    = 1.0   # Coupling from ∂v/∂x to u-equation (shear)
a2    = 1.0   # Coupling from ∂u/∂x to v-equation (shear)


# ======================================================
# BENCHMARK SOLUTION CONFIGURATION: LEGENDRE BASIS
# ======================================================
# Polynomial degrees and time-scaling factors used in analytical (manufactured) solution.

m_u = 20       # Degree of Legendre polynomial for spatial component of u(x, t)
m_v = 20       # Degree of Legendre polynomial for spatial component of v(x, t)

pow_u = 5      # Polynomial degree in time for displacement field u(x, t)
pow_v = 5      # Polynomial degree in time for rotation field v(x, t)

coeff_u = 1 / 16.0  # Coefficient scaling time polynomial for u(x, t)
coeff_v = 1 / 16.0  # Coefficient scaling time polynomial for v(x, t)

# Use the maximum degree for deciding default spectral resolution
degree_max = max(m_u, m_v)


# ======================================================
# TEMPORAL DISCRETIZATION
# ======================================================
# Defines the time grid and step size for the simulation.

n = 256                        # Number of time intervals (uniform)
t = np.linspace(0, T, n + 1)   # Time grid with n+1 points from 0 to T
tau = T / n                    # Time step size τ = T / n


# ======================================================
# GALERKIN METHOD CONFIGURATION
# ======================================================
# Function to determine the number of spectral modes (basis functions)
# for spatial discretization using Legendre polynomials.

def set_N(input_N: int = None) -> int:
    """
    Determine the number of Legendre basis functions for the Galerkin approximation.

    Parameters
    ----------
    input_N : int, optional
        User-specified override for number of spectral modes. Must be positive.

    Returns
    -------
    int
        Number of basis functions to use.

    Raises
    ------
    TypeError
        If input_N is not an integer.
    ValueError
        If input_N is non-positive.
    """
    if input_N is not None:
        if not isinstance(input_N, int):
            raise TypeError("input_N must be an integer.")
        if input_N <= 0:
            raise ValueError("input_N must be a positive integer.")
        return input_N

    # Default: ensure sufficient resolution for benchmark polynomial degree
    return max(2, degree_max)

# Number of basis functions for spatial projection (spectral resolution)
N = set_N()


# ======================================================
# QUADRATURE CONFIGURATION
# ======================================================
# Parameters for adaptive Gauss–Legendre integration.
# Used in inner products and projection integrals during Galerkin assembly.

quad_kwargs = {
    'tol': 1e-6,           # Absolute error tolerance for adaptive quadrature
    'min_dx': 1 / 128.0,   # Minimum width of subinterval during recursive refinement
    'n_gauss': 5,          # Initial number of Gauss–Legendre quadrature points per segment
    'max_gauss': 50        # Cap on refinement to avoid excessive subdivisions
}