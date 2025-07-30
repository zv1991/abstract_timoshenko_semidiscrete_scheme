# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # NumPy enables efficient numerical computations: array creation,
                    # mesh generation, time stepping, and math evaluations.


# ======================================================
# DOMAIN PARAMETERS
# ======================================================
# Defines the spatial and temporal boundaries of the simulation domain.

T = 2.0     # Total simulation time — defines time interval [0, T]
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
# These define the degrees of Legendre polynomial basis functions
# and the polynomial powers of time in the manufactured solution.

m_u = 35      # Degree of Legendre polynomial for spatial shape of u(x, t)
m_v = 35      # Degree of Legendre polynomial for spatial shape of v(x, t)

pow_u = 5     # Power of t in temporal component of u(x, t)
pow_v = 5     # Power of t in temporal component of v(x, t)

# Choose the greater of m_u and m_v for spectral resolution guidance
degree_max = max(m_u, m_v)


# ======================================================
# TEMPORAL DISCRETIZATION
# ======================================================
# Uniform time grid for advancing the solution in time.

n = 512                        # Number of time intervals
t = np.linspace(0, T, n + 1)   # Time grid with n+1 points: [t₀, ..., tₙ]
tau = T / n                    # Uniform time step size


# ======================================================
# GALERKIN METHOD CONFIGURATION
# ======================================================
# Set the number of spectral modes (basis functions) for spatial approximation.

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

    # Auto-selection: minimum 2 modes unless benchmark requires more
    return max(2, degree_max)

# Determine spectral resolution
N = set_N()  # Number of Legendre modes (basis functions) in spatial projection


# ======================================================
# QUADRATURE CONFIGURATION
# ======================================================
# Parameters for adaptive Gauss–Legendre quadrature used in numerical projections.

quad_kwargs = {
    'tol': 1e-6,           # Integration convergence tolerance (absolute)
    'min_dx': 1 / 128.0,   # Smallest subinterval before stopping refinement
    'n_gauss': 5,          # Initial number of Gauss–Legendre points per subinterval
    'max_gauss': 50        # Upper bound on Gauss points to prevent over-refinement
}