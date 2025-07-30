# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # NumPy is used for efficient numerical computations including
                    # array operations, time discretization, and function evaluations.


# ======================================================
# DOMAIN PARAMETERS
# ======================================================
# These parameters define the simulation's spatial and temporal domains.

T = 2.0     # Total simulation duration — defines time interval [0, T]
ell = 2.0   # Length of the beam — defines spatial domain [0, ell]


# ======================================================
# EQUATION COEFFICIENTS
# ======================================================
# Constants defining the material behavior and damping characteristics
# of the nonlinear Timoshenko beam model.

alpha = 1.0   # Memory-type damping in the displacement (u) equation
beta  = 1.0   # Nonlinear stiffness factor for the displacement field (u)
gamma = 1.0   # Stiffness coefficient in the rotation equation (v)
delta = 1.0   # Linear damping in the rotation (v) equation
a1    = 1.0   # Coupling coefficient: ∂v/∂x term in u-equation
a2    = 1.0   # Coupling coefficient: ∂u/∂x term in v-equation


# ======================================================
# LEGENDRE POLYNOMIAL CONFIGURATION FOR BENCHMARK SOLUTION
# ======================================================
# These values are used to construct a known analytical solution (manufactured solution)
# via shifted Legendre polynomials of specified degrees.

m_u = 3  # Degree of Legendre polynomial for displacement u(x, t)
m_v = 5  # Degree of Legendre polynomial for rotation v(x, t)

# Use the maximum degree to determine minimum necessary spectral resolution
degree_max = max(m_u, m_v)  # Ensures enough basis functions to resolve both fields


# ======================================================
# TEMPORAL DISCRETIZATION
# ======================================================
# Discretize the time domain [0, T] using uniform intervals.

n = 64                         # Number of time steps; affects time resolution
t = np.linspace(0, T, n + 1)   # Time grid points array: t₀, t₁, ..., tₙ
tau = T / n                    # Uniform time step size τ


# ======================================================
# GALERKIN SPECTRAL METHOD CONFIGURATION
# ======================================================
# Selects number of Legendre basis functions used in Galerkin projection.

def set_N(input_N: int = None) -> int:
    """
    Determine number of Legendre basis functions for Galerkin approximation.

    Parameters
    ----------
    input_N : int, optional
        User-specified override for number of basis functions.

    Returns
    -------
    int
        Final number of basis functions to use.

    Raises
    ------
    TypeError
        If input_N is provided and not an integer.
    ValueError
        If input_N is non-positive.
    """
    if input_N is not None:
        if not isinstance(input_N, int):
            raise TypeError("input_N must be an integer.")
        if input_N <= 0:
            raise ValueError("input_N must be a positive integer.")
        return input_N

    # Default: use enough modes to capture highest polynomial degree in benchmark
    return 2 if degree_max == 1 else degree_max

# Assign basis function count
N = set_N()  # Spatial resolution parameter for Legendre basis


# ======================================================
# QUADRATURE CONFIGURATION
# ======================================================
# Controls settings for adaptive Gauss–Legendre integration used in projections.

quad_kwargs = {
    'tol': 1e-6,           # Absolute integration tolerance (controls convergence)
    'min_dx': 1 / 128.0,   # Minimum subinterval width (prevents oversubdivision)
    'n_gauss': 5,          # Starting number of Gauss nodes per subinterval
    'max_gauss': 50        # Maximum Gauss nodes allowed during adaptive refinement
}