# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # NumPy: Core numerical library for arrays, grids, and computations


# ======================================================
# DOMAIN PARAMETERS
# ======================================================
# Defines simulation bounds for time and space domains.

T = 1.0      # Final time for simulation; upper bound of time interval [0, T]
ell = 2.0    # Beam length; defines spatial domain [0, ℓ]


# ======================================================
# EQUATION COEFFICIENTS
# ======================================================
# Coefficients controlling physical behavior and coupling in Timoshenko beam equations.

alpha = 1.0  # Damping in the displacement (u) equation (memory-type)
beta  = 1.0  # Nonlinear stiffness scaling in the u-equation
gamma = 1.0  # Stiffness coefficient in the rotational (v) equation
delta = 1.0  # Damping coefficient in the v-equation
a1    = 1.0  # Coupling coefficient: ∂v/∂x term in u-equation
a2    = 1.0  # Coupling coefficient: ∂u/∂x term in v-equation


# ======================================================
# BENCHMARK POLYNOMIAL DEGREES
# ======================================================
# Defines the spatial polynomial structure of the analytical benchmark solution.

m1_u, m2_u = 10, 5  # Spatial powers for u(x): h_u(x) = x^m1_u * (ℓ - x)^m2_u
m1_v, m2_v = 5, 10  # Spatial powers for v(x): h_v(x) = x^m1_v * (ℓ - x)^m2_v

# Compute the maximum total polynomial degree used (for resolution/basis guidance)
degree_max = max(m1_u + m2_u, m1_v + m2_v)


# ======================================================
# TEMPORAL DISCRETIZATION
# ======================================================
# Creates a uniform time grid and defines time step size.

n = 128                        # Number of time intervals
t = np.linspace(0, T, n + 1)   # Time grid of n+1 points (including endpoints)
tau = T / n                    # Time step size (Δt)


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