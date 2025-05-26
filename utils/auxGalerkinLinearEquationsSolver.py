import numpy as np  # For efficient numerical operations on arrays
from auxGaussLegendreCoeff import coeff_B, coeff_C

""" Solver function for the derived subsystem of the Galerkin linear system """

def sys_soln(f: np.ndarray, N: int, a: float, b: float, ell: float) -> np.ndarray:
    """
    Solves a tridiagonal system of equations using a specialized forward elimination 
    and backward substitution method tailored to a spectral problem.

    Parameters:
        f (np.ndarray): Right-hand side vector of shape (N,).
        N (int): Number of equations (must be >= 2).
        a (float): Coefficient in the system matrix.
        b (float): Coefficient in the system matrix.
        ell (float): Scaling parameter related to domain length or physical context.

    Returns:
        np.ndarray: Solution vector `w` of shape (N,).
    
    Raises:
        ValueError: If N is less than 2.
    """
    
    if N < 2:
        raise ValueError("N must be at least 2 for the system to be solvable.")

    # Allocate working arrays
    d = np.empty(N)  # Diagonal of the modified matrix
    z = np.empty(N)  # Modified right-hand side
    w = np.empty(N)  # Solution vector

    # Precompute first two diagonal entries
    d[0] = coeff_C(1) + (4 * b) / (a * ell ** 2)
    d[1] = coeff_C(2) + (4 * b) / (a * ell ** 2)

    # Copy first two entries from RHS
    z[0] = f[0]
    z[1] = f[1]

    # Compute up to the midpoint (since it's symmetric/stencil-based)
    half_N = (N + 1) // 2  # Works for odd and even N

    # --- Forward elimination ---
    for j in range(2, half_N + 1):
        idx = 2 * (j - 1)  # Access even/odd pairs

        if idx < N:
            # Update even-indexed diagonal and RHS
            d[idx] = (
                coeff_C(idx + 1) + (4 * b) / (a * ell ** 2) 
                - (coeff_B(idx) ** 2) / d[idx - 2]
            )
            z[idx] = f[idx] + (coeff_B(idx) * z[idx - 2]) / d[idx - 2]

        if idx + 1 < N:
            # Update odd-indexed diagonal and RHS
            d[idx + 1] = (
                coeff_C(idx + 2) + (4 * b) / (a * ell ** 2)
                - (coeff_B(idx + 1) ** 2) / d[idx - 1]
            )
            z[idx + 1] = f[idx + 1] + (coeff_B(idx + 1) * z[idx - 1]) / d[idx - 1]

    # --- Backward substitution ---
    w[-1] = z[-1] / d[-1]
    w[-2] = z[-2] / d[-2]

    for j in range(half_N - 1, 0, -1):
        idx = 2 * (j - 1)

        if idx + 2 < N:
            # Back-substitute even index
            w[idx] = (z[idx] + coeff_B(idx + 2) * w[idx + 2]) / d[idx]

        if idx + 3 < N:
            # Back-substitute odd index
            w[idx + 1] = (z[idx + 1] + coeff_B(idx + 3) * w[idx + 3]) / d[idx + 1]

    return w