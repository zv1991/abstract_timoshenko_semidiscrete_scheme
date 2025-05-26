import numpy as np  # For efficient numerical operations on arrays

""" The coefficients arising from the inner products of Legendre polynomials
    and the application of the Gauss–Legendre spectral method """

def coeff_A(m: int) -> float:
    """Compute A_m = 1 / sqrt(2m + 1)."""
    if not isinstance(m, int) or m < 0:
        raise ValueError("m must be a non-negative integer.")
    return 1 / np.sqrt(2 * m + 1)

def coeff_B(m: int) -> float:
    """Compute B_m = A_(m-1) * A_m^2 * A_(m+1)."""
    if m <= 0:
        raise ValueError("m must be greater than 0.")
    A_m1, A_m, A_m2 = coeff_A(m - 1), coeff_A(m), coeff_A(m + 1)
    return A_m1 * (A_m ** 2) * A_m2

def coeff_C(m: int) -> float:
    """Compute C_m = 2 * A_(m-1)^2 * A_(m+1)^2."""
    if m <= 0:
        raise ValueError("m must be greater than 0.")
    A_m1, A_m2 = coeff_A(m - 1), coeff_A(m + 1)
    return 2 * (A_m1 ** 2) * (A_m2 ** 2)