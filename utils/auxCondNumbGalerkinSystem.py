from utils.auxLegendreGalerkinSpectralMethod import associated_operators
from scipy.sparse import identity  # For sparse matrix construction
from numpy.linalg import cond  # Computes condition number using 2-norm

# --- Condition Number Computation ---
def condition_number_associated_matrix(N: int, ell: float, a: float, b: float) -> float:
    """
    Compute the condition number κ₂(A) of a modified Galerkin matrix:
        A = H + (4b / (a * ell²)) * I

    Parameters:
        N (int): Matrix size.
        ell (float): Scaling parameter.
        a (float): Physical coefficient.
        b (float): Additive coefficient.

    Returns:
        float: 2-norm condition number of the matrix.
    """
    # Galerkin identity operator H
    H = associated_operators(N, operator="identity")

    # Identity matrix scaled by constant
    scalar = (4 * b) / (a * ell ** 2)
    I_scaled = scalar * identity(N, format="csr")

    # Final matrix A
    A = H + I_scaled

    # Convert to dense for condition number computation
    return cond(A.toarray(), p=2)