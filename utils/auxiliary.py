import numpy as np  # For efficient numerical operations on arrays
from numpy.polynomial.legendre import leggauss  # Efficient Gauss-Legendre nodes/weights
from scipy.sparse import identity, diags, csr_matrix  # For sparse matrix construction
from numpy.linalg import cond  # Computes condition number using 2-norm

""" Legendre-Galerkin Integral Projections and Initial Condition Processing for PDE Solvers """

def compute_initial_integrals(u, v, N, ell):
    """
    Project initial condition functions onto basis functions and compute spatial derivatives.
    
    Parameters:
        u   : list of functions [u0, u1], representing u(x, t=0) and ∂u/∂t at t=0
        v   : list of functions [v0, v1], same as above for v
        N   : int, number of basis functions
        ell : float, scaling parameter for the domain
    
    Returns:
        A dictionary containing:
            - u_proj      : [∫ u0*φ_m dx, ∫ u1*φ_m dx] for m = 1..N
            - v_proj      : [∫ v0*φ_m dx, ∫ v1*φ_m dx] for m = 1..N
            - diff1_u1    : ∫ du1/dx * φ_m dx = ∫ -u1 * φ_m' dx for m = 1..N
            - diff1_v1    : same for v1
            - diff2_u     : [[∫ d²u0/dx² * φ_m dx], [∫ d²u1/dx² * φ_m dx]]
            - diff2_v     : same for v
    """
    # Initialize projection arrays
    u_proj = [np.zeros(N), np.zeros(N)]     # Projections of u0 and u1
    v_proj = [np.zeros(N), np.zeros(N)]     # Projections of v0 and v1
    diff1_u1 = np.zeros(N)                  # First derivative of u1 projected
    diff1_v1 = np.zeros(N)                  # First derivative of v1 projected
    diff2_u = np.zeros((2, N))              # Second derivative terms for u0 and u1
    diff2_v = np.zeros((2, N))              # Second derivative terms for v0 and v1

    for m in range(N):
        m_idx = m + 1  # φ_m index starts at 1 in auxiliary functions

        # --- Projections of initial values onto φ_m(x) ---
        u_proj[0][m], _ = integrate_with_phi_m(u[0], m_idx, ell)  # u0
        u_proj[1][m], _ = integrate_with_phi_m(u[1], m_idx, ell)  # u1
        v_proj[0][m], _ = integrate_with_phi_m(v[0], m_idx, ell)  # v0
        v_proj[1][m], _ = integrate_with_phi_m(v[1], m_idx, ell)  # v1

        # --- First derivative: ∂/∂x of u1 and v1, projected via weak form ---
        diff1_u1[m], _ = adaptive_gauss_legendre(
            lambda x: -u[1](x) * normalized_shifted_legendre(m_idx, ell, x), ell
        )
        diff1_v1[m], _ = adaptive_gauss_legendre(
            lambda x: -v[1](x) * normalized_shifted_legendre(m_idx, ell, x), ell
        )

        # --- Second derivative terms from weak form using ∫ f' * φ'_m dx ---
        for i in range(2):
            diff2_u[i][m], _ = adaptive_gauss_legendre_integrate_fprime_leg(
                lambda x, i=i: -u[i](x), m_idx, ell
            )
            diff2_v[i][m], _ = adaptive_gauss_legendre_integrate_fprime_leg(
                lambda x, i=i: -v[i](x), m_idx, ell
            )

    # Return all components in a structured dict
    return {
        'u_proj': u_proj,
        'v_proj': v_proj,
        'diff1_u1': diff1_u1,
        'diff1_v1': diff1_v1,
        'diff2_u': diff2_u,
        'diff2_v': diff2_v,
    }

# --- Identity Operator Assembly ---
def associated_identity_operator(N: int) -> csr_matrix:
    """
    Assemble a symmetric mass-like identity operator with a 3-point stencil.

    Parameters:
        N (int): Number of basis functions (matrix size).

    Returns:
        csr_matrix: Sparse matrix representing the identity operator.
    """
    # Main diagonal: C(m+1)
    main_diag = np.array([coeff_C(m + 1) for m in range(N)])

    # Off-diagonals: -B(m+2), symmetric about ±2 diagonals
    off_diag = np.array([-coeff_B(m + 2) for m in range(N - 2)])

    # Create sparse matrix with diagonals at positions 0, ±2
    H = diags(
        diagonals=[main_diag, off_diag, off_diag],
        offsets=[0, -2, 2],
        shape=(N, N),
        format="csr"
    )
    return H

# --- First-Order Operator Assembly ---
def associated_first_order_operator(N: int) -> csr_matrix:
    """
    Assemble a skew-symmetric first-order derivative operator using a ±1 stencil.

    Parameters:
        N (int): Number of basis functions (matrix size).

    Returns:
        csr_matrix: Sparse matrix representing the first-order operator.
    """
    # Upper diagonal: A(m+1)*A(m+2), size (N-1)
    upper_diag = np.array([
        coeff_A(m + 1) * coeff_A(m + 2) for m in range(N - 1)
    ])

    # Lower diagonal: Negative of upper for skew-symmetry
    lower_diag = -upper_diag

    # Assemble sparse skew-symmetric matrix
    B = diags(
        diagonals=[lower_diag, upper_diag],
        offsets=[-1, 1],
        shape=(N, N),
        format="csr"
    )
    return B

# --- Operator Dispatcher ---
def associated_operators(N: int, operator: str) -> csr_matrix:
    """
    Dispatcher for assembling different Galerkin operator matrices.

    Parameters:
        N (int): Matrix size (number of basis functions).
        operator (str): Operator type: "identity" or "first-order".

    Returns:
        csr_matrix: Assembled sparse operator matrix.
    
    Raises:
        ValueError: If the operator type is invalid.
    """
    if operator == "identity":
        return associated_identity_operator(N)
    elif operator == "first-order":
        return associated_first_order_operator(N)
    else:
        raise ValueError(f"Unknown operator type '{operator}'. Use 'identity' or 'first-order'.")

# --- Galerkin Operator Application ---
def galerkin_stencils(N: int, v: np.ndarray, operator: str = "identity") -> np.ndarray:
    """
    Apply a Galerkin operator to a vector using sparse matrix multiplication.

    Parameters:
        N (int): Vector size (must match operator matrix size).
        v (np.ndarray): Input vector (shape: (N,)).
        operator (str): Operator type to apply.

    Returns:
        np.ndarray: Output vector A * v.
    
    Raises:
        ValueError: If vector shape doesn't match N or invalid operator.
    """
    if v.shape[0] != N:
        raise ValueError(f"Input vector length mismatch: expected {N}, got {v.shape[0]}.")

    A = associated_operators(N, operator)
    return A.dot(v)

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

def galerkin_approx(N: int, ell: float, coeff: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute the Galerkin approximation:
        u(x) ≈ sum_{m=1}^{N} coeff[m-1] * phi_m(m, ell, x)

    Parameters:
    - N (int): Number of basis functions to include.
    - ell (float): Problem-specific parameter used by each basis function phi_m.
    - coeff (np.ndarray): Coefficient array of shape (N,), typically from solving a linear system.
    - x (float or np.ndarray): Input location(s) where the approximation is evaluated.

    Returns:
    - float or np.ndarray: The evaluated approximation. Returns a scalar if input x is scalar,
      otherwise a NumPy array of shape matching the input.
    """

    # Convert coeff to a NumPy array of type float, if it's not already.
    coeff = np.asarray(coeff, dtype=float)

    # Validate the shape of the coefficient vector.
    if coeff.shape != (N,):
        raise ValueError(f"Expected coeff shape ({N},), but got {coeff.shape}")

    # Check if input x is a scalar so we can preserve output type.
    is_scalar = np.isscalar(x)

    # Ensure x is treated as a 1D NumPy array for uniform processing.
    x = np.atleast_1d(x)

    # Compute the values of the basis functions:
    # phi_vals will be a (N, len(x)) array, where each row corresponds to phi_m for a specific m.
    phi_vals = np.array([
        phi_m(m, ell, x)  # Assumes phi_m is vectorized with respect to x
        for m in range(1, N + 1)
    ])

    # Compute the dot product across the basis function axis:
    # result is a (len(x),) array holding the sum of coeff[m] * phi_m(x) for all m.
    result = np.dot(coeff, phi_vals)

    # If the original input x was scalar, return a scalar instead of a 1-element array.
    return result[0] if is_scalar else result