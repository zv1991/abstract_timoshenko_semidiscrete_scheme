import numpy as np  # For efficient numerical operations on arrays
from numpy.polynomial.legendre import leggauss  # Efficient Gauss-Legendre nodes/weights
from scipy.special import legendre  # This returns an unshifted Legendre polynomial of given degree as a polynomial object.
from scipy.sparse import identity, diags, csr_matrix  # For sparse matrix construction
from numpy.linalg import cond  # Computes condition number using 2-norm

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

def shifted_legendre(m, ell, x):
    """
    Compute the shifted Legendre polynomial P_m(x) in [0, ell].
    """
    x = np.clip(np.asarray(x), 0, ell)  # Ensure x is within valid range
    x_mapped = 2 * x / ell - 1  # Transform x from [0, ell] to [-1, 1]
    return legendre(m)(x_mapped)

def normalized_shifted_legendre(m, ell, x):
    """
    Computes the normalized shifted Legendre polynomial:
    
        P_m^*(x) = shifted_legendre(m, ell, x) / (A_m * sqrt(ell))
    
    where A_m is a normalization coefficient.

    Parameters:
        m   : Degree of the polynomial
        ell : Scaling factor
        x   : Input value (or array of values)

    Returns:
        Normalized shifted Legendre polynomial evaluated at x
    """
    # Compute the normalization coefficient A_m
    A_m = coeff_A(m)
    
    # Compute the standard shifted Legendre polynomial
    P_m_x = shifted_legendre(m, ell, x)
    
    # Compute the normalized polynomial
    return P_m_x / (A_m * np.sqrt(ell))

def phi_m(m: int, ell: float, x: np.ndarray) -> np.ndarray:
    """
    Compute the m-th Galerkin basis function φ_m(x) defined as:
        φ_m(x) = (sqrt(ell) / 2) * A_m * [P_{m+1}(x) - P_{m-1}(x)]

    Parameters:
    - m (int): Basis function index (m >= 1).
    - ell (float): Length of the interval [0, ell].
    - x (np.ndarray): Input points where φ_m is evaluated.

    Returns:
    - np.ndarray: Evaluated φ_m(x) at each x.
    """
    
    # Ensure m is a valid index (Galerkin basis functions are defined for m >= 1)
    if m < 1:
        raise ValueError("m must be >= 1.")
    
    # Compute the coefficient A_m
    A_m = coeff_A(m)
    
    # Compute shifted Legendre polynomials P_{m+1} and P_{m-1} at x
    P_plus = shifted_legendre(m + 1, ell, x)
    P_minus = shifted_legendre(m - 1, ell, x)
    
    # Evaluate φ_m(x) using the defined formula
    phi_vals = (np.sqrt(ell) / 2) * A_m * (P_plus - P_minus)
    
    return phi_vals

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

def adaptive_gauss_legendre(f, ell, tol=1e-6, max_n=1000):
    """
    Adaptive Gauss-Legendre Quadrature to approximate the integral of f(x) over [0, ell].

    Parameters:
        f      : Callable, the function f(x) to integrate. Must support NumPy arrays.
        ell    : Upper limit of integration (lower limit is fixed at 0).
        tol    : Desired absolute tolerance for convergence (default: 1e-6).
        max_n  : Maximum number of Gauss-Legendre points to try (default: 1000).

    Returns:
        integral : Approximated value of the integral.
        n        : Number of quadrature points used for the final estimate.
    """
    a, b = 0, ell  # Fixed integration interval

    # Handle edge case: zero-width interval
    if a == b:
        return 0.0, 0

    # Validate tolerance
    if tol <= 0:
        raise ValueError("Tolerance must be positive.")

    n = 2  # Start with 2 quadrature points
    prev_result = None
    scale = 0.5 * (b - a)  # Scaling for interval transformation

    # Iteratively increase the number of points until convergence
    while n <= max_n:
        xi, wi = leggauss(n)  # Nodes and weights for [-1, 1]
        x_mapped = scale * xi + 0.5 * (a + b)  # Map to [a, b]

        # Compute the weighted integral approximation
        integral = scale * np.sum(wi * f(x_mapped))

        # Convergence check: absolute difference with previous estimate
        if prev_result is not None and abs(integral - prev_result) < tol:
            return integral, n

        prev_result = integral
        n += 1  # Increase number of points

    # If max_n is reached without convergence
    raise ValueError(
        f"Integration did not converge within {max_n} points. Last estimate: {prev_result:.6f}"
    )

def integrate_with_phi_m(f, m, ell, *args, tol=1e-6, max_n=1000):
    """
    Computes the integral ∫₀^ell f(x, *args) * phi_m(m, ell, x) dx
    using adaptive Gauss-Legendre quadrature.

    Parameters:
        f      : Callable, function to integrate. Must accept x as first argument (array or scalar), then *args.
        m      : Integer, mode/order used in phi_m.
        ell    : Upper limit of integration. Lower limit is fixed at 0.
        *args  : Additional parameters to pass to function f.
        tol    : Absolute convergence tolerance (default: 1e-6).
        max_n  : Maximum number of Gauss-Legendre points allowed (default: 1000).

    Returns:
        integral_result : Numerical result of the integration.
        n_points_used   : Number of Gauss-Legendre points used.
    """
    # Define the composite integrand including the phi_m basis function
    def integrand(x):
        return f(x, *args) * phi_m(m, ell, x)  # Assumes phi_m is defined elsewhere and vectorized

    # Call the adaptive integration routine over [0, ell]
    return adaptive_gauss_legendre(integrand, ell, tol=tol, max_n=max_n)

""" Fourth-order accurate finite difference scheme for first derivative """
""" You can pass in any step size h (default is 1e-5) """

def fourth_order_derivative(f, x, h=1e-3, ell=None, auto_adjust_h=False):
    """
    Computes f'(x) using a 4th-order accurate finite difference method.
    
    Parameters:
    - f: Callable function to differentiate.
    - x: Point at which to compute the derivative.
    - h: Step size (default is 1e-3).
    - ell: Upper boundary of the domain [0, ell].
    - auto_adjust_h: If True, h is automatically reduced until h < ell / 4.
    
    Returns:
    - (deriv, h): Derivative at x and adjusted h if changed.
    """

    if ell is None:
        raise ValueError("Parameter 'ell' must be specified.")

    if auto_adjust_h:
        # Ensure h is small enough compared to the domain
        while h >= ell / 4:
            h /= 2

    # Apply one-sided or central finite difference based on x's proximity to boundary
    if x - 2 * h < 0:  # Near the left boundary
        deriv = (-25*f(x) + 48*f(x + h) - 36*f(x + 2*h) + 16*f(x + 3*h) - 3*f(x + 4*h)) / (12*h)
    elif x + 2 * h > ell:  # Near the right boundary
        deriv = (25*f(x) - 48*f(x - h) + 36*f(x - 2*h) - 16*f(x - 3*h) + 3*f(x - 4*h)) / (12*h)
    else:  # Central difference for interior points
        deriv = (f(x - 2*h) - 8*f(x - h) + 8*f(x + h) - f(x + 2*h)) / (12*h)

    return deriv, h


def adaptive_gauss_legendre_integrate_fprime_leg(f, m, ell, tol=1e-6, max_n=1000, h=1e-3):
    """
    Approximates ∫₀^ℓ f'(x) * P̃ₘ(x) dx using:
    - Adaptive Gauss–Legendre quadrature
    - 4th-order finite differences for f'
    - External normalized shifted Legendre polynomial P̃ₘ(x)

    Parameters:
    - f: Callable function f(x)
    - m: Degree of the normalized shifted Legendre polynomial
    - ell: Integration upper limit
    - tol: Convergence tolerance
    - max_n: Maximum number of quadrature nodes
    - h: Initial finite difference step size

    Returns:
    - (integral, n): Approximated integral and number of nodes used
    """

    if ell <= 0:
        raise ValueError("Parameter 'ell' must be greater than zero.")

    # Adjust h to prevent boundary instability
    while h >= ell / 4:
        h /= 2

    prev_integral = None

    for n in range(2, max_n + 1):
        # Gauss-Legendre nodes and weights on [-1, 1]
        nodes, weights = leggauss(n)

        # Map nodes to [0, ell]
        x_vals = 0.5 * ell * (nodes + 1)
        w_vals = 0.5 * ell * weights

        integral = 0.0
        for xi, wi in zip(x_vals, w_vals):
            fp, _ = fourth_order_derivative(f, xi, h=h, ell=ell)
            leg_val = normalized_shifted_legendre(m, ell, xi)  # External function
            integral += wi * fp * leg_val

        # Check convergence
        if prev_integral is not None and abs(integral - prev_integral) < tol:
            return integral, n

        prev_integral = integral

    raise RuntimeError("Adaptive integration did not converge within max_n.")

def adaptive_gauss_legendre_integrate_fprime_sq(f, ell, tol=1e-6, max_n=1000, h=1e-3):
    """
    Approximates ∫₀^ℓ [f'(x)]² dx using:
    - Adaptive Gauss–Legendre quadrature
    - 4th-order finite differences for f'

    Parameters:
    - f: Callable function f(x)
    - ell: Upper limit of integration (must be > 0)
    - tol: Convergence tolerance
    - max_n: Maximum quadrature nodes
    - h: Initial finite difference step size

    Returns:
    - (integral, n): Approximated integral and number of nodes used
    """

    if ell <= 0:
        raise ValueError("Parameter 'ell' must be greater than zero.")

    while h >= ell / 4:
        h /= 2

    prev_integral = None

    for n in range(2, max_n + 1):
        nodes, weights = leggauss(n)
        x_vals = 0.5 * ell * (nodes + 1)
        w_vals = 0.5 * ell * weights

        integral = 0.0
        for xi, wi in zip(x_vals, w_vals):
            fp, _ = fourth_order_derivative(f, xi, h=h, ell=ell)
            integral += wi * fp**2

        if prev_integral is not None and abs(integral - prev_integral) < tol:
            return integral, n

        prev_integral = integral

    raise RuntimeError("Adaptive integration did not converge within max_n.")

""" Legendre-Galerkin Integral Projections and Initial Condition Processing for PDE Solvers """

def compute_time_dependent_integrals(f, n, N, ell, t):
    """
    Compute integrals of the form ∫ f(x, t_k+1) * φ_m(x) dx over all time steps and modes.
    
    Parameters:
        f   : callable, function of (x, t)
        n   : int, number of time steps
        N   : int, number of basis functions
        ell : float, length of the domain or transformation scale
        t   : array-like, time discretization points

    Returns:
        integrals : ndarray of shape (n-1, N), the integral values for each (k, m)
    """
    integrals = np.zeros((n - 1, N))  # Preallocate for performance
    for k in range(n - 1):
        for m in range(N):
            # Compute the integral for each φ_m at time t[k+1]
            integrals[k, m], _ = integrate_with_phi_m(f, m + 1, ell, t[k + 1])
    return integrals

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