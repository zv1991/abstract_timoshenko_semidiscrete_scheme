import numpy as np  # Provides fast and vectorized numerical operations, including array manipulation and linear algebra
from scipy.special import legendre  # Returns an unshifted Legendre polynomial of specified degree as a polynomial object
from numpy.polynomial.legendre import leggauss  # Generates Gauss–Legendre quadrature nodes and weights for numerical integration
import numdifftools as nd  # Library for numerical differentiation; used here for computing derivatives via finite differences (e.g., nd.Derivative)
from scipy.sparse import identity, diags, csr_matrix  # Utilities for constructing and manipulating sparse matrices, crucial for large-scale linear systems
from numpy.linalg import cond  # Computes the condition number of a matrix using the 2-norm, indicating sensitivity to numerical errors

# --------------------------------------------------------------------------- #
""" Coefficients arising from inner products of Legendre polynomials 
    and their role in the Gauss–Legendre spectral method """
# --------------------------------------------------------------------------- #

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

# --------------------------------------------------------------------------- #
""" Functions related to Legendre polynomials, including their
    differences as employed in spectral basis constructions """
# --------------------------------------------------------------------------- #

def shifted_legendre(m, ell, x):
    """
    Compute the shifted Legendre polynomial P_m(x) in [0, ell].
    """
    x = np.clip(np.asarray(x), 0, ell)  # Ensure x is within valid range
    x_mapped = 2 * x / ell - 1  # Transform x from [0, ell] to [-1, 1]
    return legendre(m)(x_mapped)

# def shifted_legendre(n: int, ell: float, x: np.ndarray) -> np.ndarray:
#     """
#     Evaluate the n-th shifted Legendre polynomial on the interval [0, ell].

#     The shifted Legendre polynomial is defined on the domain [0, ell], 
#     but standard Legendre polynomials are defined on the domain [-1, 1]. 
#     This function performs the necessary transformation to map the input x 
#     from [0, ell] to [-1, 1] and evaluates the polynomial using its coefficients.

#     Args:
#         n (int): Degree of the polynomial (non-negative integer).
#         ell (float): The length of the interval [0, ell].
#         x (np.ndarray): Points in the domain [0, ell] where the polynomial should be evaluated.

#     Returns:
#         np.ndarray: Values of the shifted Legendre polynomial evaluated at the points x.
#     """
    
#     # Step 1: Map the input x from the domain [0, ell] to the domain [-1, 1]
#     x_mapped = 2 * x / ell - 1
    
#     # Step 2: Get the coefficients of the standard Legendre polynomial P_n (highest degree first)
#     coeffs = np.array(legendre(n).coef)  # Extract the coefficients of the Legendre polynomial P_n
    
#     # Step 3: Evaluate the polynomial using jax.numpy.polyval, which performs the evaluation using the coefficients
#     # The `coeffs` array represents the polynomial in the form:
#     #   coeffs[0] * x^n + coeffs[1] * x^(n-1) + ... + coeffs[n-1] * x + coeffs[n]
#     return np.polyval(coeffs, x_mapped)  # Return the evaluated polynomial at the mapped points x

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

# --------------------------------------------------------------------------- #
""" Solver routine for the reduced subsystem derived from the Galerkin 
    formulation of the linearized PDE system """
# --------------------------------------------------------------------------- #

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

# --------------------------------------------------------------------------- #
""" Computation of definite integrals
    using the Gauss–Legendre quadrature rule """
# --------------------------------------------------------------------------- #

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
    
    if ell <= 0:
        raise ValueError("The integration domain upper bound 'ell' must be positive.")
    
    # Define the composite integrand including the phi_m basis function
    def integrand(x):
        return f(x, *args) * phi_m(m, ell, x)  # Assumes phi_m is defined elsewhere and vectorized

    # Call the adaptive integration routine over [0, ell]
    return adaptive_gauss_legendre(integrand, ell, tol=tol, max_n=max_n)

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

# --------------------------------------------------------------------------- #
""" Fourth-order accurate finite difference scheme
    for computing the first spatial derivative """
# --------------------------------------------------------------------------- #

def first_order_derivative_nd(f, x, ell, tol=1e-12, h_init=1e-3, iter_max=50):
    """
    Estimate the first derivative of the function f at point x using a  
    4th-order finite difference method with adaptive step sizing.

    Parameters
    ----------
    f : callable
        The function for which the derivative is computed.
    x : float
        Point at which to evaluate the derivative.
    ell : float
        Upper boundary of the domain [0, ell].
    tol : float, optional
        Tolerance for convergence between successive estimates. Default is 1e-12.
    h_init : float, optional
        Initial finite difference step size. Default is 1e-3.
    iter_max : int, optional
        Maximum number of refinement iterations. Default is 50.

    Returns
    -------
    deriv : float
        The estimated first derivative of f at x.
    h : float
        Final step size used.

    Raises
    ------
    ValueError
        If 'ell' is not provided.
    """

    if ell is None:
        raise ValueError("Parameter 'ell' must be specified.")

    h = h_init              # Initialize step size
    prev_deriv = None       # Store previous derivative for convergence comparison

    for iteration in range(iter_max):
        # Choose finite difference method based on proximity to domain boundaries
        if x - 2 * h < 0:
            method = 'forward'
        elif x + 2 * h > ell:
            method = 'backward'
        else:
            method = 'central'

        try:
            # Create derivative function using 4th-order finite difference
            df = nd.Derivative(f, n=1, step=h, order=4, method=method)
            deriv = df(x)
        except Exception:
            deriv = np.nan  # Handle evaluation failure gracefully

        # If we have a previous estimate and current is valid, check for convergence
        if prev_deriv is not None and not np.isnan(deriv):
            if abs(deriv - prev_deriv) < tol:
                # print(f"Converged in {iteration + 1} iterations.")
                # print(f"Result: {deriv}, Step size: {h}")
                return deriv, h

        # Update previous derivative and refine step size
        prev_deriv = deriv
        h /= 2  # Reduce step size to improve accuracy

    # Fallback if convergence was not achieved within iter_max
    try:
        if x - 2 * h_init < 0:
            method = 'forward'
        elif x + 2 * h_init > ell:
            method = 'backward'
        else:
            method = 'central'

        df = nd.Derivative(f, n=1, step=h_init, order=4, method=method)
        deriv = df(x)
    except Exception:
        deriv = np.nan  # Return NaN if derivative calculation fails

    print(f"Did not converge within {iter_max} iterations.")
    print(f"Last estimate: {prev_deriv}, Step size fallback: {h_init}")
    return deriv, h_init

# --------------------------------------------------------------------------- #
""" Evaluation of a specific integral involving derivative functions """
# --------------------------------------------------------------------------- #


def adaptive_gauss_legendre_integrate_fprime_leg(f, m, ell, tol=1e-6, max_n=1000, h=1e-3):
    """
    Approximates the integral ∫₀^ℓ f'(x) * P̃ₘ(x) dx using:
    - Adaptive Gauss–Legendre quadrature
    - 4th-order finite differences for f'
    - Normalized shifted Legendre polynomial P̃ₘ(x)
    
    Parameters
    ----------
    f : callable
        Function f(x)
    m : int
        Degree of the normalized shifted Legendre polynomial
    ell : float
        Upper integration limit
    tol : float, optional
        Convergence tolerance
    max_n : int, optional
        Maximum number of quadrature points
    h : float, optional
        Initial finite difference step size
    
    Returns
    -------
    integral : float
        Approximated integral value
    n : int
        Number of quadrature points used
    """

    if ell <= 0:
        raise ValueError("Parameter 'ell' must be greater than zero.")

    # Prevent instability near boundaries by reducing h if too large
    while h >= ell / 4:
        h /= 2

    def integrand(x):
        """
        Callable for Gauss-Legendre integrator: computes f'(x) * P̃ₘ(x)
        """
        # Ensure x is an array to vectorize
        x = np.atleast_1d(x)
        result = np.zeros_like(x)
        
        for i, xi in enumerate(x):
            # Compute derivative using 4th-order method
            f_prime, _ = first_order_derivative_nd(f, xi, ell=ell, h_init=h)
            # Compute shifted normalized Legendre polynomial
            Pm_val = normalized_shifted_legendre(m, ell, xi)
            result[i] = f_prime * Pm_val
        
        return result if len(result) > 1 else result[0]

    # Use adaptive quadrature engine
    integral, n = adaptive_gauss_legendre(integrand, ell, tol=tol, max_n=max_n)

    return integral, n

# --------------------------------------------------------------------------- #
""" Evaluation of a specific integral involving
    the square of derivative functions """
# --------------------------------------------------------------------------- #

def adaptive_gauss_legendre_integrate_fprime_sq(f, ell, tol=1e-6, max_n=1000, h=1e-3):
    """
    Approximates the integral ∫₀^ℓ [f'(x)]² dx using:
    - 4th-order finite differences for f'(x)
    - Adaptive Gauss–Legendre quadrature (reused from external implementation)

    Parameters
    ----------
    f : callable
        Function f(x) to differentiate and square.
    ell : float
        Upper limit of integration (must be > 0).
    tol : float, optional
        Convergence tolerance (default: 1e-6).
    max_n : int, optional
        Maximum number of quadrature points (default: 1000).
    h : float, optional
        Initial finite difference step size (default: 1e-3).

    Returns
    -------
    integral : float
        Approximated value of the integral.
    n : int
        Number of quadrature nodes used.
    """

    if ell <= 0:
        raise ValueError("Parameter 'ell' must be greater than zero.")

    # Ensure step size is small enough relative to domain
    while h >= ell / 4:
        h /= 2

    def integrand(x):
        """
        Callable to compute [f'(x)]² at each point x.
        Uses 4th-order finite difference approximation.
        """
        x = np.atleast_1d(x)
        result = np.zeros_like(x)

        for i, xi in enumerate(x):
            f_prime, _ = first_order_derivative_nd(f, xi, ell=ell, h_init=h)
            result[i] = f_prime**2

        return result if len(result) > 1 else result[0]

    # Use the reusable adaptive Gauss–Legendre quadrature engine
    integral, n = adaptive_gauss_legendre(integrand, ell, tol=tol, max_n=max_n)

    return integral, n

# --------------------------------------------------------------------------- #
""" Legendre–Galerkin projections and initialization of modal coefficients 
    for partial differential equation solvers """
# --------------------------------------------------------------------------- #

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

# --------------------------------------------------------------------------- #
""" Computation of the condition number associated with the Galerkin 
    system matrix """
# --------------------------------------------------------------------------- #

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

# --------------------------------------------------------------------------- #
""" Construction of operator matrices (stencils)
    derived via the Legendre–Galerkin spectral method """
# --------------------------------------------------------------------------- #

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