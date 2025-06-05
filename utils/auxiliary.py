# Provides fast and vectorized numerical operations, including array manipulation and linear algebra
import numpy as np
# Returns an unshifted Legendre polynomial of specified degree as a polynomial object
from scipy.special import legendre
# Generates Gauss–Legendre quadrature nodes and weights for numerical integration
from numpy.polynomial.legendre import leggauss
# Library for numerical differentiation; used here for computing derivatives via finite differences (e.g., nd.Derivative)
import numdifftools as nd
# Utilities for constructing and manipulating sparse matrices, crucial for large-scale linear systems
from scipy.sparse import identity, diags, csr_matrix
# Computes the condition number of a matrix using the 2-norm, indicating sensitivity to numerical errors
from numpy.linalg import cond
# Import quad from scipy.integrate and alias it as scipy_quad to avoid naming conflicts
from scipy.integrate import quad as scipy_quad

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

# =========================================================================== #
#                            Quadrature Method Suite                          #
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Utility: Gauss-Legendre Integration over Arbitrary Interval [a, b]          #
# --------------------------------------------------------------------------- #
def gauss_legendre_integral(f, a, b, nodes, weights):
    """
    Compute the Gauss-Legendre quadrature of f over [a, b].

    Parameters:
        f       : callable
                  Function to integrate.
        a, b    : float
                  Integration interval endpoints.
        nodes   : ndarray
                  Gauss-Legendre nodes on [-1, 1].
        weights : ndarray
                  Corresponding weights on [-1, 1].

    Returns:
        float : Approximated integral of f over [a, b].
    """
    mid = 0.5 * (a + b)                         # Midpoint of interval
    half_len = 0.5 * (b - a)                    # Half length for scaling
    x_mapped = mid + half_len * nodes           # Map nodes to [a, b]

    try:
        # Attempt fast vectorized evaluation
        f_vals = np.asarray(f(x_mapped))
        if f_vals.shape != x_mapped.shape:
            raise ValueError("Function output shape mismatch.")
    except Exception:
        # Fallback to non-vectorized evaluation
        f_vals = np.array([f(xi) for xi in x_mapped])

    return half_len * np.dot(weights, f_vals)   # Compute weighted sum

# --------------------------------------------------------------------------- #
# Method 1: Iterative Gauss-Legendre Quadrature ("glq")                       #
# --------------------------------------------------------------------------- #
def iter_gauss_legendre_quad(f, ell, tol=1e-6, max_n=1000):
    """
    Estimate ∫₀^ℓ f(x) dx using increasing Gauss-Legendre points until convergence.

    Parameters:
        f       : callable
        ell     : float, upper integration limit
        tol     : float, absolute error tolerance
        max_n   : int, maximum number of quadrature points

    Returns:
        integral : float
        error    : float, difference between last two estimates
        n        : int, number of points used
    """
    if ell < 0:
        raise ValueError("Upper limit 'ell' must be non-negative.")
    if ell == 0:
        return 0.0, 0.0, 0

    a, b = 0.0, ell
    n = 2
    prev_result = None

    while n <= max_n:
        nodes, weights = leggauss(n)
        integral = gauss_legendre_integral(f, a, b, nodes, weights)

        if prev_result is not None:
            error = abs(integral - prev_result)
            if error < tol:
                return integral, error, n

        prev_result = integral
        n += 1

    raise ValueError(
        f"Did not converge within max_n = {max_n}. Last estimate: {prev_result:.6f}"
    )

# --------------------------------------------------------------------------- #
# Method 2: Halving Gauss-Legendre Quadrature ("hglq")                        #
# --------------------------------------------------------------------------- #
def halving_gauss_legendre_quadrature(f, ell, tol=1e-6, max_depth=20, n_gauss=10):
    """
    Adaptive Gauss-Legendre integration using dyadic interval halving.

    Parameters:
        f         : callable
        ell       : float, upper integration limit
        tol       : float, absolute error tolerance
        max_depth : int, maximum number of refinement levels
        n_gauss   : int, number of quadrature points per subinterval

    Returns:
        integral : float
        error    : float
        depth    : int, number of refinements used
    """
    if ell < 0:
        raise ValueError("Parameter 'ell' must be non-negative.")
    if ell == 0:
        return 0.0, 0.0, 0

    nodes, weights = leggauss(n_gauss)
    prev_integral = gauss_legendre_integral(f, 0.0, ell, nodes, weights)

    for k in range(1, max_depth + 1):
        n_intervals = 2 ** k
        dx = ell / n_intervals
        current_integral = 0.0

        # Apply quadrature to each subinterval
        for i in range(n_intervals):
            a, b = i * dx, (i + 1) * dx
            current_integral += gauss_legendre_integral(f, a, b, nodes, weights)

        error = abs(current_integral - prev_integral)
        if error < tol:
            return current_integral, error, k

        prev_integral = current_integral

    raise RuntimeError(
        f"Failed to converge within max_depth = {max_depth}. "
        f"Last error: {error:.3e}"
    )

# --------------------------------------------------------------------------- #
# Method 3: SciPy Built-in Quadrature ("scipy")                               #
# --------------------------------------------------------------------------- #
def scipy_quad_wrapper(f, ell, tol=1e-6):
    """
    Estimate ∫₀^ℓ f(x) dx using SciPy's adaptive quadrature.

    Parameters:
        f    : callable
        ell  : float
        tol  : float, absolute error tolerance

    Returns:
        integral : float
        error    : float
        None     : placeholder for compatibility
    """
    if ell < 0:
        raise ValueError("Upper limit 'ell' must be non-negative.")
    if ell == 0:
        return 0.0, 0.0, None

    result, error = scipy_quad(f, 0.0, ell, epsabs=tol)
    return result, error, None

# --------------------------------------------------------------------------- #
# Dispatcher: Unified Adaptive Quadrature Interface                           #
# --------------------------------------------------------------------------- #
def unified_adaptive_quadrature(
    f, ell, tol=1e-6, method="hglq", max_n=1000, max_depth=20, n_points=10
):
    """
    Unified interface for multiple quadrature (numerical integration) schemes.

    Parameters:
        f         : callable
                    The function to integrate.
        ell       : float
                    The upper limit of the integration interval. Lower limit is assumed to be 0.
        tol       : float, optional (default=1e-6)
                    Absolute error tolerance for the integration.
        method    : str, optional (default='hglq')
                    Integration method to use: 'glq' (iterative Gauss-Legendre),
                                               'hglq' (halving Gauss-Legendre),
                                               'scipy' (SciPy’s quad function).
        max_n     : int, optional (default=1000)
                    Maximum number of points for the 'glq' method.
        max_depth : int, optional (default=20)
                    Maximum recursion depth for the 'hglq' method.
        n_points  : int, optional (default=10)
                    Number of Gauss points per subinterval for the 'hglq' method.

    Returns:
        tuple:
            - integral : float
                         Estimated value of the integral.
            - metric   : float or int
                         Method-dependent metric:
                             - 'glq' returns number of points used.
                             - 'hglq' returns estimated error.
                             - 'scipy' returns estimated error.
    """
    # Dispatch integration based on selected method
    if method == "glq":
        # Use iterative Gauss-Legendre quadrature
        return iter_gauss_legendre_quad(f, ell, tol, max_n)
    
    elif method == "hglq":
        # Use adaptive halving with Gauss-Legendre subinterval quadrature
        return halving_gauss_legendre_quadrature(
            f, ell, tol=tol, max_depth=max_depth, n_gauss=n_points
        )
    
    elif method == "scipy":
        # Use SciPy's built-in quad function
        return scipy_quad_wrapper(f, ell, tol)
    
    else:
        # Raise an error for unsupported methods
        raise ValueError(
            f"Invalid method '{method}'. Must be 'glq', 'hglq', or 'scipy'."
        )

def integrate_with_phi_m(f, ell, m, *args, **quad_kwargs):
    """
    Computes the integral ∫₀^ell f(x, *args) * φₘ(x) dx,
    where φₘ is the m-th basis function (e.g., sine/cosine, orthogonal polynomial),
    using a unified adaptive quadrature method.

    Parameters:
        f            : callable
                       Function to integrate. Must accept 'x' as the first argument, followed by *args.
        ell          : float
                       Upper limit of integration. Must be positive. The lower limit is assumed to be 0.
        m            : int
                       Index/order of the basis function φₘ.
        *args        : tuple
                       Additional arguments to pass to 'f' (e.g., parameters like time `t[k+1]`).
        **quad_kwargs: dict
                       Optional keyword arguments passed to `unified_adaptive_quadrature`, such as:
                           - tol: float         (absolute error tolerance)
                           - method: str        ("glq", "hglq", or "scipy")
                           - max_n: int         (for "glq")
                           - max_depth: int     (for "hglq")
                           - n_points: int      (for "hglq")

    Returns:
        tuple:
            - integral_value : float
                               Estimated value of the integral.
            - convergence_info: float or int
                               Method-dependent diagnostic information
                               (e.g., error estimate or point count).
    """

    # --- Safety check on integration domain ---
    if ell <= 0:
        raise ValueError("The integration upper bound 'ell' must be strictly positive.")

    # --- Construct the integrand ---
    # Multiply user-defined function f(x, *args) with basis function φₘ(x)
    # Assumes phi_m is defined globally and vectorized to work on arrays.
    def integrand(x):
        return f(x, *args) * phi_m(m, ell, x)

    # --- Compute the integral using the selected quadrature method ---
    # Pass integration control settings via **quad_kwargs
    result = unified_adaptive_quadrature(
        integrand,  # The integrand function: f(x) * φₘ(x)
        ell,        # Integration upper bound
        **quad_kwargs
    )

    # --- Return only the value and diagnostic info ---
    # Typically returns (value, error)
    return result[:2]

def compute_time_dependent_integrals(f, N, ell, t, **quad_kwargs):
    """
    Computes integrals of the form:
        ∫₀^ell f(x, t_{k+1}) * φₘ(x) dx
    for each time step `k` and basis function index `m`.

    Parameters:
        f             : callable
                        A function of (x, t), representing a time-dependent spatial function.
        N             : int
                        Number of basis functions φₘ(x) used in the decomposition.
        ell           : float
                        Upper limit of the spatial integration domain.
        t             : array-like
                        1D array of time discretization points of length `n`. Must have at least two elements.
        **quad_kwargs : dict
                        Optional keyword arguments forwarded to `integrate_with_phi_m`, e.g.:
                            - tol       : float (absolute tolerance)
                            - method    : str {"glq", "hglq", "scipy"}
                            - max_n     : int  (for "glq")
                            - max_depth : int  (for "hglq")
                            - n_points  : int  (for "hglq")

    Returns:
        integrals : np.ndarray, shape (n-1, N)
                    Array where:
                        integrals[k, m] ≈ ∫₀^ell f(x, t[k+1]) * φₘ₊₁(x) dx
    """

    # --- Validate time vector ---
    n = len(t)
    if n < 2:
        raise ValueError("Time array 't' must contain at least two time points.")

    # --- Preallocate result array ---
    # Rows: time intervals (from k = 0 to n - 2)
    # Cols: basis functions m = 1 to N
    integrals = np.zeros((n - 1, N))

    # --- Main computation loop ---
    for k in range(n - 1):
        t_next = t[k + 1]  # Advance to the next time step

        for m in range(N):
            # Compute integral ∫ f(x, t_next) * φₘ₊₁(x) dx
            # Note: basis function index passed to phi_m is m+1
            value, _ = integrate_with_phi_m(f, ell, m + 1, t_next, **quad_kwargs)

            # Store only the computed integral (ignore diagnostics)
            integrals[k, m] = value

    return integrals

# --------------------------------------------------------------------------- #
""" Fourth-order accurate finite difference scheme
    for computing the first spatial derivative """
# --------------------------------------------------------------------------- #

# --- Helper Function for Input Validation and Step Adjustment ---
def _validate_and_prepare_input(x, ell, h_init):
    """
    Validates and preprocesses the input values for derivative functions.
    
    Parameters:
        x (float or array-like): Input point(s).
        ell (float): Upper bound of the domain.
        h_init (float): Initial step size.

    Returns:
        tuple: (Processed x as np.ndarray, is_scalar flag, adjusted h_init)
    """
    if ell is None:
        raise ValueError("Parameter 'ell' must be specified.")

    # Ensure x is at least 1D and determine if the original input was scalar
    is_scalar = np.isscalar(x)
    x = np.atleast_1d(x)

    # Reduce h_init until it is small enough relative to the domain
    while h_init > ell / 4:
        h_init /= 2

    return x, is_scalar, h_init

# --- First Derivative Estimation Using numdifftools (4th-order accuracy) ---
def first_order_derivative_nd(f, x, ell, h_init=1e-3):
    """
    Estimate the first derivative using numdifftools with adaptive 4th-order method.

    Parameters:
        f (callable): Function to differentiate.
        x (float or array-like): Evaluation point(s).
        ell (float): Upper bound of the domain.
        h_init (float): Initial step size.

    Returns:
        tuple: (Estimated derivative(s), final step size used)
    """
    x, is_scalar, h_init = _validate_and_prepare_input(x, ell, h_init)
    derivs = []

    for xi in x:
        # Determine direction of finite difference based on boundary proximity
        if xi - 2 * h_init < 0:
            method = 'forward'
        elif xi + 2 * h_init > ell:
            method = 'backward'
        else:
            method = 'central'

        try:
            df = nd.Derivative(f, n=1, step=h_init, order=4, method=method)
            deriv = df(xi)
        except Exception:
            deriv = np.nan  # Fallback for exceptions
        derivs.append(deriv)

    result = np.array(derivs)
    return (result[0] if is_scalar else result), h_init

# --- First Derivative Estimation Using Manual 4th-Order Finite Differences ---
def first_order_derivative(f, x, ell, h_init=1e-3):
    """
    Estimate the first derivative using manually implemented 4th-order finite differences.

    Parameters:
        f (callable): Function to differentiate.
        x (float or array-like): Evaluation point(s).
        ell (float): Upper bound of the domain.
        h_init (float): Initial step size.

    Returns:
        tuple: (Estimated derivative(s), final step size used)
    """
    x, is_scalar, h_init = _validate_and_prepare_input(x, ell, h_init)
    derivs = []

    for xi in x:
        try:
            if xi - 2 * h_init < 0:
                # Forward 4th-order finite difference
                deriv = (-25 * f(xi) + 48 * f(xi + h_init) - 36 * f(xi + 2 * h_init)
                         + 16 * f(xi + 3 * h_init) - 3 * f(xi + 4 * h_init)) / (12 * h_init)
            elif xi + 2 * h_init > ell:
                # Backward 4th-order finite difference
                deriv = (25 * f(xi) - 48 * f(xi - h_init) + 36 * f(xi - 2 * h_init)
                         - 16 * f(xi - 3 * h_init) + 3 * f(xi - 4 * h_init)) / (12 * h_init)
            else:
                # Central 4th-order finite difference
                deriv = (-f(xi + 2 * h_init) + 8 * f(xi + h_init) - 8 * f(xi - h_init)
                         + f(xi - 2 * h_init)) / (12 * h_init)
        except Exception:
            deriv = np.nan  # In case of domain errors or runtime issues
        derivs.append(deriv)

    result = np.array(derivs)
    return (result[0] if is_scalar else result), h_init

# --- Unified Interface for Derivative Estimation (numdifftools or manual method) ---
def first_order_derivative_unified(f, x, ell, derivmeth='nd', h_init=1e-3):
    """
    Unified API to estimate first-order derivatives using either numdifftools
    or manual finite difference methods.

    Parameters:
        f (callable): Function to differentiate.
        x (float or array-like): Evaluation point(s).
        ell (float): Upper bound of the domain.
        derivmeth (str): Derivative method, either 'nd' (numdifftools) or 'sfd' (manual).
        h_init (float): Initial step size.

    Returns:
        tuple: (Estimated derivative(s), final step size used)
    """
    if derivmeth == 'nd':
        return first_order_derivative_nd(f, x, ell, h_init=h_init)
    elif derivmeth == 'sfd':
        return first_order_derivative(f, x, ell, h_init=h_init)
    else:
        raise ValueError("Invalid method. Use 'nd' (numdifftools) or 'sfd' (standard finite difference).")

# --------------------------------------------------------------------------- #
""" Gauss-Legendre Integration of f'(x)·P̃ₘ(x) Using 4th-Order Derivatives """
# --------------------------------------------------------------------------- #

def gauss_legendre_integrate_fprime_leg(f, m, ell, h=1e-3, derivmeth='nd', **quad_kwargs):
    """
    Approximates the integral:
        ∫₀^ell f'(x) * P̃ₘ(x) dx

    using:
        - An adaptive quadrature routine
        - A 4th-order finite difference scheme for estimating f'
        - A normalized shifted Legendre polynomial basis P̃ₘ(x)

    Parameters
    ----------
    f : callable
        The function f(x) whose derivative will be integrated.
    m : int
        Degree of the normalized shifted Legendre polynomial.
    ell : float
        Upper limit of integration. Must be strictly positive.
    h : float, optional
        Initial step size for finite difference derivative. Default is 1e-3.
    derivmeth : str, optional
        Derivative estimation method: 'nd' (numdifftools) or 'sfd' (manual FD).
    **quad_kwargs : dict, optional
        Additional arguments passed to `unified_adaptive_quadrature`:
            - tol: float (tolerance)
            - method: {"glq", "hglq", "scipy"}
            - max_n, max_depth, n_points, etc.

    Returns
    -------
    integral : float
        The numerical approximation of the integral.
    metric : float or int
        Additional diagnostic (e.g., error estimate or sample count).
    """

    # --- Safety check for domain ---
    if ell <= 0:
        raise ValueError("The upper integration limit 'ell' must be greater than zero.")

    # --- Ensure h is small enough relative to ell ---
    while h >= ell / 4:
        h /= 2

    # --- Define the integrand function: f'(x) * P̃ₘ(x) ---
    def integrand(x):
        x = np.atleast_1d(x)                 # Ensure x is array-like for loop handling
        result = np.zeros_like(x)           # Preallocate output array

        for i, xi in enumerate(x):
            # Estimate the derivative f'(xi) using the chosen method
            f_prime, _ = first_order_derivative_unified(
                f, xi, ell=ell, derivmeth=derivmeth, h_init=h
            )

            # Evaluate normalized shifted Legendre polynomial P̃ₘ(xi)
            Pm_val = normalized_shifted_legendre(m, ell, xi)

            # Multiply the derivative with the polynomial
            result[i] = f_prime * Pm_val

        # Return scalar if input was scalar
        return result[0] if result.size == 1 else result

    # --- Apply adaptive quadrature to the integrand over [0, ell] ---
    integral, metric, *_ = unified_adaptive_quadrature(
        integrand, ell, **quad_kwargs
    )

    return integral, metric

# --------------------------------------------------------------------------- #
""" Evaluation of a specific integral involving
    the square of derivative functions """
# --------------------------------------------------------------------------- #

def gauss_legendre_integrate_fprime_sq(f, ell, h=1e-3, derivmeth='nd', **quad_kwargs):
    """
    Approximates the integral ∫₀^ℓ [f'(x)]² dx using numerical differentiation and adaptive quadrature.

    Parameters
    ----------
    f : callable
        The function f(x) whose squared derivative is to be integrated.
    ell : float
        The upper limit of integration. Must be positive.
    h : float, optional
        Initial step size for derivative approximation. Default is 1e-3.
    derivmeth : str, optional
        Method for derivative approximation. Options include:
            - 'nd' : numdifftools (if available)
            - 'sfd': simple finite difference (manual)
            - others as supported by `first_order_derivative_unified`.
    **quad_kwargs : dict
        Additional keyword arguments passed to `unified_adaptive_quadrature`, e.g.:
            - tol : float
            - method : {"glq", "hglq", "scipy"}
            - max_n : int
            - max_depth : int
            - n_points : int

    Returns
    -------
    integral : float
        Approximated value of the integral ∫₀^ℓ [f'(x)]² dx.
    metric : float or int
        A metric from the quadrature routine (e.g., estimated error, number of nodes).
    """
    
    # --- Validate the domain ---
    if not callable(f):
        raise TypeError("Parameter 'f' must be a callable function.")
    if not (isinstance(ell, (int, float)) and ell > 0):
        raise ValueError("Parameter 'ell' must be a positive float.")
    
    # --- Sanitize/adjust h to avoid too coarse a grid ---
    while h >= ell / 4:
        h /= 2

    # --- Define the integrand: [f'(x)]² ---
    def integrand(x):
        x = np.atleast_1d(x)                    # Ensure x is iterable (1D array)
        result = np.empty_like(x)               # Allocate output array of same shape

        for i, xi in enumerate(x):
            # Compute derivative at xi using selected method
            f_prime, _ = first_order_derivative_unified(
                f, xi, ell=ell, h_init=h, derivmeth=derivmeth
            )
            result[i] = f_prime ** 2            # Square of the derivative value

        return result[0] if result.size == 1 else result  # Return scalar or array

    # --- Compute integral using an adaptive quadrature method ---
    integral, metric, *_ = unified_adaptive_quadrature(
        integrand, ell, **quad_kwargs
    )

    return integral, metric

# --------------------------------------------------------------------------- #
""" Legendre–Galerkin projections and initialization of modal coefficients 
    for partial differential equation solvers """
# --------------------------------------------------------------------------- #

def compute_initial_integrals(u, v, N, ell, *, h=1e-3, derivmeth='nd', **quad_kwargs):
    """
    Projects the initial conditions onto normalized Legendre basis functions and computes 
    their first and second derivatives via integration by parts, assuming homogeneous 
    Dirichlet boundary conditions (functions vanish at x=0 and x=ell).

    Parameters
    ----------
    u, v : list of callable
        Initial condition function lists: [u0, u1] and [v0, v1].
    N : int
        Number of Legendre basis functions (φ₁ through φ_N).
    ell : float
        Length of the spatial domain [0, ell].
    h : float, optional
        Step size for numerical differentiation.
    derivmeth : str, optional
        Method for approximating derivatives (e.g., 'nd', 'sfd').
    **quad_kwargs : dict
        Additional keyword arguments passed to all quadrature routines:
            - unified_adaptive_quadrature
            - integrate_with_phi_m
            - gauss_legendre_integrate_fprime_leg

    Returns
    -------
    dict
        Dictionary containing:
            'u_proj':   List [∫ u₀·φₘ dx, ∫ u₁·φₘ dx]
            'v_proj':   List [∫ v₀·φₘ dx, ∫ v₁·φₘ dx]
            'diff1_u1': Array ∫ u₁′·φₘ dx
            'diff1_v1': Array ∫ v₁′·φₘ dx
            'diff2_u':  Array [∫ u₀″·φₘ dx, ∫ u₁″·φₘ dx]
            'diff2_v':  Array [∫ v₀″·φₘ dx, ∫ v₁″·φₘ dx]
    """

    # --- Validate input types and shapes ---
    if not (isinstance(u, list) and isinstance(v, list) and 
            len(u) == 2 and len(v) == 2 and 
            callable(u[0]) and callable(u[1]) and 
            callable(v[0]) and callable(v[1])):
        raise ValueError("u and v must be lists of two callable functions each.")

    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")

    if not isinstance(ell, (int, float)) or ell <= 0:
        raise ValueError("ell must be a positive real number.")

    # --- Allocate memory for all projections ---
    u_proj = [np.zeros(N), np.zeros(N)]  # ∫ u₀·φₘ, ∫ u₁·φₘ
    v_proj = [np.zeros(N), np.zeros(N)]  # ∫ v₀·φₘ, ∫ v₁·φₘ
    diff1_u1 = np.zeros(N)               # ∫ u₁′·φₘ
    diff1_v1 = np.zeros(N)               # ∫ v₁′·φₘ
    diff2_u = np.zeros((2, N))           # ∫ u₀″·φₘ, ∫ u₁″·φₘ
    diff2_v = np.zeros((2, N))           # ∫ v₀″·φₘ, ∫ v₁″·φₘ

    # --- Loop over each Legendre basis function φₘ ---
    for m in range(N):
        m_idx = m + 1  # φₘ corresponds to index m+1 in 1-based indexing

        # --- Compute L² projection: ∫ u[i]·φₘ dx and ∫ v[i]·φₘ dx ---
        for i in range(2):  # Loop over u₀/u₁ and v₀/v₁
            u_proj[i][m], _ = integrate_with_phi_m(u[i], ell, m_idx, **quad_kwargs)
            v_proj[i][m], _ = integrate_with_phi_m(v[i], ell, m_idx, **quad_kwargs)

        # --- First derivative projections: ∫ u₁′·φₘ dx = -∫ u₁·φₘ′ dx ---
        #     Integration by parts: assume u₁ and φₘ vanish at endpoints ⇒ no boundary term
        diff1_u1[m] = unified_adaptive_quadrature(
            lambda x: -u[1](x) * normalized_shifted_legendre(m_idx, ell, x),
            ell,
            **quad_kwargs
        )[0]

        diff1_v1[m] = unified_adaptive_quadrature(
            lambda x: -v[1](x) * normalized_shifted_legendre(m_idx, ell, x),
            ell,
            **quad_kwargs
        )[0]

        # --- Second derivative projections: ∫ u″·φₘ dx = -∫ u′·φₘ′ dx ---
        #     Obtained by applying integration by parts once, assuming φₘ vanishes at the boundaries
        for i in range(2):
            diff2_u[i][m], _ = gauss_legendre_integrate_fprime_leg(
                lambda x, i=i: -u[i](x),
                m_idx,
                ell,
                h=h,
                derivmeth=derivmeth,
                **quad_kwargs
            )

            diff2_v[i][m], _ = gauss_legendre_integrate_fprime_leg(
                lambda x, i=i: -v[i](x),
                m_idx,
                ell,
                h=h,
                derivmeth=derivmeth,
                **quad_kwargs
            )

    # --- Return results in organized dictionary ---
    return {
        'u_proj': u_proj,
        'v_proj': v_proj,
        'diff1_u1': diff1_u1,
        'diff1_v1': diff1_v1,
        'diff2_u': diff2_u,
        'diff2_v': diff2_v
    }

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