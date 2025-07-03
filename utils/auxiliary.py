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
# Import the warnings module to issue runtime alerts without raising exceptions
import warnings
# Contains simulation configuration: domain length `ell`, time vector `t`, and step count `n`
import utils.config as cfg

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
    Solves a banded system of equations of spectral-Galerkin origin using 
    a specialized forward elimination and backward substitution scheme.
    
    The matrix has a structure that allows optimized traversal and reuse 
    of recurrence coefficients.

    Parameters:
        f (np.ndarray): Right-hand side vector of shape (N,).
        N (int): Number of equations (must be >= 2).
        a (float): Scalar coefficient multiplying the identity matrix.
        b (float): Scalar coefficient multiplying the Laplacian-like operator.
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
    Compute the Gauss-Legendre quadrature of function f over [a, b].

    Parameters:
        f       : callable
                  Function to integrate.
        a, b    : float
                  Interval endpoints.
        nodes   : ndarray
                  Gauss-Legendre nodes on [-1, 1].
        weights : ndarray
                  Corresponding weights on [-1, 1].

    Returns:
        float : Approximated integral of f over [a, b].
    """
    mid = 0.5 * (a + b)              # Midpoint of the interval
    half_len = 0.5 * (b - a)         # Half the interval length
    x_mapped = mid + half_len * nodes  # Transform nodes to [a, b]

    try:
        # Attempt fast, vectorized evaluation of f at mapped nodes
        f_vals = np.asarray(f(x_mapped))
        if f_vals.shape != x_mapped.shape:
            raise ValueError("Function output shape mismatch.")
    except Exception:
        # Fallback: evaluate f pointwise if vectorization fails
        f_vals = np.array([f(xi) for xi in x_mapped])

    return half_len * np.dot(weights, f_vals)  # Weighted sum approximation


# --------------------------------------------------------------------------- #
# Method 1: Iterative Gauss-Legendre Quadrature ("glq")                       #
# --------------------------------------------------------------------------- #
def iter_gauss_legendre_quad(f, ell, tol=1e-6, max_n=1000):
    """
    Estimate ∫₀^ℓ f(x) dx by increasing Gauss-Legendre nodes until convergence.

    Parameters:
        f       : callable
                  Function to integrate.
        ell     : float
                  Upper integration limit (must be ≥ 0).
        tol     : float
                  Absolute error tolerance.
        max_n   : int
                  Maximum number of quadrature points.

    Returns:
        integral : float
                   Final integral estimate.
        error    : float
                   Difference between last two estimates.
        n        : int
                   Number of nodes used in final iteration.
    """
    if ell < 0:
        raise ValueError("Upper limit 'ell' must be non-negative.")
    if ell == 0:
        return 0.0, 0.0, 0

    a, b = 0.0, ell
    n = 2                      # Start with minimal node count
    prev_result = None

    while n <= max_n:
        nodes, weights = leggauss(n)
        integral = gauss_legendre_integral(f, a, b, nodes, weights)

        if prev_result is not None:
            error = abs(integral - prev_result)
            if error < tol:
                return integral, error, n  # Converged

        prev_result = integral
        n += 1

    # Reached max_n without satisfying tolerance
    raise ValueError(
        f"Did not converge within max_n = {max_n}. "
        f"Last estimate: {prev_result:.6f}"
    )


# --------------------------------------------------------------------------- #
# Method 2: Halving Gauss-Legendre Quadrature ("hglq")                        #
# --------------------------------------------------------------------------- #
def halving_gauss_legendre_quadrature(f, ell, tol=1e-6, max_depth=20, n_gauss=10):
    """
    Adaptive Gauss-Legendre quadrature using interval halving.

    Parameters:
        f         : callable
                    Function to integrate.
        ell       : float
                    Upper integration limit (must be ≥ 0).
        tol       : float
                    Absolute error tolerance.
        max_depth : int
                    Maximum number of interval refinements.
        n_gauss   : int
                    Gauss-Legendre nodes per subinterval.

    Returns:
        integral : float
                   Final integral estimate.
        error    : float
                   Final error estimate.
        depth    : int
                   Number of halving refinements performed.
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

        # Sum contributions from each subinterval
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
               Function to integrate.
        ell  : float
               Upper integration limit (must be ≥ 0).
        tol  : float
               Absolute error tolerance.

    Returns:
        integral : float
                   Estimated integral value.
        error    : float
                   Estimated absolute error.
        None     : Placeholder for compatibility with other methods.
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
    Computes the integral ∫₀^ell f(x, *args) · φₘ(x) dx,
    where φₘ is the m-th Legendre-based basis function, using a unified
    adaptive quadrature strategy.

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
""" 
Module: Finite Difference Derivative Estimators

Provides fourth-order accurate numerical schemes for estimating the first 
derivative of a function using either:
- numdifftools (adaptive, black-box style)
- manually coded fourth-order finite differences (forward, backward, central)
"""
# --------------------------------------------------------------------------- #

# --- Helper Function for Input Validation and Step Adjustment ---
def _validate_and_prepare_input(x, ell, h_init):
    """
    Validates and preprocesses the input values for derivative estimation.

    Parameters:
        x       : float or array-like
                  Evaluation point(s).
        ell     : float
                  Upper bound of the domain.
        h_init  : float
                  Initial step size.

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
    Estimate the first derivative using numdifftools with a 4th-order scheme.

    Parameters:
        f       : callable
                  Function to differentiate.
        x       : float or array-like
                  Evaluation point(s).
        ell     : float
                  Upper domain bound.
        h_init  : float
                  Initial step size.

    Returns:
        tuple: (Estimated derivative(s), step size used)
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
        except Exception as e:
            warnings.warn(f"Derivative estimation failed at x={xi}: {e}")
            deriv = np.nan  # Fallback for exceptions
        
        derivs.append(deriv)

    result = np.array(derivs)
    return (result[0] if is_scalar else result), h_init

# --- First Derivative Estimation Using Manual 4th-Order Finite Differences ---
def first_order_derivative(f, x, ell, h_init=1e-3):
    """
    Estimate the first derivative using 4th-order finite difference manually.

    Parameters:
        f       : callable
                  Function to differentiate.
        x       : float or array-like
                  Evaluation point(s).
        ell     : float
                  Upper domain bound.
        h_init  : float
                  Initial step size.

    Returns:
        tuple: (Estimated derivative(s), step size used)
    """
    x, is_scalar, h_init = _validate_and_prepare_input(x, ell, h_init)
    derivs = []

    for xi in x:
        try:
            if xi - 2 * h_init < 0:
                # Forward 4th-order finite difference scheme near left boundary
                deriv = (-25 * f(xi) + 48 * f(xi + h_init) - 36 * f(xi + 2 * h_init)
                         + 16 * f(xi + 3 * h_init) - 3 * f(xi + 4 * h_init)) / (12 * h_init)
            elif xi + 2 * h_init > ell:
                # Backward 4th-order finite difference scheme near right boundary
                deriv = (25 * f(xi) - 48 * f(xi - h_init) + 36 * f(xi - 2 * h_init)
                         - 16 * f(xi - 3 * h_init) + 3 * f(xi - 4 * h_init)) / (12 * h_init)
            else:
                # Central 4th-order finite difference scheme in the interior
                deriv = (-f(xi + 2 * h_init) + 8 * f(xi + h_init) - 8 * f(xi - h_init)
                         + f(xi - 2 * h_init)) / (12 * h_init)
        except Exception as e:
            warnings.warn(f"Manual finite difference failed at x={xi}: {e}")
            deriv = np.nan  # In case of domain errors or runtime issues
        
        derivs.append(deriv)

    result = np.array(derivs)
    return (result[0] if is_scalar else result), h_init

# --- Unified Interface for Derivative Estimation (numdifftools or manual method) ---
def first_order_derivative_unified(f, x, ell, derivmeth='nd', h_init=1e-3):
    """
    Unified interface for estimating first-order derivatives.

    Parameters:
        f         : callable
                    Function to differentiate.
        x         : float or array-like
                    Evaluation point(s).
        ell       : float
                    Upper domain bound.
        derivmeth : str
                    Method: 'nd' for numdifftools, 'sfd' for manual scheme.
        h_init    : float
                    Initial step size.

    Returns:
        tuple: (Estimated derivative(s), step size used)
    """
    # Convert the derivative method string to lowercase to allow case-insensitive input
    derivmeth = derivmeth.lower()
    
    if derivmeth == 'nd':
        return first_order_derivative_nd(f, x, ell, h_init=h_init)
    elif derivmeth == 'sfd':
        return first_order_derivative(f, x, ell, h_init=h_init)
    else:
        raise ValueError("Invalid method. Use 'nd' (numdifftools) or 'sfd' (standard finite difference).")

# --------------------------------------------------------------------------- #
"""
    This module evaluates integrals involving the first derivative f'(x) over 
    the interval [0, ell], using either:

    - Squared derivative form:    ∫₀^ell [f'(x)]² dx       (form = 'squared')
    - Legendre projection form:   ∫₀^ell f'(x)·P̃ₘ(x) dx   (form = 'legendre')

    The derivative f'(x) can be:
        - Provided analytically via `df`
        - Estimated numerically via `f` using:
            - 4th-order finite differences ('sfd')
            - External package (e.g., numdifftools) ('nd')

    Assumed utility functions:
        - first_order_derivative_unified: estimates f′(x) numerically
        - normalized_shifted_legendre: evaluates P̃ₘ(x)
        - unified_adaptive_quadrature: performs adaptive numerical integration
"""
# --------------------------------------------------------------------------- #

def integrate_derivative_form(f=None, df=None, ell=None, m=None, form='squared',
                              h=1e-3, derivmeth='nd', **quad_kwargs):
    """
    Computes derivative-based integrals over [0, ell].

    Parameters
    ----------
    f : callable, optional
        Function f(x). Used if df is not provided (numerical differentiation).
    df : callable, optional
        Analytical derivative of f(x). Used directly if provided.
    ell : float
        Upper integration limit (must be > 0).
    m : int, optional
        Degree of normalized shifted Legendre polynomial (required if form='legendre').
    form : {'squared', 'legendre'}
        Type of integral:
            - 'squared'  → ∫₀^ell [f'(x)]² dx
            - 'legendre' → ∫₀^ell f'(x)·P̃ₘ(x) dx
    h : float, optional
        Step size for finite difference derivative (only if f is used).
    derivmeth : {'nd', 'sfd'}, optional
        Method for numerical differentiation.
    **quad_kwargs : dict
        Additional arguments forwarded to `unified_adaptive_quadrature`.

    Returns
    -------
    integral : float
        Numerical result of the integral.
    metric : float or int
        Diagnostic output (e.g., error estimate or node count).
    """

    # ------------------ Input validation ------------------ #

    if ell is None or ell <= 0:
        raise ValueError("The upper limit 'ell' must be a positive number.")

    if (f is None and df is None) or (f is not None and df is not None):
        raise ValueError("Specify exactly one of 'f' or 'df', not both.")

    if form not in ('squared', 'legendre'):
        raise ValueError("The 'form' argument must be either 'squared' or 'legendre'.")

    if form == 'legendre' and m is None:
        raise ValueError("Degree 'm' is required when form='legendre'.")

    # ------------------ Step size adjustment ------------------ #

    if f is not None:
        # Prevent step size from exceeding domain scale
        while h >= ell / 4:
            h /= 2

    # ------------------ Define the integrand ------------------ #

    def integrand(x):
        """
        Inner integrand function to be passed to the adaptive integrator.
        Computes either [f′(x)]² or f′(x)·P̃ₘ(x).
        """
        x = np.atleast_1d(x)           # Ensure input is array-like
        result = np.empty_like(x)      # Allocate result array

        for i, xi in enumerate(x):
            # Compute f′(x) either analytically or numerically
            if df is not None:
                f_prime = df(xi)
            else:
                f_prime, _ = first_order_derivative_unified(
                    f, xi, ell=ell, h_init=h, derivmeth=derivmeth
                )

            # Choose form-specific integrand computation
            if form == 'squared':
                result[i] = f_prime * f_prime
            elif form == 'legendre':
                Pm_val = normalized_shifted_legendre(m, ell, xi)
                result[i] = f_prime * Pm_val

        return result[0] if result.size == 1 else result

    # ------------------ Adaptive numerical integration ------------------ #

    integral, metric, *_ = unified_adaptive_quadrature(integrand, ell, **quad_kwargs)

    return integral, metric

# --------------------------------------------------------------------------- #
""" 
Legendre–Galerkin projections and initialization of modal coefficients 
for partial differential equation solvers.

This routine computes the modal representation of initial conditions 
(u, v) and their spatial derivatives using normalized shifted Legendre 
polynomials as basis functions over the 1D domain [0, ℓ].

Key outputs include:
- L² projections: ⟨uᵢ, φₘ⟩ and ⟨vᵢ, φₘ⟩
- First derivative projections: ⟨u₁′, φₘ⟩ and ⟨v₁′, φₘ⟩, via integration by parts
- Second derivative projections: ⟨uᵢ″, φₘ⟩ and ⟨vᵢ″, φₘ⟩, also via integration by parts,
  by computing ⟨−uᵢ′, φₘ′⟩ and ⟨−vᵢ′, φₘ′⟩

Inputs `du` and `dv` are optional lists of analytical first derivatives 
corresponding to `u` and `v`. When supplied, they are used to improve the 
accuracy of second derivative projections. If not provided, numerical 
differentiation is applied (using either 'nd' or 'sfd' methods).

This routine supports spectral Galerkin discretizations for time-dependent 
partial differential equations with homogeneous Dirichlet boundary conditions.
"""
# --------------------------------------------------------------------------- #

def compute_initial_integrals(u, v, N, ell, *,
                              du=None, dv=None,
                              h=1e-3, derivmeth='nd', **quad_kwargs):
    """
    Computes Legendre–Galerkin modal coefficients of functions u(x), v(x),
    including their L² projections and first/second spatial derivative projections,
    using normalized shifted Legendre polynomials φₘ on the interval [0, ell].

    Parameters
    ----------
    u, v : list of callable
        Initial conditions. Each list should contain functions like:
            u = [u₀(x), u₁(x)], v = [v₀(x), v₁(x)]
    du, dv : list of callable or None
        Optional analytical first derivatives of u and v:
            du = [du₀(x), du₁(x)], dv = [dv₀(x), dv₁(x)]
            If None or entries are None, numerical differentiation is used.
    N : int
        Number of Legendre basis functions φ₁ to φ_N.
    ell : float
        Length of spatial domain (integration interval is [0, ell]).
    h : float, optional
        Step size for numerical differentiation. Defaults to 1e-3.
    derivmeth : {'nd', 'sfd'}, optional
        Method for numerical differentiation: 'nd' = numdifftools, 'sfd' = finite difference.
    **quad_kwargs : dict
        Extra keyword arguments passed to quadrature and projection routines.

    Returns
    -------
    dict
        A dictionary with modal coefficient arrays:
            - 'u_proj'   : list of ∫ u[i]·φₘ dx projections
            - 'v_proj'   : list of ∫ v[i]·φₘ dx projections
            - 'diff1_u1' : array of ∫ u₁′·φₘ dx (first derivative via parts)
            - 'diff1_v1' : array of ∫ v₁′·φₘ dx
            - 'diff2_u'  : array of ∫ u[i]″·φₘ dx (2D array, shape [len(u), N])
            - 'diff2_v'  : array of ∫ v[i]″·φₘ dx (same shape)
    """

    # --- Input validation helpers --- #
    def is_valid_func_list(lst):
        return isinstance(lst, list) and all(callable(f) for f in lst)

    def is_valid_deriv_list(lst):
        return isinstance(lst, list) and all(callable(f) or f is None for f in lst)

    # --- Validate inputs --- #
    if not (is_valid_func_list(u) and is_valid_func_list(v)):
        raise ValueError("Inputs 'u' and 'v' must be lists of callable functions.")

    # If derivatives not given, initialize as None-lists
    du = [None] * len(u) if du is None else du
    dv = [None] * len(v) if dv is None else dv

    if not (is_valid_deriv_list(du) and is_valid_deriv_list(dv)):
        raise ValueError("Inputs 'du' and 'dv' must be lists of callables or None.")

    if not (len(u) == len(v) == len(du) == len(dv)):
        raise ValueError("Lists 'u', 'v', 'du', and 'dv' must all be the same length.")

    if not isinstance(N, int) or N <= 0:
        raise ValueError("Parameter 'N' must be a positive integer.")

    if not isinstance(ell, (int, float)) or ell <= 0:
        raise ValueError("Parameter 'ell' must be a positive float.")

    # --- Allocate modal coefficient arrays --- #
    num_components = len(u)
    u_proj = [np.zeros(N) for _ in range(num_components)]
    v_proj = [np.zeros(N) for _ in range(num_components)]
    diff1_u1 = np.zeros(N)  # ∫ u₁′·φₘ dx
    diff1_v1 = np.zeros(N)  # ∫ v₁′·φₘ dx
    diff2_u = np.zeros((num_components, N))  # ∫ u″·φₘ dx
    diff2_v = np.zeros((num_components, N))  # ∫ v″·φₘ dx

    # --- Loop over all modal indices m = 1 to N --- #
    for m in range(N):
        m_idx = m + 1  # Legendre basis uses 1-based indexing (φ₁, φ₂, ...)

        # --- Compute L² projections: ∫ u[i]·φₘ dx and ∫ v[i]·φₘ dx --- #
        for i in range(num_components):
            u_proj[i][m], _ = integrate_with_phi_m(u[i], ell, m_idx, **quad_kwargs)
            v_proj[i][m], _ = integrate_with_phi_m(v[i], ell, m_idx, **quad_kwargs)

        # --- First derivatives by integration by parts ---
        #     ∫ u₁′·φₘ dx = -∫ u₁·φₘ′ dx (assuming zero boundary values)
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

        # --- Second derivatives: ∫ u″·φₘ dx = -∫ u′·φₘ′ dx --- #
        for i in range(num_components):
            # Prepare u[i]′ as df_u if available, else use f_u and compute numerically
            f_u = None if du[i] else (lambda x, i=i: -u[i](x))
            df_u = (lambda x, i=i: -du[i](x)) if du[i] else None

            diff2_u[i][m], _ = integrate_derivative_form(
                f=f_u,
                df=df_u,
                ell=ell,
                m=m_idx,
                form='legendre',
                h=h,
                derivmeth=derivmeth,
                **quad_kwargs
            )

            f_v = None if dv[i] else (lambda x, i=i: -v[i](x))
            df_v = (lambda x, i=i: -dv[i](x)) if dv[i] else None

            diff2_v[i][m], _ = integrate_derivative_form(
                f=f_v,
                df=df_v,
                ell=ell,
                m=m_idx,
                form='legendre',
                h=h,
                derivmeth=derivmeth,
                **quad_kwargs
            )

    # --- Return modal coefficient dictionary --- #
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

# ----------------------------------------------------------------------
# Function: galerkin_approx
# Purpose : Computes the Galerkin approximation of u(x) using a set of 
#           basis functions φₘ over a spatial domain [0, ell]. The 
#           approximation is formed as a linear combination of φₘ 
#           weighted by provided coefficients for each temporal layer.
# Inputs  : 
#    - ell   : Length of spatial domain (float)
#    - coeff : Coefficient matrix of shape (n, N), where each row 
#              corresponds to a temporal layer
#    - x     : Spatial locations (float or np.ndarray)
# Output  : 
#    - Approximation result at x (scalar or np.ndarray of shape (n, len(x)))
# ----------------------------------------------------------------------

def galerkin_approx(ell: float, coeff: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute the Galerkin approximation:
        u(x) ≈ sum_{m=1}^{N} coeff[m-1] * phi_m(m, ell, x)

    Parameters:
    - ell (float): Length of spatial interval [0, ell] for each basis function phi_m(m, ell, x).
    - coeff (np.ndarray): Coefficient matrix of shape (n, N), typically from solving a Galerkin system.
                          Each row corresponds to a temporal layer.
    - x (float or np.ndarray): Locations at which the approximation is evaluated.

    Returns:
    - np.ndarray or float: Evaluated approximation. Returns a scalar if input x is scalar,
                           otherwise returns an array of shape (n, len(x)).
    """
    
    # Ensure 'ell' is explicitly a float (e.g., if passed as an int)
    ell = float(ell)

    # Convert input arrays to float-type NumPy arrays to avoid unexpected behavior
    coeff = np.asarray(coeff, dtype=float)
    x = np.asarray(x, dtype=float)

    # Determine the number of basis functions (assumes coeff shape is (n, N))
    N = coeff.shape[1]

    # Check if input 'x' is a scalar (to format output accordingly)
    is_scalar = np.isscalar(x)

    # Flatten 'x' to ensure it's always a 1D array for consistent processing
    x = np.atleast_1d(x)

    # Evaluate basis functions φₘ(x) for m = 1 to N
    # Resulting shape: (N, len(x)), each row is phi_m for m in 1..N
    phi_vals = np.array([phi_m(m + 1, ell, x) for m in range(N)])

    # Compute linear combination: matrix multiplication of coeff (n x N) with phi_vals (N x len(x))
    result = coeff @ phi_vals

    # If x was scalar, return a vector (n,) with each row's result at that scalar x
    return result[:, 0] if is_scalar else result

# =============================================================================
# Analytical Solution Evaluation Utilities for Timoshenko Beam Model
# -----------------------------------------------------------------------------
# These two functions allow evaluation and inspection of exact analytical
# solutions (u or v) of the Timoshenko beam model, either as precomputed arrays
# over time-space grids or as callables for on-demand evaluation.
# =============================================================================


# -----------------------------------------------------------------------------
# Function: exact_solution_on_grid
# Description: Evaluate the analytical solution u(x, t) or v(x, t)
#              at a spatial point or over a spatial grid for all/specific times.
# -----------------------------------------------------------------------------
def exact_solution_on_grid(
    func: callable,
    unif_prt_spc: int = None,
    x_val: float = None,
    k: int = None
) -> np.ndarray | float:
    """
    Evaluate the analytical solution of the Timoshenko beam model on a spatial grid or specific point at given time(s).

    Parameters
    ----------
    func : callable
        The analytical solution of the Timoshenko beam model:
        - `func(x, t)` must return either displacement u(x, t) or rotation v(x, t).
        - Must support vectorized `x` and scalar `t`.
    unif_prt_spc : int, optional
        Number of uniform spatial partitions in the interval [0, cfg.ell].
    x_val : float, optional
        A single spatial coordinate in [0, cfg.ell] at which to evaluate the solution.
    k : int, optional
        Specific time step index from the time grid in cfg.t.

    Returns
    -------
    np.ndarray or float
        - If `k` is None: returns values at all time steps for the spatial grid or point.
        - If `k` is specified: returns value(s) only at time index `k`.
    """

    # Require at least one spatial argument
    if x_val is None and unif_prt_spc is None:
        raise ValueError("Specify either `x_val` or `unif_prt_spc`.")

    # Create spatial point or grid
    if x_val is not None:
        if not (0 <= x_val <= cfg.ell):
            raise ValueError(f"x_val = {x_val} is outside domain [0, {cfg.ell}].")
        x = np.array([x_val])  # Single-point input wrapped for vector compatibility
    else:
        x = np.linspace(0, cfg.ell, unif_prt_spc + 1)  # Uniform spatial discretization

    # Evaluate the function across all times at spatial points
    values = np.array([func(x, t_i) for t_i in cfg.t])  # Shape: (len(t), len(x))

    # Return result at specific time index if requested
    if k is not None:
        if not (0 <= k <= cfg.n):
            raise ValueError(f"Time index k = {k} out of range [0, {cfg.n}].")
        return values[k]  # Return values at specific time step

    return values  # Return full time evolution at grid or point

# -----------------------------------------------------------------------------
# Function: callable_exact_solution
# Description: Return callable(s) or evaluated result(s) for u(x, t) or v(x, t)
#              depending on whether x_vals and/or time index k is provided.
# -----------------------------------------------------------------------------
def callable_exact_solution(
    func: callable,
    k: int = None,
    x_vals: float | int | list | np.ndarray = None
):
    """
    Return callable(s) or evaluated values of the analytical solution of the Timoshenko beam model.

    Parameters
    ----------
    func : callable
        The analytical solution of the Timoshenko beam model:
        - Accepts `func(x, t)` and returns u(x, t) or v(x, t).
        - Must support vectorized `x` and scalar `t`.
    k : int, optional
        Specific time index from cfg.t. If None, operates over all time steps.
    x_vals : float | int | list | np.ndarray, optional
        Spatial locations to evaluate the solution, or leave None to return callable(s).

    Returns
    -------
    callable | list[callable] | float | np.ndarray
        - If k is set and x_vals is None: returns a callable in x for fixed t_k.
        - If both k and x_vals are provided: returns evaluation at (x_vals, t_k).
        - If k is None and x_vals is None: returns a list of callables, one per time step.
        - If x_vals is provided but k is None: returns np.ndarray of evaluations over time.
    """

    def validate_and_convert_x_vals(x_input):
        """Convert supported x inputs to float or ndarray."""
        if isinstance(x_input, (float, int)):
            return float(x_input)
        elif isinstance(x_input, list):
            return np.array(x_input, dtype=float)
        elif isinstance(x_input, np.ndarray):
            return x_input.astype(float)
        elif x_input is None:
            return None
        else:
            raise TypeError("x_vals must be float, int, list, or np.ndarray.")

    x_vals = validate_and_convert_x_vals(x_vals)

    def construct_exact_function_at_k(k_idx: int):
        """Construct a function x ↦ func(x, t_k) for fixed time index k."""
        if not (0 <= k_idx <= cfg.n):
            raise ValueError(f"Time index k = {k_idx} must be in [0, {cfg.n}].")
        return lambda x: func(x, cfg.t[k_idx])

    if k is not None:
        fn = construct_exact_function_at_k(k)
        return fn if x_vals is None else fn(x_vals)

    # Return list of callables or evaluate each at x_vals
    all_functions = [construct_exact_function_at_k(k_idx) for k_idx in range(cfg.n + 1)]
    return all_functions if x_vals is None else np.array([fn(x_vals) for fn in all_functions])

# ==============================================================
# Module: compute_L2_norm_galerkin_approx
# --------------------------------------------------------------
# Computes the L2 norm of a Galerkin-approximated solution over
# the spatial domain [0, ell] using numerical integration:
#     L2 = sqrt(∫₀^ell [ũ_k(x)]² dx)
#
# Requirements:
#   - A solver object that provides:
#       • .callable_compute_ansatz(solution_type) → callable(s)
#       • .ell → float (spatial domain length)
#
# Supports:
#   - Single or multiple time step evaluation
#   - Quadrature methods: GLQ, HGLQ, or adaptive integration (SciPy)
# ==============================================================

def compute_L2_norm_galerkin_approx(
    solver,
    solution_type: str,
    k: int = None,
    tol: float = 1e-6,
    method: str = "hglq"
) -> float | list[float]:
    """
    Compute the L2 norm(s) of a Galerkin-approximated solution via numerical integration.

    Parameters
    ----------
    solver : object
        Must provide:
            - callable_compute_ansatz(solution_type: str) → callable or list of callables
            - ell : float (spatial domain length)

    solution_type : str
        Specifies which solution component to use: either 'u' or 'v'.

    k : int, optional
        Time index to evaluate. If None, evaluates all available time steps.

    tol : float, optional
        Tolerance for the numerical integration (default: 1e-6).

    method : str, optional
        Integration method:
            - 'glq'   : Gauss–Legendre quadrature
            - 'hglq'  : Hierarchical GLQ (default)
            - 'scipy' : SciPy's adaptive quadrature

    Returns
    -------
    float or list[float]
        - Single L2 norm (if k is provided)
        - List of L2 norms (if k is None)

    Raises
    ------
    ValueError
        - If the solver lacks required methods/attributes.
        - If solution_type is invalid or callable fetching fails.
        - If k is outside the range of available approximations.
    """

    # ----------------------------------------------------------
    # STEP 1: Retrieve list of callable functions for u_k(x) or v_k(x)
    # ----------------------------------------------------------
    if not hasattr(solver, 'callable_compute_ansatz'):
        raise ValueError("The solver must implement 'callable_compute_ansatz(solution_type: str)'.")

    try:
        approx_solution_generator = solver.callable_compute_ansatz(solution_type=solution_type)
    except Exception as e:
        raise ValueError(f"Failed to retrieve solution callables for '{solution_type}': {e}")

    # Convert a single callable into a list to simplify downstream logic
    if callable(approx_solution_generator):
        approx_solution_generator = [approx_solution_generator]

    # ----------------------------------------------------------
    # STEP 2: Extract spatial domain length ℓ
    # ----------------------------------------------------------
    if not hasattr(solver, 'ell'):
        raise ValueError("The solver must have an attribute 'ell' (spatial domain length).")
    ell = solver.ell

    # ----------------------------------------------------------
    # STEP 3: Define function to compute L2 norm at one time step
    # ----------------------------------------------------------
    def compute_norm_at_k(k_idx: int) -> float:
        """
        Compute L2 norm at a specific time index.

        Parameters
        ----------
        k_idx : int
            Index of the time step in approx_solution_generator.

        Returns
        -------
        float
            L2 norm at the time step.
        """
        approx_fn = approx_solution_generator[k_idx]

        # Define integrand: squared approximation at this time step
        def squared_fn(x: float) -> float:
            return approx_fn(x) ** 2

        # Perform numerical integration ∫₀^ell (ũ_k(x))² dx
        integral, _, _ = unified_adaptive_quadrature(
            squared_fn, ell=ell, tol=tol, method=method
        )

        return np.sqrt(integral)

    # ----------------------------------------------------------
    # STEP 4: Compute L2 norm for a specific time index
    # ----------------------------------------------------------
    if k is not None:
        if not (0 <= k < len(approx_solution_generator)):
            raise ValueError(
                f"Time index k = {k} is out of bounds. Valid range: 0 to {len(approx_solution_generator) - 1}."
            )
        return compute_norm_at_k(k)

    # ----------------------------------------------------------
    # STEP 5: Compute L2 norms across all available time steps
    # ----------------------------------------------------------
    return [compute_norm_at_k(i) for i in range(len(approx_solution_generator))]


# ==============================================================
# Module: compute_L2_norm_from_galerkin_coeffs
# --------------------------------------------------------------
# Computes exact values of the L2 norm of an approximate Galerkin
# solution using a matrix-vector-based formulation:
#     L2 = (ell / 2) * sqrt(cᵀ * H * c)
#
# Assumes:
# - Time is discretized as: t = np.linspace(0, T, n + 1)
# - Galerkin coefficients are stored in shape (n - 1, N)
# - Galerkin approximation uses differences of Legendre polynomials
#   starting from time layer k = 2 (initial layers k = 0 and k = 1
#   are given and not part of the approximation).
# ==============================================================

def compute_L2_norm_from_galerkin_coeffs(
    solver,
    solution_type: str,
    time_layer: int = None
) -> float | list[float]:
    """
    Compute the L2 norm(s) of a Galerkin-approximated solution (either 'u' or 'v'),
    using the exact matrix-vector formula:
        L2 = (ell / 2) * sqrt(cᵀ * H * c)

    Parameters
    ----------
    solver : object
        Object containing:
        - solver.tilde_u and/or solver.tilde_v : np.ndarray of shape (n - 1, N)
        - solver.ell : float, domain length

    solution_type : str
        Selects which solution to compute ('u' or 'v').

    time_layer : int, optional
        Time step index k (must satisfy k ≥ 2). If None, compute norms for all valid time steps.

    Returns
    -------
    float or list[float]
        - Single L2 norm if `time_layer` is provided.
        - List of L2 norms over all valid time steps if `time_layer` is None.

    Raises
    ------
    ValueError
        If `solution_type` is invalid or required solver attributes are missing.

    IndexError
        If `time_layer` is < 2 or beyond the available time steps.

    Notes
    -----
    - The coefficient matrix only includes approximated time layers (k = 2 to k = n).
    - Layers k = 0 and k = 1 are excluded because they are initial conditions.
    """

    # ----------------------------------------------------------
    # Step 1: Select the appropriate coefficient matrix
    # ----------------------------------------------------------
    if solution_type == 'u':
        if not hasattr(solver, 'tilde_u'):
            raise ValueError("solver must have attribute 'tilde_u' for solution_type='u'")
        coeff = solver.tilde_u
    elif solution_type == 'v':
        if not hasattr(solver, 'tilde_v'):
            raise ValueError("solver must have attribute 'tilde_v' for solution_type='v'")
        coeff = solver.tilde_v
    else:
        raise ValueError("solution_type must be either 'u' or 'v'")

    # ----------------------------------------------------------
    # Step 2: Extract domain length
    # ----------------------------------------------------------
    if not hasattr(solver, 'ell'):
        raise ValueError("solver must have attribute 'ell'")
    ell = solver.ell

    # ----------------------------------------------------------
    # Step 3: Get dimensions of the coefficient matrix
    # ----------------------------------------------------------
    num_layers = coeff.shape[0]  # Total number of approximated time steps (rows)
    N = coeff.shape[1]           # Number of Galerkin basis functions (columns)

    # ----------------------------------------------------------
    # Internal function: L2 norm for a single time index
    # ----------------------------------------------------------
    def compute_single(k_index: int) -> float:
        """
        Compute the L2 norm for one time layer (k = k_index + 2).

        Parameters
        ----------
        k_index : int
            Row index in the coefficient matrix

        Returns
        -------
        float
            Exact L2 norm at that time layer
        """
        c_k = coeff[k_index, :]  # Coefficient vector at the current time layer
        H_c = galerkin_stencils(N=N, v=c_k, operator="identity")  # Apply mass matrix
        l2_squared = np.dot(c_k, H_c)  # Compute cᵀ * H * c
        return (ell / 2.0) * np.sqrt(l2_squared)  # Return final scaled L2 norm

    # ----------------------------------------------------------
    # Step 4: Compute norm for a specific time layer (if given)
    # ----------------------------------------------------------
    if time_layer is not None:
        if time_layer < 2:
            raise IndexError(
                f"Invalid time_layer = {time_layer}. Must be ≥ 2 (k = 0 and 1 are initial conditions)."
            )

        k_index = time_layer - 2  # Convert to 0-based index for matrix access

        if k_index >= num_layers:
            raise IndexError(
                f"time_layer = {time_layer} exceeds available layers "
                f"(got shape {coeff.shape}, valid k = 2 to {num_layers + 1})."
            )

        return compute_single(k_index)

    # ----------------------------------------------------------
    # Step 5: Compute norms for all time layers (k = 2 to k = n)
    # ----------------------------------------------------------
    return [compute_single(k_idx) for k_idx in range(num_layers)]



# ============================================
# Function: compute_L2_error
# Short Description: 
#   Computes the L2 error between exact and Galerkin-approximated solutions over a specified spatial domain.
#
# Detailed Description:
#   This function calculates the L2 error, which measures the difference between the exact solution and the 
#   Galerkin-approximated solution across the domain [0, ell]. It supports multiple integration methods (e.g., 
#   Gauss-Legendre quadrature, hierarchical Gauss-Legendre quadrature, or scipy's integrate.quad) for 
#   adaptive numerical integration. The L2 error can be computed either for a specific time step (k) or for all 
#   time steps if no time index is specified.
#
# Parameters:
#   exact_solution_generator : callable or list of callables
#       Exact solution(s) for each time step: either a single function or a list of functions.
#   approx_solution_generator : callable or list of callables
#       Galerkin approximation(s) for each time step: either a single function or a list of functions.
#   ell : float
#       The domain length for integration [0, ell].
#   k : int, optional
#       A specific time step index to compute the error for. If None, computes for all time steps.
#   tol : float, optional
#       The integration tolerance (default is 1e-6).
#   method : str, optional
#       The integration method to use: 'glq', 'hglq', or 'scipy' (default is 'hglq').
#
# Returns:
#   float or list of floats
#       The L2 error at time t_k if k is specified, or the L2 errors for all time steps if k is None.
# ============================================

def compute_L2_error(
    exact_solution_generator,
    approx_solution_generator,
    ell: float,
    k: int = None,
    tol: float = 1e-6,
    method: str = "hglq"
):
    """
    Compute the L2 error between exact and Galerkin-approximated solutions.

    The L2 error quantifies how closely the numerical (Galerkin) approximation matches
    the exact solution by computing the root of the integral of the squared error.

    Parameters
    ----------
    exact_solution_generator : callable or list of callables
        Exact solution(s). Each function maps spatial input x to u(x, t_k).
        Can be a single function or a list of time-indexed functions.
        
    approx_solution_generator : callable or list of callables
        Galerkin approximation(s). Each function maps x to ũ_k(x).
        Can also be a single function or a list.

    ell : float
        Length of the spatial domain [0, ell].

    k : int, optional
        Specific time index. If provided, compute the error only at that time step.

    tol : float, optional
        Tolerance for the numerical integration method.

    method : str, optional
        Integration method to use. Supported options:
            - 'glq': Gauss-Legendre quadrature
            - 'hglq': Hierarchical Gauss-Legendre quadrature
            - 'scipy': Uses scipy.integrate.quad

    Returns
    -------
    float or list of floats
        L2 error at time step k, or a list of L2 errors across all time steps.
    """

    # Ensure inputs are list-like: wrap single callables into single-element lists
    if callable(exact_solution_generator):
        exact_solution_generator = [exact_solution_generator]
    if callable(approx_solution_generator):
        approx_solution_generator = [approx_solution_generator]

    # Ensure both solution lists have the same length
    if len(exact_solution_generator) != len(approx_solution_generator):
        raise ValueError("Mismatch: exact and approx solution lists must have the same length.")

    def compute_error_at_k(k_idx):
        """
        Compute the L2 error for a specific time index.

        Parameters
        ----------
        k_idx : int
            Index at which to compute L2 error.

        Returns
        -------
        float
            L2 norm of the difference between exact and approximate solutions.
        """
        exact_fn = exact_solution_generator[k_idx]
        approx_fn = approx_solution_generator[k_idx]

        # Define pointwise squared difference function
        def squared_diff(x):
            return (exact_fn(x) - approx_fn(x))**2

        # Perform numerical integration of the squared error over [0, ell]
        integral, _, _ = unified_adaptive_quadrature(
            squared_diff, ell=ell, tol=tol, method=method
        )

        # Return the square root of the integral to get the L2 norm
        return np.sqrt(integral)

    # Case: specific time step
    if k is not None:
        if not (0 <= k < len(exact_solution_generator)):
            raise ValueError(f"Time index k = {k} is out of bounds.")
        return compute_error_at_k(k)

    # Case: all time steps — return list of L2 errors
    return [compute_error_at_k(i) for i in range(len(exact_solution_generator))]

# =============================================================================
# Function: plot_L2_errors_over_time
# -----------------------------------------------------------------------------
# Purpose:
# This function generates a high-quality, LaTeX-styled plot of time-dependent
# L2 errors for both displacement (u) and rotation (v) in the Timoshenko beam
# model. It uses a color-blind–friendly palette (Okabe-Ito) to ensure visual
# clarity and accessibility for all users, including those with color vision
# deficiencies.
#
# Inputs:
# - time_array     : 1D array of simulation time points
# - error_u        : L2 error array for displacement u (denoted E_{1,k})
# - error_v        : L2 error array for rotation v (denoted E_{2,k})
# - config         : simulation configuration object with:
#                    - config.n: number of time steps
#                    - config.N: number of Galerkin modes (spatial resolution)
# - output_dir     : path to directory where the plot will be saved (default = 'plots')
#
# Output:
# - Saves a timestamped PDF figure showing both L2 error curves with LaTeX math labels.
# - Returns the absolute file path as a string.
# =============================================================================

def plot_L2_errors_over_time(
    time_array,
    error_u,
    error_v,
    config,
    output_dir: str = "plots"
) -> str:
    """
    Generate and save a high-quality LaTeX-styled plot of L2 errors over time
    for both displacement (u) and rotation (v) in the Timoshenko beam model.

    Parameters
    ----------
    time_array : array-like
        Array of time values (e.g., config.t)

    error_u : array-like
        L2 error values for displacement u at each time step

    error_v : array-like
        L2 error values for rotation v at each time step

    config : object
        Simulation config with required attributes:
            - config.n : number of time steps
            - config.N : number of Galerkin basis functions

    output_dir : str, optional
        Directory where the PDF plot will be saved (default is 'plots')

    Returns
    -------
    str
        Absolute path to the saved PDF file
    """

    # -------------------------
    # Constants and Styling
    # -------------------------
    LINE_WIDTH = 2.00  # Line thickness for plots

    # -------------------------
    # Imports (local to avoid clutter at module level)
    # -------------------------
    from pathlib import Path                  # Filesystem path abstraction
    from datetime import datetime             # For timestamping output file
    import matplotlib.pyplot as plt           # Main plotting interface
    from matplotlib import rcParams           # LaTeX rendering configuration

    # -------------------------
    # Input Validation
    # -------------------------
    if not (len(time_array) and len(error_u) and len(error_v)):
        raise ValueError("Inputs 'time_array', 'error_u', and 'error_v' must be non-empty.")
    if not (len(time_array) == len(error_u) == len(error_v)):
        raise ValueError("Input arrays must be of equal length.")

    # -------------------------
    # Configure LaTeX Rendering for Matplotlib
    # -------------------------
    rcParams["text.usetex"] = True
    rcParams["font.family"] = "lmodern"
    rcParams["text.latex.preamble"] = r"""
    \usepackage[utf8]{inputenc}
    \usepackage[T1]{fontenc}
    \usepackage{lmodern}
    \usepackage{slantsc}
    \usepackage{dsfont}
    \usepackage{upgreek}
    \usepackage{amsmath,amssymb,amsthm,amstext,amsfonts}
    \usepackage{mathtools}
    \usepackage{nicefrac}
    \usepackage{xcolor}
    """

    # -------------------------
    # Colorblind-Friendly Palette (Okabe-Ito)
    # -------------------------
    color_u = "#0072B2"  # Blue: displacement u (E₁ₖ)
    color_v = "#E69F00"  # Orange: rotation v (E₂ₖ)

    # -------------------------
    # Ensure Output Directory Exists
    # -------------------------
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Get Time Range for Labeling
    # -------------------------
    t_min, t_max = float(time_array[0]), float(time_array[-1])

    # -------------------------
    # Initialize Plot
    # -------------------------
    plt.figure(figsize=(8, 4))  # Width × Height in inches

    # Plot L2 error for displacement u
    plt.plot(
        time_array,
        error_u,
        marker='o',
        linestyle='-',
        linewidth=LINE_WIDTH,
        color=color_u,
        label=r"$E_{1,k} = \left\| u \left( \cdot, t_k \right) - \tilde{u}_{k,N} \left( \cdot \right) \right\|$"
    )

    # Plot L2 error for rotation v
    plt.plot(
        time_array,
        error_v,
        marker='s',
        linestyle='--',
        linewidth=LINE_WIDTH,
        color=color_v,
        label=r"$E_{2,k} = \left\| v \left( \cdot, t_k \right) - \tilde{v}_{k,N} \left( \cdot \right) \right\|$"
    )

    # -------------------------
    # Axis Labels, Title, and Layout
    # -------------------------
    plt.xlabel(rf"Time $t \in \left[ {t_min:.0f}, {t_max:.0f} \right]$")
    plt.ylabel(r"$E_k$")
    plt.title(r"$L^2$ Error Evolution for $u(x, t)$ and $v(x, t)$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()  # Prevent clipping of labels

    # -------------------------
    # Save Plot to PDF with Timestamp
    # -------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"L2_errors_n{config.n}_N{config.N}_{timestamp}.pdf"

    plt.savefig(filename)   # Save figure as PDF
    plt.close()             # Free memory and avoid overlap on future plots

    return str(filename)    # Return full file path


# =============================================================================
# Function: plot_exact_vs_approx_solution_at_time_k
# -----------------------------------------------------------------------------
# Purpose:
#   Compare the analytical (exact) and Galerkin (approximate) solution
#   at a fixed time layer t_k for either displacement u or rotation v.
#   Generates a LaTeX-styled, publication-quality plot with colorblind-safe colors.
#
# Inputs:
#   - exact_soln      : callable for exact solution, expects (x, t)
#   - approx_solver   : solver object with callable_compute_ansatz()
#   - solution_type   : 'u' for displacement or 'v' for rotation
#   - time_layer      : integer index k ∈ [0, config.n]
#   - config          : simulation config with attributes: ell, t, N, n
#   - output_dir      : (optional) folder to save plot (default: 'plots')
#
# Returns:
#   - str: full path to the saved PDF plot
# =============================================================================

def plot_exact_vs_approx_solution_at_time_k(
    exact_soln: callable,
    approx_solver: object,
    solution_type: str,
    time_layer: int,
    config,
    output_dir: str = "plots"
) -> str:
    """
    Generate a comparison plot between the exact and Galerkin-approximated solutions
    for a given time layer. The result is a LaTeX-styled, publication-quality plot.

    Parameters:
    - exact_soln      : callable, function returning exact solution values (x, t) -> array
    - approx_solver   : object, must implement `callable_compute_ansatz(solution_type, k, x_vals)`
    - solution_type   : str, 'u' for displacement or 'v' for rotation
    - time_layer      : int, time index k in range [0, config.n]
    - config          : object with attributes: ell (domain length), t (array of times), N (DOFs), n (max index)
    - output_dir      : str, path to save output plots (default: "plots")

    Returns:
    - str: Full path to the saved PDF file
    """

    # -------------------------
    # Constants and Config
    # -------------------------
    LINE_WIDTH = 3.00  # Thickness of plotted lines for visual clarity

    # -------------------------
    # Imports
    # -------------------------
    from pathlib import Path                   # For safe, cross-platform file path handling
    from datetime import datetime              # To create a unique timestamped filename
    import numpy as np                         # For numerical array operations
    import matplotlib.pyplot as plt            # Main plotting interface
    from matplotlib import rcParams            # For configuring LaTeX text rendering

    # -------------------------
    # Input validation
    # -------------------------
    if not (0 <= time_layer <= config.n):
        raise ValueError(f"time_layer must be in range [0, {config.n}]")

    # -------------------------
    # Generate spatial domain and time slice
    # -------------------------
    num_points = 200                           # Resolution of spatial grid
    x_vals = np.linspace(0, config.ell, num_points)  # Spatial grid over [0, ell]
    t_k = config.t[time_layer]                 # Time value at layer k

    # -------------------------
    # Evaluate exact and approximate solutions
    # -------------------------
    exact_values = exact_soln(x_vals, t_k)     # Compute exact solution
    approx_values = approx_solver.callable_compute_ansatz(
        solution_type=solution_type,
        k=time_layer,
        x_vals=x_vals
    )                                          # Compute Galerkin approximation

    # -------------------------
    # Define colors (Okabe-Ito palette: colorblind-safe)
    # -------------------------
    color_exact = "#009E73"   # Green for exact solution
    color_approx = "#D55E00"  # Vermilion for approximate solution

    # -------------------------
    # Enable LaTeX-style text rendering
    # -------------------------
    rcParams["text.usetex"] = True
    rcParams["font.family"] = "lmodern"
    rcParams["text.latex.preamble"] = r"""
    \usepackage[utf8]{inputenc}
    \usepackage[T1]{fontenc}
    \usepackage{lmodern}
    \usepackage{slantsc}
    \usepackage{dsfont}
    \usepackage{upgreek}
    \usepackage{amsmath,amssymb,amsthm,amstext,amsfonts}
    \usepackage{mathtools}
    \usepackage{nicefrac}
    \usepackage{xcolor}
    """

    # -------------------------
    # Create output directory if it doesn't exist
    # -------------------------
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Begin plotting
    # -------------------------
    plt.figure(figsize=(8, 4))  # Set figure size in inches

    # Plot exact solution
    plt.plot(
        x_vals,
        exact_values,
        label=rf"Exact: ${solution_type}(x, t_{{{time_layer}}})$",
        color=color_exact,
        linestyle='-',
        linewidth=LINE_WIDTH
    )

    # Plot approximate solution
    plt.plot(
        x_vals,
        approx_values,
        label=rf"Approximate: $\tilde{{{solution_type}}}_{{k,N}}(x)$",
        color=color_approx,
        linestyle='--',
        linewidth=LINE_WIDTH
    )

    # -------------------------
    # Axis and plot formatting
    # -------------------------
    plt.xlabel(rf"Spatial coordinate $x \in \left[ 0, {config.ell:.0f} \right]$")
    plt.ylabel(r"Solution value")
    plt.title(rf"Exact vs Approximate Solution: ${solution_type}(x, t_{{{time_layer}}})$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()  # Adjust layout to avoid label cutoff

    # -------------------------
    # Export figure to file
    # -------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    filename = output_path / f"solution_{solution_type}_t{time_layer}_N{config.N}_{timestamp}.pdf"
    
    plt.savefig(filename)   # Save as PDF
    plt.close()             # Close figure to free memory

    return str(filename)    # Return full file path