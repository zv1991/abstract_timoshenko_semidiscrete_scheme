import numpy as np  # For efficient numerical operations on arrays
from utils.auxLegendrePolynomials import normalized_shifted_legendre
from utils.auxComputIntegrals import adaptive_gauss_legendre
import numdifftools as nd  # Make sure this is imported if using nd.Derivative

""" Fourth-order accurate finite difference scheme for first derivative """

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