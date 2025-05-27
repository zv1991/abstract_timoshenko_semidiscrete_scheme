import numpy as np  # For efficient numerical operations on arrays
from numpy.polynomial.legendre import leggauss  # Efficient Gauss-Legendre nodes/weights
from utils.auxLegendrePolynomials import phi_m


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