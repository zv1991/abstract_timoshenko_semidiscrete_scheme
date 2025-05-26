import numpy as np  # For efficient numerical operations on arrays
from numpy.polynomial.legendre import leggauss  # Efficient Gauss-Legendre nodes/weights
from auxLegendrePolynomials import phi_m


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

def integrate_with_phi_m(
    f: callable,
    m: int,
    ell: float,
    *args,
    tol: float = 1e-6,
    max_n: int = 1000
):
    """
    Computes the integral: ∫₀^ℓ f(x, *args) * φₘ(x) dx
    using adaptive Gauss–Legendre quadrature.

    Parameters:
    ----------
    f : callable
        Function to integrate. Signature must be f(x, *args), where x is scalar or array-like.
    m : int
        Mode index for the basis function φₘ.
    ell : float
        Upper bound of the integration interval [0, ℓ].
    *args : tuple
        Additional arguments to pass to f(x, *args).
    tol : float, optional
        Absolute convergence tolerance for quadrature (default: 1e-6).
    max_n : int, optional
        Maximum allowed number of Gauss–Legendre nodes (default: 1000).

    Returns:
    -------
    integral_result : float
        Estimated value of the integral.
    n_points_used : int
        Number of quadrature points used.
    """
    if ell <= 0:
        raise ValueError("The integration domain upper bound 'ell' must be positive.")

    # Define the composite integrand: f(x, ...) * φₘ(x)
    def integrand(x):
        return f(x, *args) * phi_m(m, ell, x)  # Assumes phi_m is defined and vectorized

    # Perform adaptive Gauss-Legendre integration over [0, ell]
    return adaptive_gauss_legendre(integrand, ell, tol=tol, max_n=max_n)


def compute_time_dependent_integrals(
    f: callable,
    n: int,
    N: int,
    ell: float,
    t: np.ndarray
) -> np.ndarray:
    """
    Compute a grid of integrals:
    I[k, m] = ∫₀^ℓ f(x, t[k+1]) * φₘ(x) dx
    for all time steps k = 0 to n-2 and basis modes m = 1 to N.

    Parameters:
    ----------
    f : callable
        Function of the form f(x, t) to be integrated.
    n : int
        Number of time steps in the time discretization.
    N : int
        Number of basis functions φₘ(x).
    ell : float
        Domain upper limit for integration (i.e., length scale).
    t : array-like
        Time discretization points, must have length n.

    Returns:
    -------
    integrals : np.ndarray
        Array of shape (n-1, N) with computed integral values.
    """
    if len(t) != n:
        raise ValueError("Length of time array 't' must match 'n'.")

    if ell <= 0:
        raise ValueError("Parameter 'ell' must be positive.")

    integrals = np.zeros((n - 1, N))  # Preallocate output array

    # Loop through each time step and basis function
    for k in range(n - 1):
        t_k1 = t[k + 1]  # Current evaluation time t_{k+1}
        for m in range(N):
            # Integrate f(x, t_{k+1}) * φₘ(x) over [0, ell]
            integrals[k, m], _ = integrate_with_phi_m(f, m + 1, ell, t_k1)

    return integrals