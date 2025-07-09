# ==========================================================
# Module Imports
# ==========================================================

# NumPy is used for efficient numerical computation, particularly with arrays and vectorized operations
import numpy as np

# leggauss returns the Gauss–Legendre nodes and weights on the interval [-1, 1]
# These are used for approximating integrals using Gauss–Legendre quadrature
from numpy.polynomial.legendre import leggauss

import utils.auxiliary as aux
import utils.config as cfg


# ==========================================================
# Function: gauss_legendre_integral
# Purpose: Numerically integrate a function using Gauss–Legendre quadrature
# ==========================================================
def gauss_legendre_integral(f, a, b, n_gauss):
    """
    Compute the Gauss–Legendre quadrature of function `f` over the interval [a, b].

    This method approximates the definite integral of a given function by evaluating it at
    carefully chosen points (`nodes`) within the interval, weighted by corresponding values
    (`weights`). These values are derived from Legendre polynomials and provide high accuracy
    with fewer evaluation points compared to equally spaced methods.

    Parameters:
        f       : callable
                  Function to integrate. Ideally supports NumPy vectorized input.
        a       : float
                  Lower limit of integration.
        b       : float
                  Upper limit of integration.
        n_gauss : int
                  Number of Gauss–Legendre nodes to use (higher = better accuracy).

    Returns:
        float
            Approximate value of the integral ∫ₐᵇ f(x) dx using Gauss–Legendre quadrature.
    """

    # ==========================================================
    # Step 1: Generate Gauss–Legendre nodes and weights on [-1, 1]
    # These are roots of the Legendre polynomial of degree `n_gauss`
    # ==========================================================
    nodes, weights = leggauss(n_gauss)

    # ==========================================================
    # Step 2: Affine Transformation of Nodes to Interval [a, b]
    # Convert from reference interval [-1, 1] to user-defined interval [a, b]
    # x = 0.5 * (b - a) * node + 0.5 * (a + b)
    # ==========================================================
    mid = 0.5 * (a + b)                # Center point of [a, b]
    half_len = 0.5 * (b - a)           # Half-length of [a, b]
    x_mapped = mid + half_len * nodes  # Transformed nodes in [a, b]

    # ==========================================================
    # Step 3: Evaluate Function at Transformed Nodes
    # Try vectorized evaluation first; fallback to scalar loop if needed
    # ==========================================================
    try:
        # Attempt vectorized evaluation for performance
        f_vals = np.asarray(f(x_mapped))

        # Confirm the shape of the returned array matches expectations
        if f_vals.shape != x_mapped.shape:
            raise ValueError("Function output shape mismatch. Expected shape: "
                             f"{x_mapped.shape}, got: {f_vals.shape}")
    except Exception:
        # Graceful fallback for non-vectorized functions
        # Also useful for debugging evaluation issues
        f_vals = np.array([f(xi) for xi in x_mapped])

    # ==========================================================
    # Step 4: Compute Weighted Sum to Approximate Integral
    # Apply the Gauss–Legendre formula: ∫ f(x) dx ≈ Σ wᵢ·f(xᵢ)·(b−a)/2
    # ==========================================================
    result = half_len * np.dot(weights, f_vals)

    return result

# ==========================================================
# Function: halving_gauss_legendre_quadrature
# Purpose: Adaptive integration using Gauss–Legendre quadrature
#          with increasing node count and recursive interval halving
# ==========================================================
def halving_gauss_legendre_quadrature(
    f: callable,
    ell: float,
    tol: float = 1e-6
) -> tuple[float, float, int]:
    """
    Approximate the integral of a function `f` over the interval [0, ell]
    using Gauss–Legendre quadrature with adaptive node refinement and interval halving.

    The method first attempts to converge over the whole interval by increasing
    the number of Gauss–Legendre nodes. If convergence fails, the interval is
    halved recursively and the same process is applied per subinterval.

    Parameters:
        f   : callable
              Function to integrate. Must accept float input.
        ell : float
              Upper bound of the interval [0, ell]; must be non-negative.
        tol : float, optional
              Convergence tolerance for absolute error (default: 1e-6).

    Returns:
        tuple[float, float, int]
            (final integral estimate, estimated error, number of halving iterations)
    """

    # ==========================================================
    # Step 1: Validate input
    # ==========================================================
    if ell < 0:
        raise ValueError("Parameter 'ell' must be non-negative.")
    if ell == 0:
        return 0.0, 0.0, 0

    # ==========================================================
    # Step 2: Try convergence using increasing n_gauss on [0, ell]
    # ==========================================================
    n_gauss = 5
    max_gauss = 50
    converged = False

    integral_prev = gauss_legendre_integral(f, 0.0, ell, n_gauss)

    while n_gauss <= max_gauss:
        n_gauss += 5
        integral_curr = gauss_legendre_integral(f, 0.0, ell, n_gauss)

        if abs(integral_curr - integral_prev) < tol:
            estimated_error = abs(integral_curr - integral_prev)
            converged = True
            return integral_curr, estimated_error, 0  # Converged on full interval

        integral_prev = integral_curr

    # ==========================================================
    # Step 3: If not converged, begin interval halving
    # ==========================================================
    min_dx = 1 / 128.0
    counter = 0
    prev_total = None

    while not converged:
        counter += 1
        n_intervals = 2 ** counter
        dx = ell / n_intervals

        if dx < min_dx:
            break  # Stop refinement if intervals become too small

        total_integral = 0.0
        converged = True  # Assume converged unless any subinterval fails

        for i in range(n_intervals):
            a = i * dx
            b = (i + 1) * dx

            # Gauss–Legendre convergence on each subinterval
            n_gauss_local = 5
            integral_prev = gauss_legendre_integral(f, a, b, n_gauss_local)

            local_converged = False

            while n_gauss_local <= max_gauss:
                n_gauss_local += 5
                integral_curr = gauss_legendre_integral(f, a, b, n_gauss_local)

                if abs(integral_curr - integral_prev) < tol:
                    local_converged = True
                    total_integral += integral_curr
                    break

                integral_prev = integral_curr

            if not local_converged:
                # Still accept the last computed value even if not converged
                total_integral += integral_curr
                converged = False  # At least one subinterval failed to converge

        # Check for convergence on the entire domain
        if converged:
            if prev_total is not None and abs(total_integral - prev_total) < tol:
                estimated_error = abs(total_integral - prev_total)
                return total_integral, estimated_error, counter

            prev_total = total_integral

    # ==========================================================
    # Step 4: Fallback — return last estimate if min_dx reached
    # ==========================================================
    estimated_error = abs(total_integral - prev_total) if prev_total is not None else float('inf')
    return total_integral, estimated_error, counter


f = lambda x: (55 * np.pi) * np.sin(55 * np.pi * x)

integral_global = gauss_legendre_integral(f=f, a=0.0, b=1.0, n_gauss=51)

current_estimate, estimated_error, counter = halving_gauss_legendre_quadrature(f=f, ell=1.0)