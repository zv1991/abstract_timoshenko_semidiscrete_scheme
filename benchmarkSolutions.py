# Import project-specific configuration module (e.g., domain size, constants)
import config as cfg

# Import auxiliary functions
import auxiliary as aux

# Import AutoDiff class for partial derivatives
from autoDiff import AutoDiff

# Import core JAX library for automatic differentiation and JIT compilation
import jax

# Enable 64-bit floating-point precision for more accurate numerical computations
jax.config.update("jax_enable_x64", True)

# Import quad from scipy.integrate
from scipy.integrate import quad

# Import JAX's version of NumPy under the alias 'jnp' for array operations
import jax.numpy as jnp


# --------------------------------------------------------------------------------
# Benchmark Solutions: Exact Test Functions u(x, t) and v(x, t)
# --------------------------------------------------------------------------------

def u(x: float, t: float) -> float:
    """
    Define the benchmark/test solution u(x, t) as:
        u(x, t) = t * phi_1(x)

    phi_1 is typically a basis function or mode.

    Args:
        x (float): Spatial coordinate
        t (float): Temporal coordinate

    Returns:
        float: Value of the test function u at position x and time t
    """
    return t * aux.phi_m(1, cfg.ell, x)  # Compute t * φ₁(x), where φ₁ is mode 1 of the basis


def v(x: float, t: float) -> float:
    """
    Define the benchmark/test solution v(x, t), which is identical to u(x, t):
        v(x, t) = t * phi_1(x)

    Args:
        x (float): Spatial coordinate
        t (float): Temporal coordinate

    Returns:
        float: Value of the test function v at position x and time t
    """
    return t * aux.phi_m(1, cfg.ell, x)  # Same as u(x, t); included for symmetry/testing

# Create an AutoDiff object
ad_u = AutoDiff(u)
ad_v = AutoDiff(v)

""" Initial conditions for u and v and their derivatives """

def diff2t_u(x, t):
    return ad_u.second_deriv(x, t, var='t')

def diff2t_v(x, t):
    return ad_v.second_deriv(x, t, var='t')

def diff2x_u(x, t):
    return ad_u.second_deriv(x, t, var='x')

def diff2x_v(x, t):
    return ad_v.second_deriv(x, t, var='x')

def diff1x_u(x, t):
    return ad_u.first_deriv(x, t, var='x')

def diff1x_v(x, t):
    return ad_v.first_deriv(x, t, var='x')

def diff1t_u(x, t):
    return ad_u.first_deriv(x, t, var='x')

def adaptive_quadrature(f, ell, tol=1e-6):
    """
    Adaptive integration using scipy.integrate.quad to approximate the integral of f(x) over [0, ell].

    Args:
        f      : Callable, the function f(x) to integrate. Must support JAX arrays.
        ell    : Upper limit of integration (lower limit is fixed at 0).
        tol    : Desired absolute tolerance for convergence (default: 1e-6).

    Returns:
        integral : Approximated value of the integral.
        error    : Estimated error in the integral.
    """
    # Handle edge case: zero-width interval
    if ell == 0:
        return 0.0, 0.0

    # Define the function to integrate
    # Here, we need to ensure the function `f` can work with NumPy arrays directly
    def integrand(x):
        return f(x)

    # Perform the integration using quad
    result, error = quad(integrand, 0, ell, epsabs=tol)

    return result, error

def integr_term(t):
    """
    Compute the integral of (u_x')^2 over the interval [0, ell].
    
    Args:
        t (float): Time variable.
        
    Returns:
        float: Result of the integral of (u_x')^2 over [0, ell].
    """
    # Define the function to integrate: (u_x')^2
    integrand = lambda x: diff1t_u(x, t) ** 2  # This is the integrand, (u_x')^2.
    
    # Compute the integral using adaptive Gauss-Legendre quadrature
    result, _ = adaptive_quadrature(integrand, cfg.ell)
    return result

print(integr_term(5))