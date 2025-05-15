# ----------------------------------------------------------------------
# Imports and Configuration
# ----------------------------------------------------------------------

import config as cfg                   # Domain and model constants
import auxiliary as aux               # Basis functions and helpers
from autoDiff import AutoDiff         # Differentiation wrapper

import jax
import jax.numpy as jnp               # JAX-compatible NumPy

from scipy.integrate import quad      # Adaptive quadrature

jax.config.update("jax_enable_x64", True)  # High-precision mode


# ----------------------------------------------------------------------
# Benchmark Solutions: u(x, t) and v(x, t)
# ----------------------------------------------------------------------

def u(x: float, t: float) -> float:
    """
    Benchmark solution: u(x, t) = t * φ₁(x),
    where φ₁ is the first mode of the basis functions.
    """
    return t * aux.phi_m(1, cfg.ell, x)


def v(x: float, t: float) -> float:
    """
    Benchmark solution: v(x, t) = t * φ₁(x).
    Mirrors u(x, t) for symmetry in coupled systems.
    """
    return t * aux.phi_m(1, cfg.ell, x)


# ----------------------------------------------------------------------
# AutoDiff Initialization
# ----------------------------------------------------------------------

ad_u = AutoDiff(u)  # For derivatives of u
ad_v = AutoDiff(v)  # For derivatives of v


# ----------------------------------------------------------------------
# Derivative Functions
# ----------------------------------------------------------------------

# Second-order time derivatives
def diff2t_u(x: float, t: float) -> float:
    return ad_u.second_deriv(x, t, var='t')

def diff2t_v(x: float, t: float) -> float:
    return ad_v.second_deriv(x, t, var='t')

# Second-order spatial derivatives
def diff2x_u(x: float, t: float) -> float:
    return ad_u.second_deriv(x, t, var='x')

def diff2x_v(x: float, t: float) -> float:
    return ad_v.second_deriv(x, t, var='x')

# First-order spatial derivatives
def diff1x_u(x: float, t: float) -> float:
    return ad_u.first_deriv(x, t, var='x')

def diff1x_v(x: float, t: float) -> float:
    return ad_v.first_deriv(x, t, var='x')


# ----------------------------------------------------------------------
# Integration Utility
# ----------------------------------------------------------------------

def adaptive_quadrature(f, ell: float, tol: float = 1e-6) -> tuple[float, float]:
    """
    Numerically compute ∫₀^ell f(x) dx using adaptive quadrature.

    Args:
        f   : Callable[[float], float], function to integrate
        ell : Upper limit of integration
        tol : Absolute error tolerance

    Returns:
        (integral, error): Approximate result and integration error
    """
    if ell <= 0:
        return 0.0, 0.0

    def integrand(x):
        return float(f(x))  # Ensure scalar output

    result, error = quad(integrand, 0, ell, epsabs=tol)
    return result, error


# ----------------------------------------------------------------------
# Coupling Term
# ----------------------------------------------------------------------

def integr_term(t: float) -> float:
    """
    Computes the nonlocal energy-like term:
        ∫₀^ell (∂u/∂x)² dx at time t.
    Used to scale ∂²u/∂x² in f1.
    """
    def integrand(x):
        return diff1x_u(x, t) ** 2

    result, _ = adaptive_quadrature(integrand, cfg.ell)
    return result


# ----------------------------------------------------------------------
# RHS of Coupled PDE System
# ----------------------------------------------------------------------

def f1(x: float, t: float) -> float:
    """
    RHS of the first PDE:
        ∂²u/∂t² - (α + β ∫ (∂u/∂x)² dx) ∂²u/∂x² + a₁ ∂v/∂x
    """
    nonlinear_coeff = cfg.alpha + cfg.beta * integr_term(t)
    return (
        diff2t_u(x, t)
        - nonlinear_coeff * diff2x_u(x, t)
        + cfg.a1 * diff1x_v(x, t)
    )


def f2(x: float, t: float) -> float:
    """
    RHS of the second PDE:
        ∂²v/∂t² - γ ∂²v/∂x² + δ v - a₂ ∂u/∂x
    """
    return (
        diff2t_v(x, t)
        - cfg.gamma * diff2x_v(x, t)
        + cfg.delta * v(x, t)
        - cfg.a2 * diff1x_u(x, t)
    )