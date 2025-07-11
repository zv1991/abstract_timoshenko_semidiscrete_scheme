# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np                      # NumPy: core numerical computation and math
import utils.config as cfg              # Problem-specific configuration with constants:
                                        # e.g., ell (domain length), alpha, beta, a1, a2, etc.

# ======================================================
# GLOBAL CONSTANTS
# ======================================================

lam = 17  # Oscillation frequency (mode index) used in the benchmark solutions

# ======================================================
# MULTIPLIER FUNCTION AND ITS DERIVATIVES
# ======================================================

# ------------------------------------------------------
# Function: g
# Purpose: Time-dependent multiplier function g(t)
# Used in exact benchmark solutions u(x, t) and v(x, t)
# ------------------------------------------------------
def g(t: float) -> float:
    """
    Time-dependent multiplier function:
        g(t) = exp(πt)

    Parameters
    ----------
    t : float
        Time input

    Returns
    -------
    float
        Value of g(t) = exp(πt)
    """
    return np.exp(np.pi * t)

# ------------------------------------------------------
# Function: dg
# Purpose: First time derivative of g(t)
# ------------------------------------------------------
def dg(t: float) -> float:
    """
    First derivative of g(t) with respect to time:
        dg(t) = d/dt [exp(πt)] = π * exp(πt)

    Parameters
    ----------
    t : float
        Time input

    Returns
    -------
    float
        First derivative dg(t)
    """
    return np.pi * np.exp(np.pi * t)

# ------------------------------------------------------
# Function: d2g
# Purpose: Second time derivative of g(t)
# ------------------------------------------------------
def d2g(t: float) -> float:
    """
    Second derivative of g(t) with respect to time:
        d²g(t)/dt² = d/dt [π * exp(πt)] = π² * exp(πt)

    Parameters
    ----------
    t : float
        Time input

    Returns
    -------
    float
        Second derivative d²g(t)/dt²
    """
    return np.pi ** 2 * np.exp(np.pi * t)

# # ------------------------------------------------------
# # Function: g
# # Purpose: Defines the time-dependent multiplier g(t)
# # ------------------------------------------------------
# def g(t: float) -> float:
#     """Time-dependent multiplier function for u and v."""
#     return t  # Linear multiplier for simplicity

# # ------------------------------------------------------
# # Function: dg
# # Purpose: First time derivative of g(t)
# # ------------------------------------------------------
# def dg(t: float) -> float:
#     """First derivative of g(t) with respect to time."""
#     return 1.0  # Derivative of t is constant

# # ------------------------------------------------------
# # Function: d2g
# # Purpose: Second time derivative of g(t)
# # ------------------------------------------------------
# def d2g(t: float) -> float:
#     """Second derivative of g(t) with respect to time."""
#     return 0.0  # Second derivative of t is zero

# ======================================================
# EXACT SOLUTIONS (DISPLACEMENT u AND ROTATION v)
# ======================================================

# ------------------------------------------------------
# Function: u
# Purpose: Exact displacement u(x, t)
# ------------------------------------------------------
def u(x: float, t: float) -> float:
    """
    Compute the exact displacement solution u(x, t).

    Returns
    -------
    float : Value of u at point (x, t)
    """
    return np.sin(lam * np.pi * x / cfg.ell) * g(t)

# ------------------------------------------------------
# Function: v
# Purpose: Exact rotation v(x, t)
# ------------------------------------------------------
def v(x: float, t: float) -> float:
    """
    Compute the exact rotation solution v(x, t).

    Returns
    -------
    float : Value of v at point (x, t)
    """
    return np.sin(lam * np.pi * x / cfg.ell) * g(t)

# ======================================================
# PARTIAL DERIVATIVES OF DISPLACEMENT u(x, t)
# ======================================================

# ------------------------------------------------------
# Function: diff1t_u
# Purpose: ∂u/∂t
# ------------------------------------------------------
def diff1t_u(x: float, t: float) -> float:
    return np.sin(lam * np.pi * x / cfg.ell) * dg(t)

# ------------------------------------------------------
# Function: diff2t_u
# Purpose: ∂²u/∂t²
# ------------------------------------------------------
def diff2t_u(x: float, t: float) -> float:
    return np.sin(lam * np.pi * x / cfg.ell) * d2g(t)

# ------------------------------------------------------
# Function: diff1x_u
# Purpose: ∂u/∂x
# ------------------------------------------------------
def diff1x_u(x: float, t: float) -> float:
    return (lam * np.pi / cfg.ell) * np.cos(lam * np.pi * x / cfg.ell) * g(t)

# ------------------------------------------------------
# Function: diff2x_u
# Purpose: ∂²u/∂x²
# ------------------------------------------------------
def diff2x_u(x: float, t: float) -> float:
    return - (lam * np.pi / cfg.ell) ** 2 * np.sin(lam * np.pi * x / cfg.ell) * g(t)

# ======================================================
# PARTIAL DERIVATIVES OF ROTATION v(x, t)
# ======================================================

# ------------------------------------------------------
# Function: diff1t_v
# Purpose: ∂v/∂t
# ------------------------------------------------------
def diff1t_v(x: float, t: float) -> float:
    return np.sin(lam * np.pi * x / cfg.ell) * dg(t)

# ------------------------------------------------------
# Function: diff2t_v
# Purpose: ∂²v/∂t²
# ------------------------------------------------------
def diff2t_v(x: float, t: float) -> float:
    return np.sin(lam * np.pi * x / cfg.ell) * d2g(t)

# ------------------------------------------------------
# Function: diff1x_v
# Purpose: ∂v/∂x
# ------------------------------------------------------
def diff1x_v(x: float, t: float) -> float:
    return (lam * np.pi / cfg.ell) * np.cos(lam * np.pi * x / cfg.ell) * g(t)

# ------------------------------------------------------
# Function: diff2x_v
# Purpose: ∂²v/∂x²
# ------------------------------------------------------
def diff2x_v(x: float, t: float) -> float:
    return - (lam * np.pi / cfg.ell) ** 2 * np.sin(lam * np.pi * x / cfg.ell) * g(t)

# ======================================================
# NONLINEAR ENERGY-LIKE INTEGRAL TERM
# ======================================================

# ------------------------------------------------------
# Function: integr_term
# Purpose: ∫₀^ℓ (∂u/∂x)² dx
# ------------------------------------------------------
def integr_term(t: float) -> float:
    """
    Compute the nonlinear energy-like term in the Timoshenko system:
        ∫₀^ℓ (∂u/∂x)² dx = (λπ)² / (2ℓ) * [g(t)]²

    Parameters
    ----------
    t : float
        Time value

    Returns
    -------
    float
        Integral value at time t
    """
    return (lam * np.pi) ** 2 / (2.0 * cfg.ell) * g(t) ** 2

# ======================================================
# RIGHT-HAND SIDE (RHS) FUNCTIONS FOR THE PDE SYSTEM
# ======================================================

# ------------------------------------------------------
# Function: f1
# Purpose: RHS of displacement equation (u)
# ------------------------------------------------------
def f1(x: float, t: float) -> float:
    """
    Compute right-hand side of the displacement equation:

        f₁ = ∂²u/∂t² - (α + β ∫(∂u/∂x)² dx) ∂²u/∂x² + a₁ ∂v/∂x

    Parameters
    ----------
    x : float
        Spatial coordinate

    t : float
        Time coordinate

    Returns
    -------
    float
        Value of f₁(x, t)
    """
    return (
        diff2t_u(x, t)
        - (cfg.alpha + cfg.beta * integr_term(t)) * diff2x_u(x, t)
        + cfg.a1 * diff1x_v(x, t)
    )

# ------------------------------------------------------
# Function: f2
# Purpose: RHS of rotation equation (v)
# ------------------------------------------------------
def f2(x: float, t: float) -> float:
    """
    Compute right-hand side of the rotation equation:

        f₂ = ∂²v/∂t² - γ ∂²v/∂x² + δ v - a₂ ∂u/∂x

    Parameters
    ----------
    x : float
        Spatial coordinate

    t : float
        Time coordinate

    Returns
    -------
    float
        Value of f₂(x, t)
    """
    return (
        diff2t_v(x, t)
        - cfg.gamma * diff2x_v(x, t)
        + cfg.delta * v(x, t)
        - cfg.a2 * diff1x_u(x, t)
    )
