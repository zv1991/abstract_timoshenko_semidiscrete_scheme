# ======================================================
# MODULE IMPORTS
# ======================================================

import sympy as sp  # Symbolic math engine (algebra, calculus, simplification)
import utils.config as cfg  # Problem-specific constants: ell, alpha, beta, etc.
from utils.auxiliary import integrate_derivative_form  # Numerical quadrature for integrals


# ======================================================
# SYMBOLIC VARIABLES AND FIXED CONSTANTS
# ======================================================

# Define symbolic variables for space and time
x, t = sp.symbols('x t', real=True)

# Fixed domain length (numeric value, not symbolic)
ell = cfg.ell


# ======================================================
# GALERKIN BASIS FUNCTION DEFINITIONS
# ======================================================

def coeff_A_sym(m: int) -> sp.Expr:
    """
    Compute normalization coefficient for Galerkin basis functions.
    A_m = 1 / sqrt(2m + 1)

    Parameters:
    -----------
    m : int
        Basis function index

    Returns:
    --------
    sp.Expr
        Symbolic normalization coefficient A_m
    """
    return 1 / sp.sqrt(2 * m + 1)


def shifted_legendre_sym(m: int, x_sym: sp.Symbol) -> sp.Expr:
    """
    Construct the m-th shifted Legendre polynomial over [0, ell].

    The standard domain [-1, 1] is mapped from [0, ell] using:
        ξ = 2x / ell - 1

    Parameters:
    -----------
    m : int
        Degree of the Legendre polynomial
    x_sym : sp.Symbol
        Symbolic spatial variable

    Returns:
    --------
    sp.Expr
        Shifted Legendre polynomial P_m(ξ)
    """
    xi = 2 * x_sym / ell - 1
    return sp.legendre(m, xi)


def phi_m_sym(m: int, x_sym: sp.Symbol) -> sp.Expr:
    """
    Construct φ_m(x), the m-th Galerkin basis function.

    φ_m(x) = (√ell / 2) * A_m * [P_{m+1}(ξ) - P_{m-1}(ξ)]

    Parameters:
    -----------
    m : int
        Index of the basis function (must be ≥ 1)
    x_sym : sp.Symbol
        Symbolic spatial variable

    Returns:
    --------
    sp.Expr
        Symbolic Galerkin basis function φ_m(x)

    Raises:
    -------
    ValueError
        If m < 1
    """
    if m < 1:
        raise ValueError("Basis index m must be ≥ 1.")

    A_m = coeff_A_sym(m)
    return (sp.sqrt(ell) / 2) * A_m * (
        shifted_legendre_sym(m + 1, x_sym) - shifted_legendre_sym(m - 1, x_sym)
    )


# ======================================================
# SYMBOLIC TEST FIELDS (BENCHMARK DISPLACEMENT/ROTATION)
# ======================================================

# def u_sym(x_sym: sp.Symbol, t_sym: sp.Symbol) -> sp.Expr:
#     """Define symbolic displacement field: u(x, t) = t · φ₁(x)"""
#     return sp.exp(sp.pi * t_sym) * sp.sin(14 * sp.pi * x / ell)

# def v_sym(x_sym: sp.Symbol, t_sym: sp.Symbol) -> sp.Expr:
#     """Define symbolic rotation field: v(x, t) = t · φ₁(x)"""
#     return sp.exp(sp.pi * t_sym) * sp.sin(14 * sp.pi * x / ell)

# def u_sym(x_sym: sp.Symbol, t_sym: sp.Symbol) -> sp.Expr:
#     """
#     Symbolic displacement field u(x, t) for the Timoshenko model.

#     Parameters
#     ----------
#     x_sym : sp.Symbol
#         Symbol representing the spatial variable x.
#     t_sym : sp.Symbol
#         Symbol representing the time variable t.

#     Returns
#     -------
#     sp.Expr
#         Symbolic expression for u(x, t) = exp(π·t) * φₘ(x)
#     """
#     return sp.exp(sp.pi * t_sym) * phi_m_sym(35, x_sym)  # using mode m=35


# # ------------------------------------------------------------------
# # Define symbolic rotation field v(x, t)
# # ------------------------------------------------------------------
# def v_sym(x_sym: sp.Symbol, t_sym: sp.Symbol) -> sp.Expr:
#     """
#     Symbolic rotation field v(x, t) for the Timoshenko model.

#     Parameters
#     ----------
#     x_sym : sp.Symbol
#         Symbol representing the spatial variable x.
#     t_sym : sp.Symbol
#         Symbol representing the time variable t.

#     Returns
#     -------
#     sp.Expr
#         Symbolic expression for v(x, t) = exp(π·t) * φₘ(x)
#     """
#     return sp.exp(sp.pi * t_sym) * phi_m_sym(35, x_sym)  # same basis function

def u_sym(x_sym: sp.Symbol, t_sym: sp.Symbol) -> sp.Expr:
    """Define symbolic displacement field: u(x, t) = t · φ₁(x)"""
    return 64.0 * t_sym * phi_m_sym(5, x_sym)

def v_sym(x_sym: sp.Symbol, t_sym: sp.Symbol) -> sp.Expr:
    """Define symbolic rotation field: v(x, t) = t · φ₁(x)"""
    return 64.0 * t_sym * phi_m_sym(5, x_sym)


# ======================================================
# SYMBOLIC EXPRESSIONS (PRECOMPUTED)
# ======================================================

u_expr = u_sym(x, t)  # Symbolic expression for u(x, t)
v_expr = v_sym(x, t)  # Symbolic expression for v(x, t)


# ======================================================
# COMPUTATION OF SYMBOLIC DERIVATIVES
# ======================================================

def compute_derivatives(expr: sp.Expr, variables: tuple, orders: tuple) -> dict:
    """
    Compute specified-order partial derivatives of a symbolic expression.

    Parameters:
    -----------
    expr : sp.Expr
        Symbolic function (e.g., u or v)
    variables : tuple
        Tuple of variables to differentiate with respect to
    orders : tuple
        Tuple of derivative orders to compute

    Returns:
    --------
    dict
        Derivative expressions indexed by string keys (e.g., 'dx1', 'dt2')
    """
    return {
        f'd{var.name}{order}': sp.diff(expr, var, order)
        for var in variables
        for order in orders
    }

# First and second derivatives for u and v
u_derivs = compute_derivatives(u_expr, (x, t), (1, 2))
v_derivs = compute_derivatives(v_expr, (x, t), (1, 2))


# ======================================================
# NUMERICAL LAMBDIFICATION (SYM -> NUMPY CALLABLE)
# ======================================================

def lambdify_all(expr_dict: dict) -> dict:
    """
    Lambdify all symbolic expressions to numerical functions using NumPy.

    Parameters:
    -----------
    expr_dict : dict
        Dictionary of symbolic expressions

    Returns:
    --------
    dict
        Dictionary of lambdified (x, t) functions
    """
    return {
        key: sp.lambdify((x, t), expr, modules="numpy")
        for key, expr in expr_dict.items()
    }

# Lambdified base fields
u_func = sp.lambdify((x, t), u_expr, modules="numpy")
v_func = sp.lambdify((x, t), v_expr, modules="numpy")

# Lambdified derivatives
u_funcs = lambdify_all(u_derivs)
v_funcs = lambdify_all(v_derivs)


# ======================================================
# FUNCTION EVALUATION INTERFACE
# ======================================================

def evaluate(f, x_val: float, t_val: float) -> float:
    """
    Evaluate a lambdified function at numerical input.

    Parameters:
    -----------
    f : callable
        A NumPy-compatible function f(x, t)
    x_val : float
        Spatial coordinate
    t_val : float
        Time coordinate

    Returns:
    --------
    float
        Evaluated function value
    """
    return f(x_val, t_val)

# Base fields
def u(x_val: float, t_val: float) -> float: return evaluate(u_func, x_val, t_val)
def v(x_val: float, t_val: float) -> float: return evaluate(v_func, x_val, t_val)


# ======================================================
# DERIVATIVE EVALUATION INTERFACE
# ======================================================

# Displacement u derivatives
def diff1t_u(x_val: float, t_val: float) -> float: return evaluate(u_funcs['dt1'], x_val, t_val)
def diff2t_u(x_val: float, t_val: float) -> float: return evaluate(u_funcs['dt2'], x_val, t_val)
def diff1x_u(x_val: float, t_val: float) -> float: return evaluate(u_funcs['dx1'], x_val, t_val)
def diff2x_u(x_val: float, t_val: float) -> float: return evaluate(u_funcs['dx2'], x_val, t_val)

# Rotation v derivatives
def diff1t_v(x_val: float, t_val: float) -> float: return evaluate(v_funcs['dt1'], x_val, t_val)
def diff2t_v(x_val: float, t_val: float) -> float: return evaluate(v_funcs['dt2'], x_val, t_val)
def diff1x_v(x_val: float, t_val: float) -> float: return evaluate(v_funcs['dx1'], x_val, t_val)
def diff2x_v(x_val: float, t_val: float) -> float: return evaluate(v_funcs['dx2'], x_val, t_val)


# ======================================================
# NONLINEAR ENERGY INTEGRAL TERM
# ======================================================

def integr_term(t_val: float) -> float:
    """
    Compute nonlinear energy-like term:
        ∫₀^ell (∂u/∂x)² dx

    Parameters:
    -----------
    t_val : float
        Time value at which to evaluate the integral

    Returns:
    --------
    float
        Scalar result of the integral
    """
    integrand = lambda x_: diff1x_u(x_, t_val)
    result, *_ = integrate_derivative_form(df=integrand, ell=ell)
    return result


# ======================================================
# TIMOSHENKO RIGHT-HAND SIDES
# ======================================================

def f1(x_val: float, t_val: float) -> float:
    """
    Right-hand side of displacement equation (u):

    f₁ = ∂²u/∂t² - (α + β ∫(∂u/∂x)² dx) ∂²u/∂x² + a₁ ∂v/∂x

    Returns:
    --------
    float
        f₁(x, t) value
    """
    return (
        diff2t_u(x_val, t_val)
        - (cfg.alpha + cfg.beta * integr_term(t_val)) * diff2x_u(x_val, t_val)
        + cfg.a1 * diff1x_v(x_val, t_val)
    )


def f2(x_val: float, t_val: float) -> float:
    """
    Right-hand side of rotation equation (v):

    f₂ = ∂²v/∂t² - γ ∂²v/∂x² + δ v - a₂ ∂u/∂x

    Returns:
    --------
    float
        f₂(x, t) value
    """
    return (
        diff2t_v(x_val, t_val)
        - cfg.gamma * diff2x_v(x_val, t_val)
        + cfg.delta * v(x_val, t_val)
        - cfg.a2 * diff1x_u(x_val, t_val)
    )