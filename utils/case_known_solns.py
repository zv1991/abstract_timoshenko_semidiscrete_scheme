# ======================================================
# MODULE IMPORTS
# ======================================================

import sympy as sp                              # Symbolic math engine
import utils.config as cfg                      # Domain length, PDE parameters
from utils.auxiliary import integrate_derivative_form  # Numerical quadrature

from utils.benchmark_solutions import x, t, ell, u_expr, v_expr  # Benchmark expressions and symbols


# ======================================================
# SYMBOLIC DIFFERENTIATION
# ======================================================

def compute_derivatives(expr: sp.Expr, variables: tuple, orders: tuple) -> dict:
    """
    Generate symbolic partial derivatives of an expression.

    Parameters:
    -----------
    expr      : sympy.Expr
        Symbolic expression (e.g., u(x, t))
    variables : tuple
        Variables to differentiate with respect to (e.g., (x, t))
    orders    : tuple
        Orders of derivatives (e.g., (1, 2))

    Returns:
    --------
    dict[str, sp.Expr]
        Dictionary like {'dx1': ∂u/∂x, 'dt2': ∂²u/∂t²}
    """
    return {
        f'd{var.name}{order}': sp.diff(expr, var, order)
        for var in variables
        for order in orders
    }

# Compute derivatives symbolically
u_derivs = compute_derivatives(u_expr, (x, t), (1, 2))
v_derivs = compute_derivatives(v_expr, (x, t), (1, 2))


# ======================================================
# NUMERICAL LAMBDIFICATION
# ======================================================

def lambdify_all(expr_dict: dict) -> dict:
    """
    Lambdify all symbolic expressions using NumPy backend.

    Parameters:
    -----------
    expr_dict : dict[str, sp.Expr]

    Returns:
    --------
    dict[str, callable]
        Dictionary with keys matching the input dict
    """
    return {
        key: sp.lambdify((x, t, ell), expr, modules="numpy")
        for key, expr in expr_dict.items()
    }

# Base displacement and rotation fields
u_func = sp.lambdify((x, t, ell), u_expr, modules="numpy")
v_func = sp.lambdify((x, t, ell), v_expr, modules="numpy")

# Derivative functions
u_funcs = lambdify_all(u_derivs)
v_funcs = lambdify_all(v_derivs)


# ======================================================
# UNIFIED EVALUATION INTERFACE
# ======================================================

def evaluate(f, x_val: float, t_val: float) -> float:
    """
    Evaluate a lambdified function with beam length.

    Parameters:
    -----------
    f      : callable
        Lambdified function of (x, t, ell)
    x_val  : float
        Spatial point
    t_val  : float
        Time point

    Returns:
    --------
    float
        Function evaluated at (x, t, ell)
    """
    return f(x_val, t_val, cfg.ell)


# ======================================================
# PUBLIC INTERFACE: BASE FIELDS u(x, t), v(x, t)
# ======================================================

def u(x_val: float, t_val: float) -> float:
    """Evaluate displacement u(x, t)."""
    return evaluate(u_func, x_val, t_val)

def v(x_val: float, t_val: float) -> float:
    """Evaluate rotation v(x, t)."""
    return evaluate(v_func, x_val, t_val)


# ======================================================
# DERIVATIVE EVALUATION INTERFACE
# ======================================================

# Displacement derivatives
def diff1t_u(x_val: float, t_val: float) -> float: return evaluate(u_funcs['dt1'], x_val, t_val)
def diff2t_u(x_val: float, t_val: float) -> float: return evaluate(u_funcs['dt2'], x_val, t_val)
def diff1x_u(x_val: float, t_val: float) -> float: return evaluate(u_funcs['dx1'], x_val, t_val)
def diff2x_u(x_val: float, t_val: float) -> float: return evaluate(u_funcs['dx2'], x_val, t_val)

# Rotation derivatives
def diff1t_v(x_val: float, t_val: float) -> float: return evaluate(v_funcs['dt1'], x_val, t_val)
def diff2t_v(x_val: float, t_val: float) -> float: return evaluate(v_funcs['dt2'], x_val, t_val)
def diff1x_v(x_val: float, t_val: float) -> float: return evaluate(v_funcs['dx1'], x_val, t_val)
def diff2x_v(x_val: float, t_val: float) -> float: return evaluate(v_funcs['dx2'], x_val, t_val)


# ======================================================
# NONLINEAR ENERGY INTEGRAL TERM
# ======================================================

def integr_term(t_val: float) -> float:
    """
    Compute nonlinear integral ∫₀^ell (∂u/∂x)² dx.

    Parameters:
    -----------
    t_val : float
        Time at which the integral is evaluated

    Returns:
    --------
    float
        Value of the energy-like term
    """
    integrand = lambda x_: diff1x_u(x_, t_val)
    result, *_ = integrate_derivative_form(df=integrand, ell=cfg.ell)
    return result


# ======================================================
# RIGHT-HAND SIDES FOR THE TIMOSHENKO SYSTEM
# ======================================================

def f1(x_val: float, t_val: float) -> float:
    """
    Right-hand side of the u-equation.

    f₁ = ∂²u/∂t² - (α + β ∫(∂u/∂x)² dx) ∂²u/∂x² + a₁ ∂v/∂x
    """
    return (
        diff2t_u(x_val, t_val)
        - (cfg.alpha + cfg.beta * integr_term(t_val)) * diff2x_u(x_val, t_val)
        + cfg.a1 * diff1x_v(x_val, t_val)
    )

def f2(x_val: float, t_val: float) -> float:
    """
    Right-hand side of the v-equation.

    f₂ = ∂²v/∂t² - γ ∂²v/∂x² + δ v - a₂ ∂u/∂x
    """
    return (
        diff2t_v(x_val, t_val)
        - cfg.gamma * diff2x_v(x_val, t_val)
        + cfg.delta * v(x_val, t_val)
        - cfg.a2 * diff1x_u(x_val, t_val)
    )