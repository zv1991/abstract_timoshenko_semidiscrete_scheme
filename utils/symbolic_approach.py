# ======================================================
# MODULE IMPORTS
# ======================================================

import sympy as sp  # Symbolic mathematics library
from utils.auxiliary import unified_adaptive_quadrature  # Custom numerical integration function
import config as cfg  # Configuration module for constants: ell, alpha, beta, a1, gamma, delta, a2

# ======================================================
# SYMBOLIC VARIABLE DECLARATIONS
# ======================================================

# Declare symbolic variables for position (x), time (t), and domain length (ell)
x, t, ell = sp.symbols('x t ell', real=True, positive=True)

# ======================================================
# GALERKIN BASIS FUNCTION DEFINITIONS
# ======================================================

def coeff_A_sym(m: int) -> sp.Expr:
    """
    Normalization coefficient for orthonormal basis:
    A_m = 1 / sqrt(2m + 1)
    """
    return 1 / sp.sqrt(2 * m + 1)


def shifted_legendre_sym(m: int, ell_sym: sp.Symbol, x_sym: sp.Symbol) -> sp.Expr:
    """
    Returns the m-th shifted Legendre polynomial P_m(ξ)
    Shift: ξ = 2x/ell - 1 maps x ∈ [0, ell] to ξ ∈ [-1, 1]
    """
    xi = 2 * x_sym / ell_sym - 1
    return sp.legendre(m, xi)


def phi_m_sym(m: int, ell_sym: sp.Symbol, x_sym: sp.Symbol) -> sp.Expr:
    """
    Galerkin basis function φ_m(x):
    φ_m(x) = (√ell / 2) * A_m * [P_{m+1}(ξ) - P_{m-1}(ξ)]
    Requires m ≥ 1
    """
    if m < 1:
        raise ValueError("Basis index m must be ≥ 1.")

    A_m = coeff_A_sym(m)
    return (sp.sqrt(ell_sym) / 2) * A_m * (
        shifted_legendre_sym(m + 1, ell_sym, x_sym) -
        shifted_legendre_sym(m - 1, ell_sym, x_sym)
    )

# ======================================================
# SYMBOLIC ANALYTICAL TEST FUNCTIONS
# ======================================================

def u_sym(x_sym: sp.Symbol, t_sym: sp.Symbol, ell_sym: sp.Symbol) -> sp.Expr:
    """Test function: u(x, t) = t * φ₁(x)"""
    return t_sym * phi_m_sym(1, ell_sym, x_sym)


def v_sym(x_sym: sp.Symbol, t_sym: sp.Symbol, ell_sym: sp.Symbol) -> sp.Expr:
    """Test function: v(x, t) = t * φ₁(x)"""
    return t_sym * phi_m_sym(1, ell_sym, x_sym)

# ======================================================
# PRECOMPUTED SYMBOLIC EXPRESSIONS
# ======================================================

u_expr = u_sym(x, t, ell)  # Symbolic form of u(x, t)
v_expr = v_sym(x, t, ell)  # Symbolic form of v(x, t)

# ======================================================
# SYMBOLIC DERIVATIVES
# ======================================================

# --- Time derivatives ---
def diff1t_u_sym(): return sp.diff(u_expr, t)
def diff2t_u_sym(): return sp.diff(u_expr, t, 2)
def diff1t_v_sym(): return sp.diff(v_expr, t)
def diff2t_v_sym(): return sp.diff(v_expr, t, 2)

# --- Space derivatives ---
def diff1x_u_sym(): return sp.diff(u_expr, x)
def diff2x_u_sym(): return sp.diff(u_expr, x, 2)
def diff1x_v_sym(): return sp.diff(v_expr, x)
def diff2x_v_sym(): return sp.diff(v_expr, x, 2)

# ======================================================
# LAMBDIFIED FUNCTIONS FOR NUMERICAL EVALUATION
# ======================================================

# --- Base functions ---
u_func = sp.lambdify((x, t, ell), u_expr, modules="numpy")
v_func = sp.lambdify((x, t, ell), v_expr, modules="numpy")

# --- Time derivatives ---
du_dt_func = sp.lambdify((x, t, ell), diff1t_u_sym(), modules="numpy")
d2u_dt2_func = sp.lambdify((x, t, ell), diff2t_u_sym(), modules="numpy")
dv_dt_func = sp.lambdify((x, t, ell), diff1t_v_sym(), modules="numpy")
d2v_dt2_func = sp.lambdify((x, t, ell), diff2t_v_sym(), modules="numpy")

# --- Space derivatives ---
du_dx_func = sp.lambdify((x, t, ell), diff1x_u_sym(), modules="numpy")
d2u_dx2_func = sp.lambdify((x, t, ell), diff2x_u_sym(), modules="numpy")
dv_dx_func = sp.lambdify((x, t, ell), diff1x_v_sym(), modules="numpy")
d2v_dx2_func = sp.lambdify((x, t, ell), diff2x_v_sym(), modules="numpy")

# ======================================================
# WRAPPED NUMERICAL INTERFACES USING CONFIGURATION (cfg.ell)
# ======================================================

# --- Function evaluations ---
def u(x_val: float, t_val: float) -> float: return u_func(x_val, t_val, cfg.ell)
def v(x_val: float, t_val: float) -> float: return v_func(x_val, t_val, cfg.ell)

# --- Time derivatives ---
def diff1t_u(x_val: float, t_val: float) -> float: return du_dt_func(x_val, t_val, cfg.ell)
def diff2t_u(x_val: float, t_val: float) -> float: return d2u_dt2_func(x_val, t_val, cfg.ell)
def diff1t_v(x_val: float, t_val: float) -> float: return dv_dt_func(x_val, t_val, cfg.ell)
def diff2t_v(x_val: float, t_val: float) -> float: return d2v_dt2_func(x_val, t_val, cfg.ell)

# --- Space derivatives ---
def diff1x_u(x_val: float, t_val: float) -> float: return du_dx_func(x_val, t_val, cfg.ell)
def diff2x_u(x_val: float, t_val: float) -> float: return d2u_dx2_func(x_val, t_val, cfg.ell)
def diff1x_v(x_val: float, t_val: float) -> float: return dv_dx_func(x_val, t_val, cfg.ell)
def diff2x_v(x_val: float, t_val: float) -> float: return d2v_dx2_func(x_val, t_val, cfg.ell)

# ======================================================
# INTEGRAL TERM: ENERGY-LIKE QUANTITY
# ======================================================

def integr_term(t_val: float) -> float:
    """
    Compute ∫₀^ell (∂u/∂x(x, t))² dx at fixed time t using adaptive quadrature.
    """
    result, *_ = unified_adaptive_quadrature(
        lambda x_val: diff1x_u(x_val, t_val) ** 2,
        cfg.ell
    )
    return result

# ======================================================
# RIGHT-HAND SIDE (RHS) OF THE PDE SYSTEM
# ======================================================

def f1(x: float, t: float) -> float:
    """
    Compute RHS of first PDE:
    f₁ = ∂²u/∂t² - (α + β ∫(∂u/∂x)² dx) * ∂²u/∂x² + a₁ ∂v/∂x
    """
    return (
        diff2t_u(x, t)
        - (cfg.alpha + cfg.beta * integr_term(t)) * diff2x_u(x, t)
        + cfg.a1 * diff1x_v(x, t)
    )


def f2(x: float, t: float) -> float:
    """
    Compute RHS of second PDE:
    f₂ = ∂²v/∂t² - γ ∂²v/∂x² + δ v - a₂ ∂u/∂x
    """
    return (
        diff2t_v(x, t)
        - cfg.gamma * diff2x_v(x, t)
        + cfg.delta * v(x, t)
        - cfg.a2 * diff1x_u(x, t)
    )