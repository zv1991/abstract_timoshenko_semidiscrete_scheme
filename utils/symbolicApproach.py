import sympy as sp
import config as cfg  # Ensure cfg has attributes: ell, alpha, beta, a1, gamma, delta, a2
from scipy.integrate import quad
from typing import Tuple

# ------------------------------------------------------
# Symbolic Variable Declarations
# ------------------------------------------------------

# Declare global symbolic variables for symbolic computation
x, t, ell = sp.symbols('x t ell', real=True, positive=True)

# ------------------------------------------------------
# Galerkin Basis Function Definitions
# ------------------------------------------------------

def coeff_A_sym(m: int) -> sp.Expr:
    """Compute normalization constant A_m = 1 / sqrt(2m + 1)."""
    return 1 / sp.sqrt(2 * m + 1)

def shifted_legendre_sym(m: int, ell_sym: sp.Symbol, x_sym: sp.Symbol) -> sp.Expr:
    """
    Generate the m-th shifted Legendre polynomial over [0, ell].
    
    The domain shift: ξ = 2x/ell - 1 maps [0, ell] → [-1, 1].
    """
    xi = 2 * x_sym / ell_sym - 1  # Shift to [-1, 1]
    return sp.legendre(m, xi)

def phi_m_sym(m: int, ell_sym: sp.Symbol, x_sym: sp.Symbol) -> sp.Expr:
    """
    Compute the Galerkin basis function:
    φ_m(x) = (sqrt(ell)/2) * A_m * [P_{m+1}(x) - P_{m-1}(x)]
    
    Only defined for m ≥ 1.
    """
    if m < 1:
        raise ValueError("Basis index m must be >= 1.")
    A_m = coeff_A_sym(m)
    return (sp.sqrt(ell_sym) / 2) * A_m * (
        shifted_legendre_sym(m + 1, ell_sym, x_sym) - 
        shifted_legendre_sym(m - 1, ell_sym, x_sym)
    )

# ------------------------------------------------------
# Symbolic Analytical Test Functions
# ------------------------------------------------------

def u_sym(x_sym: sp.Symbol, t_sym: sp.Symbol, ell_sym: sp.Symbol) -> sp.Expr:
    """Test function: u(x, t) = t * φ₁(x)."""
    return t_sym * phi_m_sym(1, ell_sym, x_sym)

def v_sym(x_sym: sp.Symbol, t_sym: sp.Symbol, ell_sym: sp.Symbol) -> sp.Expr:
    """Test function: v(x, t) = t * φ₁(x)."""
    return t_sym * phi_m_sym(1, ell_sym, x_sym)

# ------------------------------------------------------
# Precomputed Symbolic Expressions
# ------------------------------------------------------

u_expr = u_sym(x, t, ell)
v_expr = v_sym(x, t, ell)

# ------------------------------------------------------
# Symbolic Derivatives
# ------------------------------------------------------

# Time derivatives
def diff1t_u_sym() -> sp.Expr: return sp.diff(u_expr, t)
def diff2t_u_sym() -> sp.Expr: return sp.diff(u_expr, t, 2)
def diff1t_v_sym() -> sp.Expr: return sp.diff(v_expr, t)
def diff2t_v_sym() -> sp.Expr: return sp.diff(v_expr, t, 2)

# Space derivatives
def diff1x_u_sym() -> sp.Expr: return sp.diff(u_expr, x)
def diff2x_u_sym() -> sp.Expr: return sp.diff(u_expr, x, 2)
def diff1x_v_sym() -> sp.Expr: return sp.diff(v_expr, x)
def diff2x_v_sym() -> sp.Expr: return sp.diff(v_expr, x, 2)

# ------------------------------------------------------
# Numerical Functions (Lambdified for Evaluation)
# ------------------------------------------------------

# Main functions
u_func = sp.lambdify((x, t, ell), u_expr, modules="numpy")
v_func = sp.lambdify((x, t, ell), v_expr, modules="numpy")

# First and second time derivatives
du_dt_func   = sp.lambdify((x, t, ell), diff1t_u_sym(), modules="numpy")
d2u_dt2_func = sp.lambdify((x, t, ell), diff2t_u_sym(), modules="numpy")
dv_dt_func   = sp.lambdify((x, t, ell), diff1t_v_sym(), modules="numpy")
d2v_dt2_func = sp.lambdify((x, t, ell), diff2t_v_sym(), modules="numpy")

# First and second spatial derivatives
du_dx_func   = sp.lambdify((x, t, ell), diff1x_u_sym(), modules="numpy")
d2u_dx2_func = sp.lambdify((x, t, ell), diff2x_u_sym(), modules="numpy")
dv_dx_func   = sp.lambdify((x, t, ell), diff1x_v_sym(), modules="numpy")
d2v_dx2_func = sp.lambdify((x, t, ell), diff2x_v_sym(), modules="numpy")

# ------------------------------------------------------
# Public Numerical Evaluation Interface (with cfg.ell)
# ------------------------------------------------------

# u, v evaluation
def u(x_val: float, t_val: float) -> float: return u_func(x_val, t_val, cfg.ell)
def v(x_val: float, t_val: float) -> float: return v_func(x_val, t_val, cfg.ell)

# Time derivatives
def diff1t_u(x_val: float, t_val: float) -> float: return du_dt_func(x_val, t_val, cfg.ell)
def diff2t_u(x_val: float, t_val: float) -> float: return d2u_dt2_func(x_val, t_val, cfg.ell)
def diff1t_v(x_val: float, t_val: float) -> float: return dv_dt_func(x_val, t_val, cfg.ell)
def diff2t_v(x_val: float, t_val: float) -> float: return d2v_dt2_func(x_val, t_val, cfg.ell)

# Space derivatives
def diff1x_u(x_val: float, t_val: float) -> float: return du_dx_func(x_val, t_val, cfg.ell)
def diff2x_u(x_val: float, t_val: float) -> float: return d2u_dx2_func(x_val, t_val, cfg.ell)
def diff1x_v(x_val: float, t_val: float) -> float: return dv_dx_func(x_val, t_val, cfg.ell)
def diff2x_v(x_val: float, t_val: float) -> float: return d2v_dx2_func(x_val, t_val, cfg.ell)

# ------------------------------------------------------
# Integral: ∫₀^ell [∂u/∂x(x, t)]² dx
# ------------------------------------------------------

def integrand(x_val: float, t_val: float) -> float:
    """Function to integrate: (∂u/∂x)² at given x and t."""
    return diff1x_u(x_val, t_val) ** 2

def compute_integral_du_dx_squared(t_val: float, tol: float = 1e-6, max_iter: int = 1000) -> Tuple[float, float]:
    """
    Compute integral ∫₀^ell [∂u/∂x(x, t)]² dx using adaptive quadrature.

    Returns:
        - result: Integral approximation.
        - error: Estimated absolute error.

    Raises:
        RuntimeError: If integration error exceeds tolerance.
    """
    result, error = quad(
        integrand, 0, cfg.ell, args=(t_val,), epsabs=tol, limit=max_iter
    )

    if error > tol:
        raise RuntimeError(
            f"Integration failed: error {error:.2e} > tol {tol:.2e} (limit={max_iter})."
        )

    return result, error

def integr_term(t_val: float) -> float:
    """Helper to get just the integral value (∂u/∂x)²."""
    result, _ = compute_integral_du_dx_squared(t_val)
    return result

# ------------------------------------------------------
# Right-hand Side (RHS) Functions for PDE System
# ------------------------------------------------------

def f1(x: float, t: float) -> float:
    """
    RHS of the first PDE:
    f₁ = ∂²u/∂t² - (α + β * ∫(∂u/∂x)² dx) * ∂²u/∂x² + a₁ * ∂v/∂x
    """
    return (
        diff2t_u(x, t) 
        - (cfg.alpha + cfg.beta * integr_term(t)) * diff2x_u(x, t) 
        + cfg.a1 * diff1x_v(x, t)
    )

def f2(x: float, t: float) -> float:
    """
    RHS of the second PDE:
    f₂ = ∂²v/∂t² - γ * ∂²v/∂x² + δ * v - a₂ * ∂u/∂x
    """
    return (
        diff2t_v(x, t) 
        - cfg.gamma * diff2x_v(x, t) 
        + cfg.delta * v(x, t) 
        - cfg.a2 * diff1x_u(x, t)
    )