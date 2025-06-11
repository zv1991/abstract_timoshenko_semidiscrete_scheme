import sympy as sp

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