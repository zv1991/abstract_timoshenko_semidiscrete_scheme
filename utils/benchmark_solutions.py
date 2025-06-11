# ======================================================
# MODULE IMPORTS
# ======================================================

import sympy as sp  # Symbolic math library for algebraic expressions

# ======================================================
# SYMBOLIC VARIABLE DECLARATIONS
# ======================================================

# Define symbolic variables for position (x), time (t), and domain length (ell)
x, t, ell = sp.symbols('x t ell', real=True, positive=True)

# ======================================================
# GALERKIN BASIS FUNCTION DEFINITIONS
# ======================================================

def coeff_A_sym(m: int) -> sp.Expr:
    """
    Compute normalization coefficient for orthonormal Galerkin basis.

    A_m = 1 / sqrt(2m + 1)
    
    Args:
        m (int): Basis index
    
    Returns:
        sp.Expr: Symbolic expression for A_m
    """
    return 1 / sp.sqrt(2 * m + 1)


def shifted_legendre_sym(m: int, ell_sym: sp.Symbol, x_sym: sp.Symbol) -> sp.Expr:
    """
    Compute the m-th shifted Legendre polynomial evaluated at ξ.

    The change of variable is:
    ξ = (2 * x / ell) - 1, mapping x ∈ [0, ell] to ξ ∈ [-1, 1]

    Args:
        m (int): Degree of Legendre polynomial
        ell_sym (sp.Symbol): Domain length symbol
        x_sym (sp.Symbol): Position symbol

    Returns:
        sp.Expr: Shifted Legendre polynomial P_m(ξ)
    """
    xi = 2 * x_sym / ell_sym - 1  # Affine mapping to standard interval [-1, 1]
    return sp.legendre(m, xi)


def phi_m_sym(m: int, ell_sym: sp.Symbol, x_sym: sp.Symbol) -> sp.Expr:
    """
    Define the Galerkin basis function φ_m(x) using Legendre polynomials.

    φ_m(x) = (√ell / 2) * A_m * [P_{m+1}(ξ) - P_{m-1}(ξ)]

    Args:
        m (int): Index of the Galerkin basis function (must be ≥ 1)
        ell_sym (sp.Symbol): Domain length symbol
        x_sym (sp.Symbol): Position symbol

    Returns:
        sp.Expr: Symbolic Galerkin basis function φ_m(x)

    Raises:
        ValueError: If m < 1
    """
    if m < 1:
        raise ValueError("Basis index m must be ≥ 1.")

    A_m = coeff_A_sym(m)  # Normalization factor
    return (sp.sqrt(ell_sym) / 2) * A_m * (
        shifted_legendre_sym(m + 1, ell_sym, x_sym) -
        shifted_legendre_sym(m - 1, ell_sym, x_sym)
    )

# ======================================================
# SYMBOLIC ANALYTICAL TEST FUNCTIONS
# ======================================================

def u_sym(x_sym: sp.Symbol, t_sym: sp.Symbol, ell_sym: sp.Symbol) -> sp.Expr:
    """
    Define symbolic test function u(x, t) = t * φ₁(x)

    Args:
        x_sym (sp.Symbol): Position symbol
        t_sym (sp.Symbol): Time symbol
        ell_sym (sp.Symbol): Domain length symbol

    Returns:
        sp.Expr: Symbolic expression for u(x, t)
    """
    return t_sym * phi_m_sym(1, ell_sym, x_sym)


def v_sym(x_sym: sp.Symbol, t_sym: sp.Symbol, ell_sym: sp.Symbol) -> sp.Expr:
    """
    Define symbolic test function v(x, t) = t * φ₁(x)

    Args:
        x_sym (sp.Symbol): Position symbol
        t_sym (sp.Symbol): Time symbol
        ell_sym (sp.Symbol): Domain length symbol

    Returns:
        sp.Expr: Symbolic expression for v(x, t)
    """
    return t_sym * phi_m_sym(1, ell_sym, x_sym)

# ======================================================
# PRECOMPUTED SYMBOLIC EXPRESSIONS
# ======================================================

# Evaluate test functions symbolically using globally defined variables
u_expr = u_sym(x, t, ell)  # u(x, t) symbolic expression
v_expr = v_sym(x, t, ell)  # v(x, t) symbolic expression