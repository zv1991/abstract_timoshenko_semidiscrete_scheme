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
    return t_sym * sp.sin(11 * sp.pi * x / ell_sym)


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
    return t_sym * sp.sin(2 * sp.pi * x / ell_sym)

# ======================================================
# PRECOMPUTED SYMBOLIC EXPRESSIONS
# ======================================================

# Evaluate test functions symbolically using globally defined variables
u_expr = u_sym(x, t, ell)  # u(x, t) symbolic expression
v_expr = v_sym(x, t, ell)  # v(x, t) symbolic expression