# ==============================================================
# Module: symbolic_basis_and_test_functions
# --------------------------------------------------------------
# This module defines symbolic Galerkin basis functions, test
# functions, derivatives, and zero-valued RHS expressions for use
# in analytical verification and benchmarking of numerical solvers.
#
# Features:
# - Legendre-based Galerkin basis functions: φₘ(x)
# - Symbolic test functions: u(x, t), v(x, t)
# - Derivatives up to third order (spatial)
# - RHS expressions f₁(x, t), f₂(x, t) and their spatial derivatives
# - NumPy-compatible lambdified expressions for numerical evaluation
# ==============================================================

# ==============================================================
# MODULE IMPORTS
# ==============================================================

import sympy as sp                     # Symbolic computation engine
from functools import partial          # For fixing symbolic parameters (e.g., ell) in lambdified functions
import utils.config as cfg             # Central configuration, e.g., fixed spatial domain length (cfg.ell)

# ==============================================================
# SYMBOLIC VARIABLES
# ==============================================================

# Declare global symbolic variables (real and positive)
x, t, ell = sp.symbols('x t ell', real=True, positive=True)

# ==============================================================
# GALERKIN BASIS FUNCTION DEFINITIONS
# ==============================================================

def coeff_A_sym(m: int) -> sp.Expr:
    """
    Compute normalization factor Aₘ = 1 / sqrt(2m + 1)
    for orthonormality of Galerkin basis functions.
    """
    return 1 / sp.sqrt(2 * m + 1)

def shifted_legendre_sym(m: int, ell_sym: sp.Symbol, x_sym: sp.Symbol) -> sp.Expr:
    """
    Compute shifted Legendre polynomial Pₘ(ξ) with ξ mapped from x ∈ [0, ell] → ξ ∈ [-1, 1].

    Parameters:
        m       : Degree of Legendre polynomial
        ell_sym : Symbolic domain length
        x_sym   : Symbolic spatial coordinate

    Returns:
        sp.Expr : Shifted Legendre polynomial Pₘ(ξ)
    """
    ξ = (2 * x_sym / ell_sym) - 1
    return sp.legendre(m, ξ)

def phi_m_sym(m: int, ell_sym: sp.Symbol, x_sym: sp.Symbol) -> sp.Expr:
    """
    Define Galerkin basis function φₘ(x) using shifted Legendre polynomial differences.

    Formula:
        φₘ(x) = (√ell / 2) * Aₘ * [Pₘ₊₁(ξ) - Pₘ₋₁(ξ)]

    Parameters:
        m       : Basis index (must be ≥ 1)
        ell_sym : Symbolic domain length
        x_sym   : Symbolic spatial variable

    Returns:
        sp.Expr : Galerkin basis function φₘ(x)
    """
    if m < 1:
        raise ValueError("Basis index m must be ≥ 1.")
    A_m = coeff_A_sym(m)
    return (sp.sqrt(ell_sym) / 2) * A_m * (
        shifted_legendre_sym(m + 1, ell_sym, x_sym) -
        shifted_legendre_sym(m - 1, ell_sym, x_sym)
    )

# ==============================================================
# SYMBOLIC TEST FUNCTIONS
# ==============================================================

def u_sym(x_sym: sp.Symbol, t_sym: sp.Symbol, ell_sym: sp.Symbol) -> sp.Expr:
    """Symbolic test function: u(x, t) = t · φ₁(x)"""
    return t_sym * phi_m_sym(1, ell_sym, x_sym)

def v_sym(x_sym: sp.Symbol, t_sym: sp.Symbol, ell_sym: sp.Symbol) -> sp.Expr:
    """Symbolic test function: v(x, t) = t · φ₁(x)"""
    return t_sym * phi_m_sym(1, ell_sym, x_sym)

# Precomputed symbolic test expressions using global variables
u_expr = u_sym(x, t, ell)
v_expr = v_sym(x, t, ell)

# ==============================================================
# COMPONENT FUNCTIONS AND DERIVATIVES
# ==============================================================

# Zero-valued base test functions
varphi0_sym = sp.S(0.0)
psi0_sym    = sp.S(0.0)

# First-order test functions (using φ₁)
varphi1_sym = phi_m_sym(1, ell, x)
psi1_sym    = phi_m_sym(1, ell, x)

# First-order spatial derivatives
d1varphi0_sym = sp.diff(varphi0_sym, x)
d1psi0_sym    = sp.diff(psi0_sym, x)
d1varphi1_sym = sp.diff(varphi1_sym, x)
d1psi1_sym    = sp.diff(psi1_sym, x)

# Second-order spatial derivatives (non-zero only if base functions are non-constant)
d2varphi0_sym = sp.diff(varphi0_sym, x, 2)
d2psi0_sym    = sp.diff(psi0_sym, x, 2)

# Third-order spatial derivatives (likewise for completeness)
d3varphi0_sym = sp.diff(varphi0_sym, x, 3)
d3psi0_sym    = sp.diff(psi0_sym, x, 3)

# ==============================================================
# RHS TERMS (Zero-valued for simplified verification)
# ==============================================================

f1_sym = sp.S(0.0)
f2_sym = sp.S(0.0)
d1f1_sym = sp.diff(f1_sym, x)
d1f2_sym = sp.diff(f2_sym, x)

# ==============================================================
# SYMBOLIC DERIVATIVE REGISTRY (central dictionary)
# ==============================================================

symbolic_derivatives = {
    # Test functions
    'varphi0': varphi0_sym,
    'psi0': psi0_sym,
    'varphi1': varphi1_sym,
    'psi1': psi1_sym,

    # First derivatives
    'd1varphi0': d1varphi0_sym,
    'd1psi0': d1psi0_sym,
    'd1varphi1': d1varphi1_sym,
    'd1psi1': d1psi1_sym,

    # Second derivatives
    'd2varphi0': d2varphi0_sym,
    'd2psi0': d2psi0_sym,

    # Third derivatives
    'd3varphi0': d3varphi0_sym,
    'd3psi0': d3psi0_sym,

    # RHS terms and their first derivatives
    'f1': f1_sym,
    'f2': f2_sym,
    'd1f1': d1f1_sym,
    'd1f2': d1f2_sym,
}

# ==============================================================
# NUMPY-COMPATIBLE LAMBDIFIED EXPRESSIONS
# ==============================================================

# For space-only functions: lambdify over (x, ell), then fix ell via cfg
lambdified_derivatives = {
    name: partial(sp.lambdify((x, ell), expr, modules="numpy"), ell=cfg.ell)
    for name, expr in symbolic_derivatives.items()
    if name not in {'f1', 'f2', 'd1f1', 'd1f2'}  # These depend on time
}

# Time-dependent RHS expressions (x, t) → ℝ
lambdified_derivatives.update({
    'f1':   sp.lambdify((x, t), f1_sym, modules="numpy"),
    'f2':   sp.lambdify((x, t), f2_sym, modules="numpy"),
    'd1f1': sp.lambdify((x, t), d1f1_sym, modules="numpy"),
    'd1f2': sp.lambdify((x, t), d1f2_sym, modules="numpy"),
})