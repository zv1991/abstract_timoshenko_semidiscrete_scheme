# ======================================================
# MODULE IMPORTS
# ======================================================

import sympy as sp                                 # Symbolic math engine for analytical computation
import utils.config as cfg                         # Configuration module: contains fixed parameters like ell
from utils.benchmark_solutions import x, t, ell, phi_m_sym  # Common symbolic variables and Galerkin basis function
from functools import partial                      # Allows partial function application (used for fixing ell)


# ======================================================
# SYMBOLIC TEST FUNCTIONS
# ======================================================

# Zero-valued base test functions
varphi0_sym = sp.S(0)
psi0_sym    = sp.S(0)

# First-order test functions using the Galerkin basis φ₁(x)
varphi1_sym = phi_m_sym(1, ell, x)
psi1_sym    = phi_m_sym(1, ell, x)

# First-order spatial derivatives
d1varphi0_sym = sp.diff(varphi0_sym, x)
d1psi0_sym    = sp.diff(psi0_sym, x)
d1varphi1_sym = sp.diff(varphi1_sym, x)
d1psi1_sym    = sp.diff(psi1_sym, x)

# Second-order spatial derivatives
d2varphi0_sym = sp.diff(varphi0_sym, x, 2)
d2psi0_sym    = sp.diff(psi0_sym, x, 2)

# Third-order spatial derivatives
d3varphi0_sym = sp.diff(varphi0_sym, x, 3)
d3psi0_sym    = sp.diff(psi0_sym, x, 3)


# ======================================================
# ZERO RIGHT-HAND SIDE DEFINITIONS
# ======================================================

# Symbolic expressions representing RHS = 0 for testing
f1_sym = sp.S(0)
f2_sym = sp.S(0)

# First-order spatial derivatives of RHS
d1f1_sym = sp.diff(f1_sym, x)
d1f2_sym = sp.diff(f2_sym, x)


# ======================================================
# SYMBOLIC DERIVATIVE REGISTRY
# ======================================================

# Registry to organize all symbolic expressions
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

    # Second derivatives (only for varphi0, psi0)
    'd2varphi0': d2varphi0_sym,
    'd2psi0': d2psi0_sym,

    # Third derivatives
    'd3varphi0': d3varphi0_sym,
    'd3psi0': d3psi0_sym,

    # RHS terms and their derivatives
    'f1': f1_sym,
    'f2': f2_sym,
    'd1f1': d1f1_sym,
    'd1f2': d1f2_sym,
}


# ======================================================
# LAMBDIFY AND PARTIAL APPLICATION OF FIXED PARAMETERS
# ======================================================

# Convert x-ell symbolic functions to numerical functions with ell fixed from cfg
lambdified_derivatives = {
    name: partial(sp.lambdify((x, ell), expr, modules="numpy"), ell=cfg.ell)
    for name, expr in symbolic_derivatives.items()
    if name not in ['f1', 'f2', 'd1f1', 'd1f2']  # These are handled separately due to time-dependence
}

# Lambdify time-dependent RHS terms with arguments (x, t)
lambdified_derivatives['f1']   = sp.lambdify((x, t), f1_sym, modules="numpy")
lambdified_derivatives['f2']   = sp.lambdify((x, t), f2_sym, modules="numpy")
lambdified_derivatives['d1f1'] = sp.lambdify((x, t), d1f1_sym, modules="numpy")
lambdified_derivatives['d1f2'] = sp.lambdify((x, t), d1f2_sym, modules="numpy")