# ===============================================================
# MODULE: symbolic_basis_and_test_functions
# ---------------------------------------------------------------
# Purpose:
#   - Symbolic definitions of test functions, spatial derivatives,
#     and right-hand side (RHS) expressions used in benchmarking
#     Galerkin-based solvers for the Timoshenko beam equations.
#   - Produces lambdified (NumPy-compatible) functions for use in
#     numerical integration, plotting, and verification.
# ===============================================================

# ===============================================================
# IMPORTS
# ===============================================================
import sympy as sp               # Symbolic computation library (algebra, calculus)
import setting.config as cfg  # Configuration file with physical model constants

# ===============================================================
# USER-DEFINED FIXED PARAMETERS
# ===============================================================
lam = 1  # Spatial frequency mode (integer ≥ 1); adjustable for test case complexity

# ===============================================================
# SYMBOLIC VARIABLES
# ===============================================================
x, t = sp.symbols("x t", real=True, positive=True)  # Continuous spatial and temporal variables
ell = cfg.ell  # Domain length (beam length) loaded from configuration

# ===============================================================
# SYMBOLIC TEST FUNCTIONS
# ---------------------------------------------------------------
# These represent benchmark solution ansatz functions used in the
# Galerkin method and for generating known exact solutions.
# ===============================================================

# Zero-valued (null) test functions for initial boundary layer states
varphi0_sym = sp.S(0.0)
psi0_sym = sp.S(0.0)

# First non-zero symbolic test functions: sinusoidal spatial form
varphi1_sym = sp.sin(lam * sp.pi * x / ell)
psi1_sym = sp.sin(lam * sp.pi * x / ell)

# ===============================================================
# SPATIAL DERIVATIVES (1st to 3rd Order)
# ---------------------------------------------------------------
# Symbolic spatial derivatives of φ and ψ functions (used in weak forms)
# ===============================================================

# First derivatives
d1varphi0_sym = sp.diff(varphi0_sym, x)
d1psi0_sym = sp.diff(psi0_sym, x)
d1varphi1_sym = sp.diff(varphi1_sym, x)
d1psi1_sym = sp.diff(psi1_sym, x)

# Second derivatives
d2varphi0_sym = sp.diff(varphi0_sym, x, 2)
d2psi0_sym = sp.diff(psi0_sym, x, 2)

# Third derivatives
d3varphi0_sym = sp.diff(varphi0_sym, x, 3)
d3psi0_sym = sp.diff(psi0_sym, x, 3)

# ===============================================================
# RHS EXPRESSIONS f₁(x, t) AND f₂(x, t)
# ---------------------------------------------------------------
# These model external forcing for each PDE in the Timoshenko system.
# f₁ corresponds to displacement, f₂ to rotation.
# They are derived analytically based on exact solution profiles.
# ===============================================================

# f1(x, t): RHS of the displacement PDE (nonlinear in time via beta term)
f1_sym = (
    (lam**2 * sp.pi**2 / ell**2) * t *
    (cfg.alpha + cfg.beta * (lam**2 * sp.pi**2 / (2 * ell)) * t**2) *
    sp.sin(lam * sp.pi * x / ell)
    + cfg.a1 * (lam * sp.pi / ell) * t * sp.cos(lam * sp.pi * x / ell)
)

# f2(x, t): RHS of the rotation PDE
f2_sym = (
    (cfg.gamma * lam**2 * sp.pi**2 / ell**2 * t + cfg.delta) * t *
    sp.sin(lam * sp.pi * x / ell)
    - cfg.a2 * (lam * sp.pi / ell) * t * sp.cos(lam * sp.pi * x / ell)
)

# First-order spatial derivatives of RHS functions
d1f1_sym = sp.diff(f1_sym, x)
d1f2_sym = sp.diff(f2_sym, x)

# ===============================================================
# SYMBOLIC DERIVATIVE REGISTRY
# ---------------------------------------------------------------
# Collects all symbolic expressions in a centralized dictionary
# for reference and lambdification.
# ===============================================================
symbolic_derivatives = {
    # Base test functions
    'varphi0': varphi0_sym,
    'psi0': psi0_sym,
    'varphi1': varphi1_sym,
    'psi1': psi1_sym,

    # First-order spatial derivatives
    'd1varphi0': d1varphi0_sym,
    'd1psi0': d1psi0_sym,
    'd1varphi1': d1varphi1_sym,
    'd1psi1': d1psi1_sym,

    # Second-order spatial derivatives
    'd2varphi0': d2varphi0_sym,
    'd2psi0': d2psi0_sym,

    # Third-order spatial derivatives
    'd3varphi0': d3varphi0_sym,
    'd3psi0': d3psi0_sym,

    # RHS expressions and spatial derivatives
    'f1': f1_sym,
    'f2': f2_sym,
    'd1f1': d1f1_sym,
    'd1f2': d1f2_sym,
}

# ===============================================================
# NUMPY-COMPATIBLE LAMBDIFIED EXPRESSIONS
# ---------------------------------------------------------------
# Converts symbolic expressions into NumPy callables for use in
# numerical computations, integration, and visualization.
# ===============================================================

# Lambdify expressions that only depend on x
lambdified_derivatives = {
    name: sp.lambdify(x, expr, modules="numpy")
    for name, expr in symbolic_derivatives.items()
    if name not in {'f1', 'f2', 'd1f1', 'd1f2'}
}

# Lambdify time-dependent RHS expressions (x, t) → ℝ
lambdified_derivatives.update({
    'f1': sp.lambdify((x, t), f1_sym, modules="numpy"),
    'f2': sp.lambdify((x, t), f2_sym, modules="numpy"),
    'd1f1': sp.lambdify((x, t), d1f1_sym, modules="numpy"),
    'd1f2': sp.lambdify((x, t), d1f2_sym, modules="numpy"),
})