# ======================================================
# MODULE IMPORTS
# ======================================================

import sympy as sp                               # Symbolic computation library
import utils.config as cfg                       # Global configuration parameters (e.g., ell, alpha, beta)
from utils.auxiliary import unified_adaptive_quadrature  # Custom quadrature routine
from utils.benchmark_sines import x, t, ell, u_expr, v_expr  # Symbolic variables and benchmark functions

# ======================================================
# SYMBOLIC DERIVATIVE CLASS
# ======================================================

class SymbolicDerivatives:
    """
    Encapsulates symbolic derivatives of benchmark solutions,
    their numerical evaluation using lambdify, and PDE right-hand sides.
    """

    # ======================================================
    # SYMBOLIC DERIVATIVES (EXACT FOR u AND v)
    # ======================================================

    # --- Time derivatives of u ---
    @staticmethod
    def diff1t_u_sym(): return sp.diff(u_expr, t)
    @staticmethod
    def diff2t_u_sym(): return sp.diff(u_expr, t, 2)

    # --- Time derivatives of v ---
    @staticmethod
    def diff1t_v_sym(): return sp.diff(v_expr, t)
    @staticmethod
    def diff2t_v_sym(): return sp.diff(v_expr, t, 2)

    # --- Spatial derivatives of u ---
    @staticmethod
    def diff1x_u_sym(): return sp.diff(u_expr, x)
    @staticmethod
    def diff2x_u_sym(): return sp.diff(u_expr, x, 2)

    # --- Spatial derivatives of v ---
    @staticmethod
    def diff1x_v_sym(): return sp.diff(v_expr, x)
    @staticmethod
    def diff2x_v_sym(): return sp.diff(v_expr, x, 2)

    # ======================================================
    # LAMBDIFIED FUNCTIONS FOR NUMERICAL EVALUATION
    # ======================================================

    # --- Base functions ---
    u_func = sp.lambdify((x, t, ell), u_expr, modules="numpy")
    v_func = sp.lambdify((x, t, ell), v_expr, modules="numpy")

    # --- Lambdified time derivatives ---
    du_dt_func   = sp.lambdify((x, t, ell), diff1t_u_sym.__func__(), modules="numpy")
    d2u_dt2_func = sp.lambdify((x, t, ell), diff2t_u_sym.__func__(), modules="numpy")
    dv_dt_func   = sp.lambdify((x, t, ell), diff1t_v_sym.__func__(), modules="numpy")
    d2v_dt2_func = sp.lambdify((x, t, ell), diff2t_v_sym.__func__(), modules="numpy")

    # --- Lambdified spatial derivatives ---
    du_dx_func   = sp.lambdify((x, t, ell), diff1x_u_sym.__func__(), modules="numpy")
    d2u_dx2_func = sp.lambdify((x, t, ell), diff2x_u_sym.__func__(), modules="numpy")
    dv_dx_func   = sp.lambdify((x, t, ell), diff1x_v_sym.__func__(), modules="numpy")
    d2v_dx2_func = sp.lambdify((x, t, ell), diff2x_v_sym.__func__(), modules="numpy")

    # ======================================================
    # NUMERICAL INTERFACES (WITH FIXED cfg.ell)
    # ======================================================

    # --- Base evaluations ---
    @staticmethod
    def u(x_val: float, t_val: float) -> float:
        """Evaluate u(x, t) numerically."""
        return SymbolicDerivatives.u_func(x_val, t_val, cfg.ell)

    @staticmethod
    def v(x_val: float, t_val: float) -> float:
        """Evaluate v(x, t) numerically."""
        return SymbolicDerivatives.v_func(x_val, t_val, cfg.ell)

    # --- Time derivatives ---
    @staticmethod
    def diff1t_u(x_val: float, t_val: float) -> float:
        """Evaluate ∂u/∂t numerically."""
        return SymbolicDerivatives.du_dt_func(x_val, t_val, cfg.ell)

    @staticmethod
    def diff2t_u(x_val: float, t_val: float) -> float:
        """Evaluate ∂²u/∂t² numerically."""
        return SymbolicDerivatives.d2u_dt2_func(x_val, t_val, cfg.ell)

    @staticmethod
    def diff1t_v(x_val: float, t_val: float) -> float:
        """Evaluate ∂v/∂t numerically."""
        return SymbolicDerivatives.dv_dt_func(x_val, t_val, cfg.ell)

    @staticmethod
    def diff2t_v(x_val: float, t_val: float) -> float:
        """Evaluate ∂²v/∂t² numerically."""
        return SymbolicDerivatives.d2v_dt2_func(x_val, t_val, cfg.ell)

    # --- Spatial derivatives ---
    @staticmethod
    def diff1x_u(x_val: float, t_val: float) -> float:
        """Evaluate ∂u/∂x numerically."""
        return SymbolicDerivatives.du_dx_func(x_val, t_val, cfg.ell)

    @staticmethod
    def diff2x_u(x_val: float, t_val: float) -> float:
        """Evaluate ∂²u/∂x² numerically."""
        return SymbolicDerivatives.d2u_dx2_func(x_val, t_val, cfg.ell)

    @staticmethod
    def diff1x_v(x_val: float, t_val: float) -> float:
        """Evaluate ∂v/∂x numerically."""
        return SymbolicDerivatives.dv_dx_func(x_val, t_val, cfg.ell)

    @staticmethod
    def diff2x_v(x_val: float, t_val: float) -> float:
        """Evaluate ∂²v/∂x² numerically."""
        return SymbolicDerivatives.d2v_dx2_func(x_val, t_val, cfg.ell)

    # ======================================================
    # ENERGY-LIKE INTEGRAL TERM
    # ======================================================

    @staticmethod
    def integr_term(t_val: float) -> float:
        """
        Evaluate energy-like integral:
        ∫₀^ell (∂u/∂x(x, t))² dx using adaptive quadrature.

        Args:
            t_val (float): Time value

        Returns:
            float: Numerical result of the integral
        """
        result, *_ = unified_adaptive_quadrature(
            lambda x_val: SymbolicDerivatives.diff1x_u(x_val, t_val) ** 2,
            cfg.ell
        )
        return result

    # ======================================================
    # PDE RIGHT-HAND SIDES (RHS)
    # ======================================================

    @staticmethod
    def f1(x: float, t: float) -> float:
        """
        Evaluate RHS for u-equation:
        f₁ = ∂²u/∂t² - (α + β * ∫ (∂u/∂x)² dx) * ∂²u/∂x² + a₁ * ∂v/∂x
        """
        return (
            SymbolicDerivatives.diff2t_u(x, t)
            - (cfg.alpha + cfg.beta * SymbolicDerivatives.integr_term(t)) *
              SymbolicDerivatives.diff2x_u(x, t)
            + cfg.a1 * SymbolicDerivatives.diff1x_v(x, t)
        )

    @staticmethod
    def f2(x: float, t: float) -> float:
        """
        Evaluate RHS for v-equation:
        f₂ = ∂²v/∂t² - γ * ∂²v/∂x² + δ * v - a₂ * ∂u/∂x
        """
        return (
            SymbolicDerivatives.diff2t_v(x, t)
            - cfg.gamma * SymbolicDerivatives.diff2x_v(x, t)
            + cfg.delta * SymbolicDerivatives.v(x, t)
            - cfg.a2 * SymbolicDerivatives.diff1x_u(x, t)
        )