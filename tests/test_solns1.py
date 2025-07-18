# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # Numerical library for arrays and mathematical operations
import setting.config as cfg  # Global configuration for beam parameters (τ, ℓ, α, β, γ, δ, a₁, a₂)
from utils.auxiliary import integrate_derivative_form  # Computes ∫(∂u/∂x)² dx — used for nonlinear stiffness
from tests.timoshenko_data import TimoshenkoTesterParent  # Abstract base class that handles data preparation


# ======================================================
# TEST CASE 1: KNOWN ANALYTICAL SOLUTION OF TIMOSHENKO SYSTEM
# ======================================================

class Testcase1(TimoshenkoTesterParent):
    """
    Symbolic benchmark test case for the nonlinear Timoshenko beam model.

    This test case uses exact analytical expressions for:
        - Displacement field u(x, t)
        - Rotation field v(x, t)
    These are based on separable temporal and spatial basis functions.
    """

    # ------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------
    def __init__(self):
        """
        Constructor initializes symbolic test case.
        Sets `known_solutions = True` to use exact formulas,
        then prepares the derived data fields.
        """
        self.known_solutions = True
        super().__init__()
        self._prepare_data()

    # ------------------------------------------------------
    # SPATIAL BASIS FUNCTIONS FOR u(x) AND v(x)
    # ------------------------------------------------------
    def h_u(self, x): return x**2 * (cfg.ell - x)  # Basis shape for displacement u
    def d1h_u(self, x): return x * (2 * cfg.ell - 3 * x)  # ∂h_u/∂x
    def d2h_u(self, x): return 2 * (cfg.ell - 3 * x)  # ∂²h_u/∂x²

    def h_v(self, x): return x * (cfg.ell - x)**2  # Basis shape for rotation v
    def d1h_v(self, x): return (cfg.ell - x) * (cfg.ell - 3 * x)  # ∂h_v/∂x
    def d2h_v(self, x): return 2 * (3 * x - 2 * cfg.ell)  # ∂²h_v/∂x²

    # ------------------------------------------------------
    # TEMPORAL BASIS FUNCTIONS FOR u(t) AND v(t)
    # ------------------------------------------------------
    def g_u(self, t): return t  # Linear growth in time
    def d1g_u(self, t): return np.float64(0.0)  # ∂g_u/∂t = 0
    def d2g_u(self, t): return np.float64(0.0)  # ∂²g_u/∂t² = 0

    def g_v(self, t): return t  # Linear growth in time
    def d1g_v(self, t): return np.float64(0.0)  # ∂g_v/∂t = 0
    def d2g_v(self, t): return np.float64(0.0)  # ∂²g_v/∂t² = 0

    # ------------------------------------------------------
    # EXACT SOLUTIONS: u(x, t) AND v(x, t)
    # ------------------------------------------------------
    def u(self, x, t): return self.h_u(x) * self.g_u(t)  # u(x,t) = h_u(x) * g_u(t)
    def v(self, x, t): return self.h_v(x) * self.g_v(t)  # v(x,t) = h_v(x) * g_v(t)

    # ------------------------------------------------------
    # DERIVATIVES OF u(x, t)
    # ------------------------------------------------------
    def diff1t_u(self, x, t): return self.h_u(x) * self.d1g_u(t)  # ∂u/∂t
    def diff2t_u(self, x, t): return self.h_u(x) * self.d2g_u(t)  # ∂²u/∂t²
    def diff1x_u(self, x, t): return self.d1h_u(x) * self.g_u(t)  # ∂u/∂x
    def diff2x_u(self, x, t): return self.d2h_u(x) * self.g_u(t)  # ∂²u/∂x²

    # ------------------------------------------------------
    # DERIVATIVES OF v(x, t)
    # ------------------------------------------------------
    def diff1t_v(self, x, t): return self.h_v(x) * self.d1g_v(t)  # ∂v/∂t
    def diff2t_v(self, x, t): return self.h_v(x) * self.d2g_v(t)  # ∂²v/∂t²
    def diff1x_v(self, x, t): return self.d1h_v(x) * self.g_v(t)  # ∂v/∂x
    def diff2x_v(self, x, t): return self.d2h_v(x) * self.g_v(t)  # ∂²v/∂x²

    # ------------------------------------------------------
    # NONLINEAR STIFFNESS TERM
    # ------------------------------------------------------
    def integr_term(self, t):
        """
        Computes the nonlinear correction ∫ (∂u/∂x)² dx at time t.

        Parameters:
        -----------
        t : float
            Time value for evaluating spatial derivative

        Returns:
        --------
        float
            Nonlinear integral used in f₁
        """
        integrand = lambda x: self.diff1x_u(x, t)
        result, *_ = integrate_derivative_form(df=integrand, ell=cfg.ell)
        return result

    # ------------------------------------------------------
    # RIGHT-HAND SIDE FOR DISPLACEMENT EQUATION (f₁)
    # ------------------------------------------------------
    def f1(self, x, t):
        """
        Right-hand side for displacement PDE:

        f₁(x, t) = ∂²u/∂t² - (α + β ∫(∂u/∂x)² dx) ∂²u/∂x² + a₁ ∂v/∂x

        Returns:
        --------
        float
            Evaluated source term f₁(x, t)
        """
        return (
            self.diff2t_u(x, t)
            - (cfg.alpha + cfg.beta * self.integr_term(t)) * self.diff2x_u(x, t)
            + cfg.a1 * self.diff1x_v(x, t)
        )

    # ------------------------------------------------------
    # RIGHT-HAND SIDE FOR ROTATION EQUATION (f₂)
    # ------------------------------------------------------
    def f2(self, x, t):
        """
        Right-hand side for rotation PDE:

        f₂(x, t) = ∂²v/∂t² - γ ∂²v/∂x² + δ v - a₂ ∂u/∂x

        Returns:
        --------
        float
            Evaluated source term f₂(x, t)
        """
        return (
            self.diff2t_v(x, t)
            - cfg.gamma * self.diff2x_v(x, t)
            + cfg.delta * self.v(x, t)
            - cfg.a2 * self.diff1x_u(x, t)
        )

    # ------------------------------------------------------
    # (OPTIONAL) POST-INITIALIZATION HOOK
    # ------------------------------------------------------
    def __post_init__(self):
        """
        Optional post-initialization method in case class is used with dataclasses.
        Not used in typical direct instantiation.
        """
        self._prepare_data()