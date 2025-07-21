# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # Numerical library for arrays and mathematical operations
from utils.auxiliary import integrate_derivative_form  # Computes ∫(∂u/∂x)² dx — used for nonlinear stiffness
from tests.timoshenko_data import TimoshenkoTesterParent  # Base class handling symbolic/numeric init


# ======================================================
# TEST CASE 1: ANALYTICAL SOLUTION FOR TIMOSHENKO SYSTEM
# ======================================================

class Testcase0(TimoshenkoTesterParent):
    """
    Symbolic benchmark test case for the nonlinear Timoshenko beam model.

    This test case provides exact analytical expressions for:
        - Displacement field u(x, t)
        - Rotation field v(x, t)

    These are constructed via separable spatial and temporal basis functions,
    allowing validation of the solver's accuracy.
    """

    # ------------------------------------------------------
    # CONSTRUCTOR WITH CONFIGURATION INJECTION
    # ------------------------------------------------------
    def __init__(self, cfg):
        """
        Initialize the symbolic benchmark with externally provided configuration.

        Parameters
        ----------
        cfg : object or dict-like
            Beam configuration with parameters:
            (tau, ell, alpha, beta, gamma, delta, a1, a2)
        """
        self.cfg = cfg
        self.name = "test0"          # Define solver name here
        self.known_solutions = True  # Activates symbolic solution mode
        super().__init__(cfg)        # Pass config to base class
        self._prepare_data()         # Initialize derived fields

    # ------------------------------------------------------
    # SPATIAL BASIS FUNCTIONS FOR DISPLACEMENT u(x)
    # ------------------------------------------------------
    def h_u(self, x): return x**2 * (self.cfg.ell - x)  # Polynomial shape for u
    def d1h_u(self, x): return x * (2 * self.cfg.ell - 3 * x)  # First derivative
    def d2h_u(self, x): return 2 * (self.cfg.ell - 3 * x)      # Second derivative

    # ------------------------------------------------------
    # SPATIAL BASIS FUNCTIONS FOR ROTATION v(x)
    # ------------------------------------------------------
    def h_v(self, x): return x * (self.cfg.ell - x)**2
    def d1h_v(self, x): return (self.cfg.ell - x) * (self.cfg.ell - 3 * x)
    def d2h_v(self, x): return 2 * (3 * x - 2 * self.cfg.ell)

    # ------------------------------------------------------
    # TEMPORAL BASIS FUNCTIONS FOR DISPLACEMENT u(t)
    # ------------------------------------------------------
    def g_u(self, t): return t  # Linear in time
    def d1g_u(self, t): return np.float64(0.0)  # ∂g_u/∂t = 0
    def d2g_u(self, t): return np.float64(0.0)  # ∂²g_u/∂t² = 0

    # ------------------------------------------------------
    # TEMPORAL BASIS FUNCTIONS FOR ROTATION v(t)
    # ------------------------------------------------------
    def g_v(self, t): return t
    def d1g_v(self, t): return np.float64(0.0)
    def d2g_v(self, t): return np.float64(0.0)

    # ------------------------------------------------------
    # EXACT SOLUTION: u(x, t) AND v(x, t)
    # ------------------------------------------------------
    def u(self, x, t): return self.h_u(x) * self.g_u(t)
    def v(self, x, t): return self.h_v(x) * self.g_v(t)

    # ------------------------------------------------------
    # DERIVATIVES OF u(x, t)
    # ------------------------------------------------------
    def diff1t_u(self, x, t): return self.h_u(x) * self.d1g_u(t)
    def diff2t_u(self, x, t): return self.h_u(x) * self.d2g_u(t)
    def diff1x_u(self, x, t): return self.d1h_u(x) * self.g_u(t)
    def diff2x_u(self, x, t): return self.d2h_u(x) * self.g_u(t)

    # ------------------------------------------------------
    # DERIVATIVES OF v(x, t)
    # ------------------------------------------------------
    def diff1t_v(self, x, t): return self.h_v(x) * self.d1g_v(t)
    def diff2t_v(self, x, t): return self.h_v(x) * self.d2g_v(t)
    def diff1x_v(self, x, t): return self.d1h_v(x) * self.g_v(t)
    def diff2x_v(self, x, t): return self.d2h_v(x) * self.g_v(t)

    # ------------------------------------------------------
    # NONLINEAR INTEGRAL TERM: ∫(∂u/∂x)² dx
    # ------------------------------------------------------
    def integr_term(self, t):
        """
        Computes the nonlinear integral ∫ (∂u/∂x)² dx used in the f₁ equation.

        Parameters
        ----------
        t : float
            Time value at which spatial derivative is evaluated

        Returns
        -------
        float
            Value of nonlinear correction term at time t
        """
        integrand = lambda x: self.diff1x_u(x, t)
        result, *_ = integrate_derivative_form(df=integrand, ell=self.cfg.ell)
        return result

    # ------------------------------------------------------
    # RIGHT-HAND SIDE: DISPLACEMENT EQUATION (f₁)
    # ------------------------------------------------------
    def f1(self, x, t):
        """
        Right-hand side of the displacement PDE:

        f₁(x, t) = ∂²u/∂t² - (α + β ∫(∂u/∂x)² dx) ∂²u/∂x² + a₁ ∂v/∂x

        Returns
        -------
        float
            Evaluated source term f₁ at (x, t)
        """
        return (
            self.diff2t_u(x, t)
            - (self.cfg.alpha + self.cfg.beta * self.integr_term(t)) * self.diff2x_u(x, t)
            + self.cfg.a1 * self.diff1x_v(x, t)
        )

    # ------------------------------------------------------
    # RIGHT-HAND SIDE: ROTATION EQUATION (f₂)
    # ------------------------------------------------------
    def f2(self, x, t):
        """
        Right-hand side of the rotation PDE:

        f₂(x, t) = ∂²v/∂t² - γ ∂²v/∂x² + δ v - a₂ ∂u/∂x

        Returns
        -------
        float
            Evaluated source term f₂ at (x, t)
        """
        return (
            self.diff2t_v(x, t)
            - self.cfg.gamma * self.diff2x_v(x, t)
            + self.cfg.delta * self.v(x, t)
            - self.cfg.a2 * self.diff1x_u(x, t)
        )

    # ------------------------------------------------------
    # OPTIONAL: POST INIT HOOK FOR DATACLASS COMPATIBILITY
    # ------------------------------------------------------
    def __post_init__(self):
        """
        Enables compatibility with `dataclasses` via delayed preparation.
        """
        self._prepare_data()