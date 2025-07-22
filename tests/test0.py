# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # Core scientific computing library: provides vectorization and numerical tools

from utils.auxiliary import integrate_derivative_form  
# Custom utility for computing spatial integrals like ∫ (∂u/∂x)² dx, needed in nonlinear terms

from tests.timoshenko_data import TimoshenkoTesterParent  
# Parent class that provides symbolic helpers and scaffolding for Timoshenko beam test cases


# ======================================================
# TEST CASE 0: ANALYTICAL SOLUTION FOR TIMOSHENKO SYSTEM
# ======================================================

class Testcase0(TimoshenkoTesterParent):
    """
    Symbolic benchmark test case for validating the nonlinear Timoshenko beam model.

    This case defines manufactured analytical solutions for:
        - Displacement field u(x, t)
        - Rotation field v(x, t)
    using separable polynomial basis functions.
    """

    # ------------------------------------------------------
    # CONSTRUCTOR: INITIALIZE WITH CONFIGURATION OBJECT
    # ------------------------------------------------------
    def __init__(self, cfg):
        """
        Constructor that sets up model parameters and initializes base class.

        Parameters
        ----------
        cfg : object
            Contains beam parameters and polynomial exponents:
            (tau, ell, alpha, beta, gamma, delta, a1, a2, m1_u, m2_u, m1_v, m2_v)
        """
        self.cfg = cfg
        self.name = "test0"          # Test name used by infrastructure or logger
        self.known_solutions = True  # Signals availability of analytical solution
        super().__init__(cfg)        # Inherit symbolic/numerical hooks from base
        self._prepare_data()         # Precompute or cache derived data

        # Polynomial degrees for u(x) and v(x)
        self.m1_u, self.m2_u = cfg.m1_u, cfg.m2_u
        self.m1_v, self.m2_v = cfg.m1_v, cfg.m2_v

    # ------------------------------------------------------
    # SPATIAL BASIS FOR DISPLACEMENT u(x)
    # ------------------------------------------------------
    def h_u(self, x): 
        """Polynomial shape for u(x): x^m1 * (ell - x)^m2"""
        return x**self.m1_u * (self.cfg.ell - x)**self.m2_u

    def d1h_u(self, x): 
        """First derivative of h_u(x) with product rule applied"""
        return (
            x**(self.m1_u - 1) * (self.cfg.ell - x)**(self.m2_u - 1) *
            (self.m1_u * self.cfg.ell - (self.m1_u + self.m2_u) * x)
        )

    def d2h_u(self, x): 
        """Second derivative of h_u(x), fully expanded"""
        term1 = self.m1_u * (self.m1_u - 1) * x**(self.m1_u - 2) * (self.cfg.ell - x)**self.m2_u
        term2 = -2 * self.m1_u * self.m2_u * x**(self.m1_u - 1) * (self.cfg.ell - x)**(self.m2_u - 1)
        term3 = self.m2_u * (self.m2_u - 1) * x**self.m1_u * (self.cfg.ell - x)**(self.m2_u - 2)
        return term1 + term2 + term3

    # ------------------------------------------------------
    # SPATIAL BASIS FOR ROTATION v(x)
    # ------------------------------------------------------
    def h_v(self, x): 
        """Polynomial shape for v(x): x^m1 * (ell - x)^m2"""
        return x**self.m1_v * (self.cfg.ell - x)**self.m2_v

    def d1h_v(self, x): 
        """First derivative of h_v(x) using product rule"""
        return (
            x**(self.m1_v - 1) * (self.cfg.ell - x)**(self.m2_v - 1) *
            (self.m1_v * self.cfg.ell - (self.m1_v + self.m2_v) * x)
        )

    def d2h_v(self, x): 
        """Second derivative of h_v(x), expanded fully"""
        term1 = self.m1_v * (self.m1_v - 1) * x**(self.m1_v - 2) * (self.cfg.ell - x)**self.m2_v
        term2 = -2 * self.m1_v * self.m2_v * x**(self.m1_v - 1) * (self.cfg.ell - x)**(self.m2_v - 1)
        term3 = self.m2_v * (self.m2_v - 1) * x**self.m1_v * (self.cfg.ell - x)**(self.m2_v - 2)
        return term1 + term2 + term3

    # ------------------------------------------------------
    # TEMPORAL BASIS FUNCTIONS FOR u(t)
    # ------------------------------------------------------
    def g_u(self, t): return t                    # Linear time-dependence for u
    def d1g_u(self, t): return np.float64(1.0)    # First time derivative ∂u/∂t = 1
    def d2g_u(self, t): return np.float64(0.0)    # Second time derivative ∂²u/∂t² = 0

    # ------------------------------------------------------
    # TEMPORAL BASIS FUNCTIONS FOR v(t)
    # ------------------------------------------------------
    def g_v(self, t): return t
    def d1g_v(self, t): return np.float64(1.0)
    def d2g_v(self, t): return np.float64(0.0)

    # ------------------------------------------------------
    # EXACT ANALYTICAL SOLUTIONS
    # ------------------------------------------------------
    def u(self, x, t): 
        """Exact displacement u(x, t) = h_u(x) * g_u(t)"""
        return self.h_u(x) * self.g_u(t)

    def v(self, x, t): 
        """Exact rotation v(x, t) = h_v(x) * g_v(t)"""
        return self.h_v(x) * self.g_v(t)

    # ------------------------------------------------------
    # TIME AND SPACE DERIVATIVES OF u(x, t)
    # ------------------------------------------------------
    def diff1t_u(self, x, t): return self.h_u(x) * self.d1g_u(t)
    def diff2t_u(self, x, t): return self.h_u(x) * self.d2g_u(t)
    def diff1x_u(self, x, t): return self.d1h_u(x) * self.g_u(t)
    def diff2x_u(self, x, t): return self.d2h_u(x) * self.g_u(t)

    # ------------------------------------------------------
    # TIME AND SPACE DERIVATIVES OF v(x, t)
    # ------------------------------------------------------
    def diff1t_v(self, x, t): return self.h_v(x) * self.d1g_v(t)
    def diff2t_v(self, x, t): return self.h_v(x) * self.d2g_v(t)
    def diff1x_v(self, x, t): return self.d1h_v(x) * self.g_v(t)
    def diff2x_v(self, x, t): return self.d2h_v(x) * self.g_v(t)

    # ------------------------------------------------------
    # NONLINEAR STIFFNESS TERM: ∫(∂u/∂x)² dx
    # ------------------------------------------------------
    def integr_term(self, t):
        """
        Compute nonlinear integral correction ∫ (∂u/∂x)² dx at time t.

        This contributes to damping/stiffness in the displacement equation.
        """
        integrand = lambda x: self.diff1x_u(x, t)  # ∂u/∂x evaluated at fixed t
        result, *_ = integrate_derivative_form(df=integrand, ell=self.cfg.ell)
        return result

    # ------------------------------------------------------
    # RIGHT-HAND SIDE FUNCTION FOR u-EQUATION (f₁)
    # ------------------------------------------------------
    def f1(self, x, t):
        """
        Construct RHS for u-equation in Timoshenko system:

        f₁(x, t) = ∂²u/∂t² - (α + β ∫(∂u/∂x)² dx) ∂²u/∂x² + a₁ ∂v/∂x
        """
        return (
            self.diff2t_u(x, t)
            - (self.cfg.alpha + self.cfg.beta * self.integr_term(t)) * self.diff2x_u(x, t)
            + self.cfg.a1 * self.diff1x_v(x, t)
        )

    # ------------------------------------------------------
    # RIGHT-HAND SIDE FUNCTION FOR v-EQUATION (f₂)
    # ------------------------------------------------------
    def f2(self, x, t):
        """
        Construct RHS for v-equation in Timoshenko system:

        f₂(x, t) = ∂²v/∂t² - γ ∂²v/∂x² + δ·v - a₂ ∂u/∂x
        """
        return (
            self.diff2t_v(x, t)
            - self.cfg.gamma * self.diff2x_v(x, t)
            + self.cfg.delta * self.v(x, t)
            - self.cfg.a2 * self.diff1x_u(x, t)
        )

    # ------------------------------------------------------
    # DATACLASS HOOK: POST-INIT ROUTINE
    # ------------------------------------------------------
    def __post_init__(self):
        """
        Hook for dataclass compatibility: ensures late initialization.
        """
        self._prepare_data()