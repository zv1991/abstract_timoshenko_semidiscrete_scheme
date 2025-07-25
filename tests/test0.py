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
        
        # ------------------------------------------------------
        # BASE INITIALIZATION AND DATA PREPARATION
        # ------------------------------------------------------
        super().__init__(cfg)        # Inherit symbolic/numerical hooks from base
        self._prepare_data()         # Precompute or cache derived data (e.g., u0, du0, f1, etc.)

        # Polynomial degrees for u(x) and v(x)
        self.m1_u, self.m2_u = cfg.m1_u, cfg.m2_u
        self.m1_v, self.m2_v = cfg.m1_v, cfg.m2_v

    # ------------------------------------------------------
    # SPATIAL BASIS FUNCTIONS FOR u(x)
    # ------------------------------------------------------
    def h_u(self, x):
        """Spatial profile h_u(x) = x^m1_u * (ℓ - x)^m2_u"""
        return x**self.m1_u * (self.cfg.ell - x)**self.m2_u

    def d1h_u(self, x):
        """First spatial derivative ∂h_u/∂x computed via product rule"""
        return (
            x**(self.m1_u - 1) * (self.cfg.ell - x)**(self.m2_u - 1) *
            (self.m1_u * self.cfg.ell - (self.m1_u + self.m2_u) * x)
        )

    # ------------------------------------------------------
    # SPATIAL BASIS FUNCTIONS FOR v(x)
    # ------------------------------------------------------
    def h_v(self, x):
        """Spatial profile h_v(x) = x^m1_v * (ℓ - x)^m2_v"""
        return x**self.m1_v * (self.cfg.ell - x)**self.m2_v

    def d1h_v(self, x):
        """First spatial derivative ∂h_v/∂x computed via product rule"""
        return (
            x**(self.m1_v - 1) * (self.cfg.ell - x)**(self.m2_v - 1) *
            (self.m1_v * self.cfg.ell - (self.m1_v + self.m2_v) * x)
        )

    # ------------------------------------------------------
    # TEMPORAL BASIS FUNCTIONS FOR u(t) AND v(t)
    # ------------------------------------------------------
    def g_u(self, t):
        """Temporal basis for u(t); linear function in time"""
        return t

    def d2g_u(self, t):
        """Second temporal derivative of g_u(t); returns 0 for linear time"""
        return np.float64(0.0)

    def g_v(self, t):
        """Temporal basis for v(t); linear function in time"""
        return t

    def d2g_v(self, t):
        """Second temporal derivative of g_v(t); returns 0 for linear time"""
        return np.float64(0.0)

    # ------------------------------------------------------
    # EXACT SOLUTIONS FOR u(x, t) AND v(x, t)
    # ------------------------------------------------------
    def u(self, x, t):
        """Exact displacement solution: u(x, t) = h_u(x) * g_u(t)"""
        return self.h_u(x) * self.g_u(t)

    def v(self, x, t):
        """Exact rotation solution: v(x, t) = h_v(x) * g_v(t)"""
        return self.h_v(x) * self.g_v(t)

    # ------------------------------------------------------
    # DERIVATIVES OF u(x, t)
    # ------------------------------------------------------
    def diff1x_u(self, x, t):
        """First spatial derivative of u(x, t): ∂u/∂x"""
        return self.d1h_u(x) * self.g_u(t)

    def diff2t_u(self, x, t):
        """Second temporal derivative of u(x, t): ∂²u/∂t²"""
        return self.h_u(x) * self.d2g_u(t)

    # ------------------------------------------------------
    # DERIVATIVES OF v(x, t)
    # ------------------------------------------------------
    def diff1x_v(self, x, t):
        """First spatial derivative of v(x, t): ∂v/∂x"""
        return self.d1h_v(x) * self.g_v(t)

    def diff2t_v(self, x, t):
        """Second temporal derivative of v(x, t): ∂²v/∂t²"""
        return self.h_v(x) * self.d2g_v(t)

    # ------------------------------------------------------
    # NONLINEAR STIFFNESS TERM α + β ∫(∂u/∂x)² dx
    # ------------------------------------------------------
    def nonlinear_term(self, t):
        """
        Compute nonlinear term: α + β ∫(∂u/∂x)² dx
        This is used in the u-equation to model geometric nonlinearities.
        """
        integrand = lambda x: self.diff1x_u(x, t)  # Function to integrate: (∂u/∂x)
        result, *_ = integrate_derivative_form(
            df=integrand,
            ell=self.cfg.ell,
            form='squared'  # Indicates (∂u/∂x)² integration
        )
        return self.cfg.alpha + self.cfg.beta * result

    # ------------------------------------------------------
    # SOURCE TERMS FOR WEAK FORMULATIONS OF u AND v
    # ------------------------------------------------------
    def source_terms(self):
        """
        Return the functional components of the source terms f₁(x, t) and f₂(x, t)
        appearing in the weak formulation of the Timoshenko beam equations.

        Each returned component corresponds to a specific term in the weak form
        that must be projected against the test functions φₘ (and their gradients P̂ₘ).

        These callable components are necessary for computing time-dependent
        Galerkin inner products of the form:
            (term(x, t), φₘ)       for mass-type terms
            (term(x, t), P̂ₘ)       for stiffness/coupling terms

        -------------------------------------------------------------------
        f₁(x, t): Source term for the u-equation (displacement field)
        -------------------------------------------------------------------
        Strong form:
            f₁(x, t) = ∂²u/∂t² 
                     - (α + β ∫(∂u/∂x)² dx) ∂²u/∂x² 
                     + a₁ ∂v/∂x

        Weak form (after integration by parts):
            (f₁, φₘ) = (∂²u/∂t², φₘ)
                     + (α + β ∫(∂u/∂x)² dx) (∂u/∂x, P̂ₘ)
                     - a₁ (v, P̂ₘ)

        Required components:
            1. ∂²u/∂t²                      — acceleration term for φₘ projection
            2. (α + β ∫(∂u/∂x)² dx) ∂u/∂x   — nonlinear term for stiffness projection
            3. a₁ v(x, t)                   — coupling term, projected against P̂ₘ

        -------------------------------------------------------------------
        f₂(x, t): Source term for the v-equation (rotation field)
        -------------------------------------------------------------------
        Strong form:
            f₂(x, t) = ∂²v/∂t² 
                     - γ ∂²v/∂x² 
                     + δ v 
                     - a₂ ∂u/∂x

        Weak form (after integration by parts):
            (f₂, φₘ) = (∂²v/∂t², φₘ)
                     + γ (∂v/∂x, P̂ₘ)
                     + δ (v, φₘ)
                     + a₂ (u, P̂ₘ)

        Required components:
            1. ∂²v/∂t²    — acceleration term for φₘ projection
            2. γ ∂v/∂x    — stiffness term for P̂ₘ projection
            3. δ v(x, t)  — mass-like term for φₘ projection
            4. a₂ u(x, t) — coupling term, projected against P̂ₘ

        Returns
        -------
        dict of str -> list of callable
            {
                "f1": [∂²u/∂t², (α + β ∫(∂u/∂x)² dx) ∂u/∂x, a₁ v(x, t)],
                "f2": [∂²v/∂t², γ ∂v/∂x, δ v(x, t), a₂ u(x, t)]
            }
        """
        return {
            "f1": [
                lambda x, t: self.diff2t_u(x, t),                          # Term 1: ∂²u/∂t² for (•, φₘ)
                lambda x, t: self.nonlinear_term(t) * self.diff1x_u(x, t), # Term 2: nonlinear term for (•, P̂ₘ)
                lambda x, t: self.cfg.a1 * self.v(x, t)                    # Term 3: a₁ v(x, t) for (•, P̂ₘ)
            ],
            "f2": [
                lambda x, t: self.diff2t_v(x, t),                         # Term 1: ∂²v/∂t² for (•, φₘ)
                lambda x, t: self.cfg.gamma * self.diff1x_v(x, t),        # Term 2: γ ∂v/∂x for (•, P̂ₘ)
                lambda x, t: self.cfg.delta * self.v(x, t),               # Term 3: δ v(x, t) for (•, φₘ)
                lambda x, t: self.cfg.a2 * self.u(x, t)                   # Term 4: a₂ u(x, t) for (•, P̂ₘ)
            ]
        }

    # ------------------------------------------------------
    # POST-INIT HOOK FOR LATE INITIALIZATION
    # ------------------------------------------------------
    def __post_init__(self):
        """
        Hook for dataclass compatibility: ensures proper late-stage initialization.
        Called after dataclass fields are set (if used in dataclass context).
        """
        self._prepare_data()