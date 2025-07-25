# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # Core scientific computing library: provides vectorization and numerical tools

import utils.auxiliary as aux
# Custom utility module: basis functions, Legendre polynomials, normalization, and projection support

from tests.timoshenko_data import TimoshenkoTesterParent  
# Parent class that provides symbolic/numerical hooks for testing Timoshenko beam models


# ======================================================
# TEST CASE 1: ANALYTICAL SOLUTION FOR TIMOSHENKO SYSTEM
# ======================================================

class Testcase1(TimoshenkoTesterParent):
    """
    Symbolic benchmark test case for validating the nonlinear Timoshenko beam model.

    Defines separable analytical solutions for:
        - Displacement field u(x, t)
        - Rotation field v(x, t)
    using polynomial Galerkin basis functions in space and linear time.
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
            (tau, ell, alpha, beta, gamma, delta, a1, a2, m_u, m_v)
        """
        self.cfg = cfg
        self.name = "test1"          # Name used for test selection/logging
        self.known_solutions = True  # Flag indicating exact solution availability
        
        super().__init__(cfg)        # Inherit symbolic utilities from parent class
        self._prepare_data()         # Precompute symbolic expressions (source terms, BCs, etc.)

        self.m_u = cfg.m_u           # Degree of spatial basis for u(x)
        self.m_v = cfg.m_v           # Degree of spatial basis for v(x)

    # ------------------------------------------------------
    # SPATIAL BASIS FUNCTION φₘ(x) FOR u(x)
    # ------------------------------------------------------
    def h_u(self, x):
        """
        Spatial shape function for displacement field u(x, t).
        φₘ(x) = (√ℓ / 2) * Aₘ * [Pₘ₊₁(x) - Pₘ₋₁(x)]
        """
        return aux.phi_m(self.m_u, self.cfg.ell, x)

    def d1h_u(self, x):
        """
        First derivative ∂φₘ/∂x for u(x).
        This is the normalized shifted Legendre polynomial: P̂ₘ(x)
        """
        return aux.normalized_shifted_legendre(self.m_u, self.cfg.ell, x)

    # ------------------------------------------------------
    # SPATIAL BASIS FUNCTION φₘ(x) FOR v(x)
    # ------------------------------------------------------
    def h_v(self, x):
        """
        Spatial shape function for rotation field v(x, t).
        φₘ(x) = (√ℓ / 2) * Aₘ * [Pₘ₊₁(x) - Pₘ₋₁(x)]
        """
        return aux.phi_m(self.m_v, self.cfg.ell, x)

    def d1h_v(self, x):
        """
        First derivative ∂φₘ/∂x for v(x).
        This is the normalized shifted Legendre polynomial: P̂ₘ(x)
        """
        return aux.normalized_shifted_legendre(self.m_v, self.cfg.ell, x)

    # ------------------------------------------------------
    # TEMPORAL BASIS FUNCTIONS FOR u(t) AND v(t)
    # ------------------------------------------------------
    def g_u(self, t):
        """Temporal basis function for u(t); here simply g_u(t) = t"""
        return t

    def d2g_u(self, t):
        """Second derivative of g_u(t); returns zero since it's linear"""
        return np.float64(0.0)

    def g_v(self, t):
        """Temporal basis function for v(t); here simply g_v(t) = t"""
        return t

    def d2g_v(self, t):
        """Second derivative of g_v(t); returns zero since it's linear"""
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
        """First spatial derivative of u: ∂u/∂x"""
        return self.d1h_u(x) * self.g_u(t)

    def diff2t_u(self, x, t):
        """Second temporal derivative of u: ∂²u/∂t²"""
        return self.h_u(x) * self.d2g_u(t)

    # ------------------------------------------------------
    # DERIVATIVES OF v(x, t)
    # ------------------------------------------------------
    def diff1x_v(self, x, t):
        """First spatial derivative of v: ∂v/∂x"""
        return self.d1h_v(x) * self.g_v(t)

    def diff2t_v(self, x, t):
        """Second temporal derivative of v: ∂²v/∂t²"""
        return self.h_v(x) * self.d2g_v(t)

    # ------------------------------------------------------
    # NONLINEAR STIFFNESS TERM FOR u-EQUATION
    # ------------------------------------------------------
    def nonlinear_term(self, t):
        """
        Nonlinear stiffness term in the displacement equation:

            q(t) = α + β ∫ (∂u/∂x)² dx

        Since u(x, t) = h_u(x) · g_u(t), and ∂u/∂x = P̂ₘ(x) · g_u(t),
        we have:

            ∫ (∂u/∂x)² dx = g_u(t)² · ∫ P̂ₘ(x)² dx

        The basis P̂ₘ(x) is orthonormal over [0, ℓ], so the integral evaluates to 1.
        Therefore:

            q(t) = α + β · g_u(t)²
        """
        return self.cfg.alpha + self.cfg.beta * self.g_u(t)**2

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