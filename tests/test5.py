# ======================================================
# MODULE IMPORTS
# ======================================================

import utils.auxiliary as aux
# Custom utility module:
# Provides spatial basis functions (Legendre), normalization, and projection utilities

from tests.timoshenko_data import TimoshenkoTesterParent
# Base class that integrates symbolic and numerical behavior
# for constructing analytical benchmark solutions to Timoshenko beam equations


# ======================================================
# CLASS: Testcase5 – Polynomial-Time and Galerkin-Space Analytical Solution
# ======================================================

class Testcase5(TimoshenkoTesterParent):
    """
    Analytical benchmark for the nonlinear Timoshenko beam system.

    The exact solution is constructed as a separable product:
        - u(x, t) = h_u(x) * g_u(t)
        - v(x, t) = h_v(x) * g_v(t)

    Spatial basis h_u, h_v: Derived from shifted Legendre polynomials (Galerkin projection)
    Temporal basis g_u, g_v: Powers of t (polynomial in time)

    Purpose: Validate numerical solvers against this exact known solution.
    """

    # ------------------------------------------------------
    # METHOD: __init__ – Initialize Configuration Parameters
    # ------------------------------------------------------
    def __init__(self, cfg):
        """
        Constructor to initialize the benchmark case with a configuration object.

        Parameters
        ----------
        cfg : object
            Contains model coefficients and spatial/temporal basis degrees:
            - tau, ell, alpha, beta, gamma, delta, a1, a2
            - m_u, m_v: Legendre polynomial mode indices for u, v
            - pow_u, pow_v: Polynomial power for time dependence
        """
        self.cfg = cfg
        self.name = "test5"
        self.known_solutions = True  # Indicates this test provides exact symbolic solutions

        super().__init__(cfg)        # Parent class sets symbolic defaults and storage
        self._prepare_data()         # Precompute symbolic source terms, BCs, and ICs

        self.m_u = cfg.m_u           # Degree of φ_m(x) for displacement
        self.m_v = cfg.m_v           # Degree of φ_m(x) for rotation

        self.pow_u = cfg.pow_u       # Exponent for temporal function g_u(t)
        self.pow_v = cfg.pow_v       # Exponent for temporal function g_v(t)

    # ------------------------------------------------------
    # METHOD: h_u – Spatial Basis for Displacement
    # ------------------------------------------------------
    def h_u(self, x):
        """
        φₘ(x) basis for u(x), constructed via Legendre polynomial combination.
        """
        return aux.phi_m(self.m_u, self.cfg.ell, x)

    def d1h_u(self, x):
        """
        First derivative of h_u(x) with respect to x.
        Evaluates normalized shifted Legendre polynomial: P̂ₘ(x)
        """
        return aux.normalized_shifted_legendre(self.m_u, self.cfg.ell, x)

    # ------------------------------------------------------
    # METHOD: h_v – Spatial Basis for Rotation
    # ------------------------------------------------------
    def h_v(self, x):
        """
        φₘ(x) basis for v(x), same structure as h_u but with m_v degree.
        """
        return aux.phi_m(self.m_v, self.cfg.ell, x)

    def d1h_v(self, x):
        """
        First derivative of h_v(x) with respect to x.
        """
        return aux.normalized_shifted_legendre(self.m_v, self.cfg.ell, x)

    # ------------------------------------------------------
    # METHOD: g_u, d2g_u – Temporal Basis for Displacement
    # ------------------------------------------------------
    def g_u(self, t):
        """Time dependence for u(t) as a monomial."""
        return t**self.pow_u

    def d2g_u(self, t):
        """Second time derivative of g_u(t) = d²g_u/dt²."""
        return self.pow_u * (self.pow_u - 1) * t**(self.pow_u - 2)

    # ------------------------------------------------------
    # METHOD: g_v, d2g_v – Temporal Basis for Rotation
    # ------------------------------------------------------
    def g_v(self, t):
        """Time dependence for v(t) as a monomial."""
        return t**self.pow_v

    def d2g_v(self, t):
        """Second time derivative of g_v(t) = d²g_v/dt²."""
        return self.pow_v * (self.pow_v - 1) * t**(self.pow_v - 2)

    # ------------------------------------------------------
    # METHOD: u, v – Exact Analytical Solutions
    # ------------------------------------------------------
    def u(self, x, t):
        """Exact analytical displacement: u(x, t) = h_u(x) * g_u(t)"""
        return self.h_u(x) * self.g_u(t)

    def v(self, x, t):
        """Exact analytical rotation: v(x, t) = h_v(x) * g_v(t)"""
        return self.h_v(x) * self.g_v(t)

    # ------------------------------------------------------
    # METHOD: diff1x_u, diff2t_u – Derivatives of u(x, t)
    # ------------------------------------------------------
    def diff1x_u(self, x, t):
        """First spatial derivative of u(x, t): ∂u/∂x"""
        return self.d1h_u(x) * self.g_u(t)

    def diff2t_u(self, x, t):
        """Second time derivative of u(x, t): ∂²u/∂t²"""
        return self.h_u(x) * self.d2g_u(t)

    # ------------------------------------------------------
    # METHOD: diff1x_v, diff2t_v – Derivatives of v(x, t)
    # ------------------------------------------------------
    def diff1x_v(self, x, t):
        """First spatial derivative of v(x, t): ∂v/∂x"""
        return self.d1h_v(x) * self.g_v(t)

    def diff2t_v(self, x, t):
        """Second time derivative of v(x, t): ∂²v/∂t²"""
        return self.h_v(x) * self.d2g_v(t)

    # ------------------------------------------------------
    # METHOD: nonlinear_term – Nonlinear q(t) in Displacement Equation
    # ------------------------------------------------------
    def nonlinear_term(self, t):
        """
        Computes nonlinear stiffness term in displacement PDE:

            q(t) = α + β ∫ (∂u/∂x)² dx

        Since:
            ∂u/∂x = P̂ₘ(x) * g_u(t)
            and ∫ P̂ₘ(x)² dx = 1 (orthonormal basis),
        it simplifies to:

            q(t) = α + β * g_u(t)²
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