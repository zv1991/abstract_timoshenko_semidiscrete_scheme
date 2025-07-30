# ======================================================
# MODULE IMPORTS
# ======================================================

import utils.auxiliary as aux
# Custom utility module:
# - Provides Legendre-based spatial basis functions (phi_m)
# - Offers normalized polynomial derivatives and projection utilities

from tests.timoshenko_data import TimoshenkoTesterParent
# Base class offering symbolic framework for constructing exact PDE solutions
# - Provides PDE structure
# - Handles storage, symbolic pre-processing, and solution formatting


# ======================================================
# CLASS: Testcase5 – Polynomial-Time and Galerkin-Space Analytical Solution
# ======================================================

class Testcase5(TimoshenkoTesterParent):
    """
    Symbolic benchmark solution for the nonlinear Timoshenko beam model.

    Constructs u(x, t) and v(x, t) as separable products:
        u(x, t) = h_u(x) * g_u(t)
        v(x, t) = h_v(x) * g_v(t)

    - Spatial basis h_u, h_v derived from shifted Legendre polynomials.
    - Time basis g_u, g_v is polynomial: t^p scaled by coefficients.

    Purpose:
    - Used to verify numerical accuracy of solvers via Method of Manufactured Solutions (MMS).
    """

    # ------------------------------------------------------
    # METHOD: __init__ – Initialize Configuration Parameters
    # ------------------------------------------------------
    def __init__(self, cfg):
        """
        Initializes benchmark problem using provided configuration.

        Parameters
        ----------
        cfg : object
            Configuration with:
            - Model coefficients: alpha, beta, gamma, delta, a1, a2
            - Domain parameters: ell, tau
            - Polynomial degrees: m_u, m_v, pow_u, pow_v
            - Coefficients: coeff_u, coeff_v
        """
        self.cfg = cfg
        self.name = "test5"
        self.known_solutions = True  # Enable L2 error comparison

        super().__init__(cfg)        # Initialize parent symbolic utilities
        self._prepare_data()         # Generate symbolic expressions (ICs, BCs, forcing)

        # Spatial basis mode degrees for u and v
        self.m_u = cfg.m_u
        self.m_v = cfg.m_v

        # Time polynomial powers for u(t) and v(t)
        self.pow_u = cfg.pow_u
        self.pow_v = cfg.pow_v

        # Time polynomial coefficients
        self.coeff_u = cfg.coeff_u
        self.coeff_v = cfg.coeff_v

    # ------------------------------------------------------
    # METHOD: h_u, d1h_u – Spatial Basis for Displacement
    # ------------------------------------------------------
    def h_u(self, x):
        """Spatial basis function for u(x): φₘ(x) using shifted Legendre form."""
        return aux.phi_m(self.m_u, self.cfg.ell, x)

    def d1h_u(self, x):
        """First spatial derivative ∂φₘ/∂x used in ∂u/∂x calculations."""
        return aux.normalized_shifted_legendre(self.m_u, self.cfg.ell, x)

    # ------------------------------------------------------
    # METHOD: h_v, d1h_v – Spatial Basis for Rotation
    # ------------------------------------------------------
    def h_v(self, x):
        """Spatial basis function for v(x): φₘ(x) using shifted Legendre form."""
        return aux.phi_m(self.m_v, self.cfg.ell, x)

    def d1h_v(self, x):
        """First spatial derivative ∂φₘ/∂x used in ∂v/∂x calculations."""
        return aux.normalized_shifted_legendre(self.m_v, self.cfg.ell, x)

    # ------------------------------------------------------
    # METHOD: g_u, d2g_u – Time Component for Displacement
    # ------------------------------------------------------
    def g_u(self, t):
        """Time multiplier for displacement u(x, t): coeff_u * t^pow_u"""
        return self.coeff_u * t**self.pow_u

    def d2g_u(self, t):
        """Second derivative ∂²g_u/∂t² needed for ∂²u/∂t²."""
        return self.coeff_u * self.pow_u * (self.pow_u - 1) * t**(self.pow_u - 2)

    # ------------------------------------------------------
    # METHOD: g_v, d2g_v – Time Component for Rotation
    # ------------------------------------------------------
    def g_v(self, t):
        """Time multiplier for rotation v(x, t): coeff_v * t^pow_v"""
        return self.coeff_v * t**self.pow_v

    def d2g_v(self, t):
        """Second derivative ∂²g_v/∂t² needed for ∂²v/∂t²."""
        return self.coeff_v * self.pow_v * (self.pow_v - 1) * t**(self.pow_v - 2)

    # ------------------------------------------------------
    # METHOD: u, v – Exact Displacement and Rotation
    # ------------------------------------------------------
    def u(self, x, t):
        """Exact displacement field: u(x, t) = h_u(x) * g_u(t)"""
        return self.h_u(x) * self.g_u(t)

    def v(self, x, t):
        """Exact rotation field: v(x, t) = h_v(x) * g_v(t)"""
        return self.h_v(x) * self.g_v(t)

    # ------------------------------------------------------
    # METHOD: diff1x_u, diff2t_u – Derivatives of u(x, t)
    # ------------------------------------------------------
    def diff1x_u(self, x, t):
        """∂u/∂x = ∂φₘ/∂x * g_u(t)"""
        return self.d1h_u(x) * self.g_u(t)

    def diff2t_u(self, x, t):
        """∂²u/∂t² = φₘ(x) * ∂²g_u/∂t²"""
        return self.h_u(x) * self.d2g_u(t)

    # ------------------------------------------------------
    # METHOD: diff1x_v, diff2t_v – Derivatives of v(x, t)
    # ------------------------------------------------------
    def diff1x_v(self, x, t):
        """∂v/∂x = ∂φₘ/∂x * g_v(t)"""
        return self.d1h_v(x) * self.g_v(t)

    def diff2t_v(self, x, t):
        """∂²v/∂t² = φₘ(x) * ∂²g_v/∂t²"""
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