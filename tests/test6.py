# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # Numerical computing library: arrays, mathematical functions, exp, sin, cos
from tests.timoshenko_data import TimoshenkoTesterParent  # Base class providing symbolic PDE test framework


# ======================================================
# CLASS: Testcase6 – Trigonometric Benchmark for Timoshenko System
# ======================================================

class Testcase6(TimoshenkoTesterParent):
    """
    Symbolic benchmark test case for the nonlinear Timoshenko beam model.

    Constructs an exact (manufactured) solution using:
    - Trigonometric spatial modes (sinusoidal)
    - Exponential-in-time temporal dependence

    Exact solution:
        u(x, t) = scaling₁ · sin(λ_u·π·x/ℓ) · exp(c₁·π·t)
        v(x, t) = scaling₂ · sin(λ_v·π·x/ℓ) · exp(c₂·π·t)

    Purpose:
        Validates numerical solvers via the Method of Manufactured Solutions (MMS).
    """

    # ------------------------------------------------------
    # CONSTRUCTOR: Initialize Configuration and Parameters
    # ------------------------------------------------------
    def __init__(self, cfg):
        """
        Initialize the benchmark test case using configuration parameters.

        Parameters
        ----------
        cfg : object
            Contains:
            - Physical coefficients: alpha, beta, gamma, delta, a1, a2
            - Domain and time step: ell, tau
            - Spatial frequencies: lam_u, lam_v
            - Time growth parameters: pow_coeff_*, mult_coeff_*
        """
        self.cfg = cfg
        self.name = "test6"
        self.known_solutions = True  # Indicates the test provides exact symbolic reference

        super().__init__(cfg)        # Initialize inherited symbolic machinery
        self._prepare_data()         # Precompute forcing terms and initial/boundary conditions

        # Spatial sine frequencies
        self.lam_u = cfg.lam_u
        self.lam_v = cfg.lam_v

        # Temporal exponential growth/decay rates and scaling
        self.pow_coeff_u = cfg.pow_coeff_u
        self.pow_coeff_v = cfg.pow_coeff_v
        self.mult_coeff_u = cfg.mult_coeff_u
        self.mult_coeff_v = cfg.mult_coeff_v

    # ------------------------------------------------------
    # METHOD: h_u – Spatial Basis for Displacement u(x)
    # ------------------------------------------------------
    def h_u(self, x):
        """Trigonometric spatial mode for displacement u(x)."""
        return np.sin(self.lam_u * np.pi * x / self.cfg.ell)

    def d1h_u(self, x):
        """First spatial derivative ∂h_u/∂x of displacement mode."""
        return (self.lam_u * np.pi / self.cfg.ell) * np.cos(self.lam_u * np.pi * x / self.cfg.ell)

    # ------------------------------------------------------
    # METHOD: h_v – Spatial Basis for Rotation v(x)
    # ------------------------------------------------------
    def h_v(self, x):
        """Trigonometric spatial mode for rotation v(x)."""
        return np.sin(self.lam_v * np.pi * x / self.cfg.ell)

    def d1h_v(self, x):
        """First spatial derivative ∂h_v/∂x of rotation mode."""
        return (self.lam_v * np.pi / self.cfg.ell) * np.cos(self.lam_v * np.pi * x / self.cfg.ell)

    # ------------------------------------------------------
    # METHOD: g_u – Temporal Function for Displacement u(t)
    # ------------------------------------------------------
    def g_u(self, t):
        """Time component of u(t): exponential time growth with scaling."""
        return self.mult_coeff_u * np.exp(self.pow_coeff_u * np.pi * t)

    def d2g_u(self, t):
        """Second time derivative ∂²g_u/∂t² for displacement dynamics."""
        return self.mult_coeff_u * (self.pow_coeff_u * np.pi)**2 * np.exp(self.pow_coeff_u * np.pi * t)

    # ------------------------------------------------------
    # METHOD: g_v – Temporal Function for Rotation v(t)
    # ------------------------------------------------------
    def g_v(self, t):
        """Time component of v(t): exponential time growth with scaling."""
        return self.mult_coeff_v * np.exp(self.pow_coeff_v * np.pi * t)

    def d2g_v(self, t):
        """Second time derivative ∂²g_v/∂t² for rotation dynamics."""
        return self.mult_coeff_v * (self.pow_coeff_v * np.pi)**2 * np.exp(self.pow_coeff_v * np.pi * t)

    # ------------------------------------------------------
    # METHOD: u – Exact Displacement Solution u(x, t)
    # ------------------------------------------------------
    def u(self, x, t):
        """Exact displacement field: u(x, t) = h_u(x) · g_u(t)"""
        return self.h_u(x) * self.g_u(t)

    # ------------------------------------------------------
    # METHOD: v – Exact Rotation Solution v(x, t)
    # ------------------------------------------------------
    def v(self, x, t):
        """Exact rotation field: v(x, t) = h_v(x) · g_v(t)"""
        return self.h_v(x) * self.g_v(t)

    # ------------------------------------------------------
    # METHOD: diff1x_u – First Spatial Derivative of u
    # ------------------------------------------------------
    def diff1x_u(self, x, t):
        """Compute ∂u/∂x = ∂h_u/∂x · g_u(t)"""
        return self.d1h_u(x) * self.g_u(t)

    # ------------------------------------------------------
    # METHOD: diff2t_u – Second Time Derivative of u
    # ------------------------------------------------------
    def diff2t_u(self, x, t):
        """Compute ∂²u/∂t² = h_u(x) · ∂²g_u/∂t²"""
        return self.h_u(x) * self.d2g_u(t)

    # ------------------------------------------------------
    # METHOD: diff1x_v – First Spatial Derivative of v
    # ------------------------------------------------------
    def diff1x_v(self, x, t):
        """Compute ∂v/∂x = ∂h_v/∂x · g_v(t)"""
        return self.d1h_v(x) * self.g_v(t)

    # ------------------------------------------------------
    # METHOD: diff2t_v – Second Time Derivative of v
    # ------------------------------------------------------
    def diff2t_v(self, x, t):
        """Compute ∂²v/∂t² = h_v(x) · ∂²g_v/∂t²"""
        return self.h_v(x) * self.d2g_v(t)

    # ------------------------------------------------------
    # METHOD: nonlinear_term – Nonlinear Source in u Equation
    # ------------------------------------------------------
    def nonlinear_term(self, t):
        """
        Nonlinear coefficient q(t) appearing in the u-equation:
            q(t) = α + β · ∫ (∂u/∂x)² dx

        Analytical simplification:
            ∂u/∂x = ∂h_u/∂x · g_u(t)
            ∫₀^ℓ (∂h_u/∂x)² dx = (λ_u² · π²) / (2ℓ)
        So:
            q(t) = α + β · g_u(t)² · (λ_u² · π²) / (2ℓ)
        """
        integr_coeff = (self.lam_u**2 * np.pi**2) / (2.0 * self.cfg.ell)
        return self.cfg.alpha + self.cfg.beta * integr_coeff * self.g_u(t)**2

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