# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # Core scientific library used for numerical operations and trigonometric functions
from tests.timoshenko_data import TimoshenkoTesterParent  # Base class providing symbolic and numerical solver integration


# ======================================================
# CLASS: Testcase3 – Trigonometric Benchmark for Timoshenko System
# ======================================================

class Testcase3(TimoshenkoTesterParent):
    """
    Symbolic benchmark test case for the nonlinear Timoshenko beam model.

    Defines exact analytical solutions for:
    - Displacement field u(x, t)
    - Rotation field v(x, t)

    These solutions allow validation of numerical solvers via direct comparison
    to analytically known results.
    """

    # ------------------------------------------------------
    # CONSTRUCTOR: Initialize Benchmark Configuration
    # ------------------------------------------------------
    def __init__(self, cfg):
        """
        Initialize the benchmark test case with configuration parameters.

        Parameters
        ----------
        cfg : object
            Configuration object with attributes:
            tau, ell, alpha, beta, gamma, delta, a1, a2, lam_u, lam_v
        """
        self.cfg = cfg
        self.name = "test3"             # Identifier for file/log output
        self.known_solutions = True     # Indicates this case has exact solutions

        super().__init__(cfg)           # Call to parent class constructor
        self._prepare_data()            # Precompute symbolic forms if applicable

        self.lam_u = cfg.lam_u          # Spatial frequency for displacement
        self.lam_v = cfg.lam_v          # Spatial frequency for rotation

    # ------------------------------------------------------
    # METHOD: Spatial Basis Function for Displacement u(x)
    # ------------------------------------------------------
    def h_u(self, x):
        """
        Trigonometric spatial basis function for displacement u(x, t).
        """
        return np.sin(self.lam_u * np.pi * x / self.cfg.ell)

    def d1h_u(self, x):
        """
        First derivative of h_u(x) with respect to x (i.e., ∂h_u/∂x).
        """
        return (self.lam_u * np.pi / self.cfg.ell) * np.cos(self.lam_u * np.pi * x / self.cfg.ell)

    # ------------------------------------------------------
    # METHOD: Spatial Basis Function for Rotation v(x)
    # ------------------------------------------------------
    def h_v(self, x):
        """
        Trigonometric spatial basis function for rotation v(x, t).
        """
        return np.sin(self.lam_v * np.pi * x / self.cfg.ell)

    def d1h_v(self, x):
        """
        First derivative of h_v(x) with respect to x (i.e., ∂h_v/∂x).
        """
        return (self.lam_v * np.pi / self.cfg.ell) * np.cos(self.lam_v * np.pi * x / self.cfg.ell)

    # ------------------------------------------------------
    # METHOD: Temporal Basis Function for Displacement u(t)
    # ------------------------------------------------------
    def g_u(self, t):
        """
        Temporal basis for displacement — sinusoidal in time.
        """
        return np.sin(0.5 * np.pi * t)

    def d2g_u(self, t):
        """
        Second derivative of g_u(t) — ∂²g_u/∂t² for use in acceleration terms.
        """
        return -0.25 * np.pi**2 * np.sin(0.5 * np.pi * t)

    # ------------------------------------------------------
    # METHOD: Temporal Basis Function for Rotation v(t)
    # ------------------------------------------------------
    def g_v(self, t):
        """
        Temporal basis for rotation — sinusoidal in time.
        """
        return np.sin(0.5 * np.pi * t)

    def d2g_v(self, t):
        """
        Second derivative of g_v(t) — ∂²g_v/∂t².
        """
        return -0.25 * np.pi**2 * np.sin(0.5 * np.pi * t)

    # ------------------------------------------------------
    # METHOD: Exact Displacement Field u(x, t)
    # ------------------------------------------------------
    def u(self, x, t):
        """
        Exact solution for displacement:
        u(x, t) = h_u(x) * g_u(t)
        """
        return self.h_u(x) * self.g_u(t)

    # ------------------------------------------------------
    # METHOD: Exact Rotation Field v(x, t)
    # ------------------------------------------------------
    def v(self, x, t):
        """
        Exact solution for rotation:
        v(x, t) = h_v(x) * g_v(t)
        """
        return self.h_v(x) * self.g_v(t)

    # ------------------------------------------------------
    # METHOD: Derivatives of u(x, t)
    # ------------------------------------------------------
    def diff1x_u(self, x, t):
        """
        First spatial derivative: ∂u/∂x = ∂h_u/∂x * g_u(t)
        """
        return self.d1h_u(x) * self.g_u(t)

    def diff2t_u(self, x, t):
        """
        Second temporal derivative: ∂²u/∂t² = h_u(x) * ∂²g_u/∂t²
        """
        return self.h_u(x) * self.d2g_u(t)

    # ------------------------------------------------------
    # METHOD: Derivatives of v(x, t)
    # ------------------------------------------------------
    def diff1x_v(self, x, t):
        """
        First spatial derivative: ∂v/∂x = ∂h_v/∂x * g_v(t)
        """
        return self.d1h_v(x) * self.g_v(t)

    def diff2t_v(self, x, t):
        """
        Second temporal derivative: ∂²v/∂t² = h_v(x) * ∂²g_v/∂t²
        """
        return self.h_v(x) * self.d2g_v(t)

    # ------------------------------------------------------
    # METHOD: Nonlinear Stiffness Term in u-Equation
    # ------------------------------------------------------
    def nonlinear_term(self, t):
        """
        Computes the nonlinear stiffness term in the u-equation:

            q(t) = α + β ∫ (∂u/∂x)² dx

        Using:
            ∂u/∂x = ∂h_u/∂x · g_u(t),
            ∫ (∂h_u/∂x)² dx = (λ_u² · π²) / (2 · ell)

        The expression simplifies to:
            q(t) = α + β · g_u(t)² · [(λ_u² · π²) / (2 · ell)]
        """
        # Integral constant based on symbolic computation over [0, ell]
        integr_coeff = (self.lam_u**2 * np.pi**2) / (2.0 * self.cfg.ell)

        # Final nonlinear term based on current time t
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