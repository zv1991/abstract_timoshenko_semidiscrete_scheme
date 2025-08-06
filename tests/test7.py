# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # Numerical computing library: used for arrays and math functions like exp, sin, cos
from utils.auxiliary import integrate_derivative_form  # Utility function for computing integrals of derivative terms
from tests.timoshenko_data import TimoshenkoTesterParent  # Base test class for symbolic Timoshenko model validation


# ======================================================
# CLASS: Testcase7 – Gaussian and Trigonometric Benchmark for Timoshenko System
# ======================================================

class Testcase7(TimoshenkoTesterParent):
    """
    Symbolic benchmark test case for the nonlinear Timoshenko beam model.

    This test constructs an exact solution using:
    - Spatial trigonometric modes modulated by a Gaussian envelope
    - Temporal periodic functions

    Use Case:
        This supports verification of numerical solvers using
        the Method of Manufactured Solutions (MMS).
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
            Configuration container with:
            - Physical coefficients: alpha, beta, gamma, delta, a1, a2
            - Domain/time: ell, T, tau
            - Oscillation frequencies: lam_u, lam_v, lam1_u, lam1_v
            - Gaussian parameters: A_u, A_v, c_u, c_v
        """
        self.cfg = cfg
        self.name = "test7"
        self.known_solutions = True  # This test has exact solutions available

        super().__init__(cfg)        # Inherit symbolic test setup
        self._prepare_data()         # Precompute source terms and boundary conditions

        # Store spatial and temporal oscillation parameters
        self.lam_u = cfg.lam_u
        self.lam_v = cfg.lam_v
        self.lam1_u = cfg.lam1_u
        self.lam1_v = cfg.lam1_v

        # Gaussian envelope parameters
        self.A_u = cfg.A_u
        self.A_v = cfg.A_v
        self.c_u = cfg.c_u
        self.c_v = cfg.c_v

    # ------------------------------------------------------
    # SPATIAL MODE: h_u – Displacement Envelope
    # ------------------------------------------------------
    def h_u(self, x):
        """
        Spatial profile for displacement u(x, t): Gaussian envelope × sine wave.
        """
        return (
            self.A_u * np.exp(-(2 * x - self.cfg.ell)**2 / self.c_u**2) *
            np.sin(self.lam_u * np.pi * x / self.cfg.ell)
        )

    def d1h_u(self, x):
        """
        First derivative ∂h_u/∂x — used in computing ∂u/∂x.
        """
        # Chain rule applied to product of Gaussian and sine
        return (
            -4.0 / self.c_u**2 * (2 * x - self.cfg.ell) * self.h_u(x) +
            (self.A_u * self.lam_u * np.pi / self.cfg.ell) *
            np.exp(-(2 * x - self.cfg.ell)**2 / self.c_u**2) *
            np.cos(self.lam_u * np.pi * x / self.cfg.ell)
        )

    # ------------------------------------------------------
    # SPATIAL MODE: h_v – Rotation Envelope
    # ------------------------------------------------------
    def h_v(self, x):
        """
        Spatial profile for rotation v(x, t): Gaussian envelope × sine wave.
        """
        return (
            self.A_v * np.exp(-(2 * x - self.cfg.ell)**2 / self.c_v**2) *
            np.sin(self.lam_v * np.pi * x / self.cfg.ell)
        )

    def d1h_v(self, x):
        """
        First derivative ∂h_v/∂x — used in computing ∂v/∂x.
        """
        return (
            -4.0 / self.c_v**2 * (2 * x - self.cfg.ell) * self.h_v(x) +
            (self.A_v * self.lam_v * np.pi / self.cfg.ell) *
            np.exp(-(2 * x - self.cfg.ell)**2 / self.c_v**2) *
            np.cos(self.lam_v * np.pi * x / self.cfg.ell)
        )

    # ------------------------------------------------------
    # TEMPORAL MODE: g_u – Displacement Oscillation
    # ------------------------------------------------------
    def g_u(self, t):
        """
        Time-dependent multiplier for u(x, t): Cosine temporal oscillation.
        """
        return 1.0 + np.cos(self.lam1_u * np.pi * t / self.cfg.T)

    def d2g_u(self, t):
        """
        Second time derivative ∂²g_u/∂t² — required for dynamic equation of u.
        """
        return -(
            self.lam1_u * np.pi / self.cfg.T
        )**2 * np.cos(self.lam1_u * np.pi * t / self.cfg.T)

    # ------------------------------------------------------
    # TEMPORAL MODE: g_v – Rotation Oscillation
    # ------------------------------------------------------
    def g_v(self, t):
        """
        Time-dependent multiplier for v(x, t): Cosine temporal oscillation.
        """
        return 1.0 + np.cos(self.lam1_v * np.pi * t / self.cfg.T)

    def d2g_v(self, t):
        """
        Second time derivative ∂²g_v/∂t² — required for dynamic equation of v.
        """
        return -(
            self.lam1_v * np.pi / self.cfg.T
        )**2 * np.cos(self.lam1_v * np.pi * t / self.cfg.T)

    # ------------------------------------------------------
    # EXACT SOLUTION: u(x, t)
    # ------------------------------------------------------
    def u(self, x, t):
        """
        Exact displacement solution u(x, t) = h_u(x) · g_u(t)
        """
        return self.h_u(x) * self.g_u(t)

    # ------------------------------------------------------
    # EXACT SOLUTION: v(x, t)
    # ------------------------------------------------------
    def v(self, x, t):
        """
        Exact rotation solution v(x, t) = h_v(x) · g_v(t)
        """
        return self.h_v(x) * self.g_v(t)

    # ------------------------------------------------------
    # DERIVATIVE: ∂u/∂x
    # ------------------------------------------------------
    def diff1x_u(self, x, t):
        """
        First spatial derivative ∂u/∂x = ∂h_u/∂x · g_u(t)
        """
        return self.d1h_u(x) * self.g_u(t)

    # ------------------------------------------------------
    # DERIVATIVE: ∂²u/∂t²
    # ------------------------------------------------------
    def diff2t_u(self, x, t):
        """
        Second time derivative ∂²u/∂t² = h_u(x) · ∂²g_u/∂t²
        """
        return self.h_u(x) * self.d2g_u(t)

    # ------------------------------------------------------
    # DERIVATIVE: ∂v/∂x
    # ------------------------------------------------------
    def diff1x_v(self, x, t):
        """
        First spatial derivative ∂v/∂x = ∂h_v/∂x · g_v(t)
        """
        return self.d1h_v(x) * self.g_v(t)

    # ------------------------------------------------------
    # DERIVATIVE: ∂²v/∂t²
    # ------------------------------------------------------
    def diff2t_v(self, x, t):
        """
        Second time derivative ∂²v/∂t² = h_v(x) · ∂²g_v/∂t²
        """
        return self.h_v(x) * self.d2g_v(t)

    # ------------------------------------------------------
    # NONLINEAR TERM: α + β ∫ (∂u/∂x)² dx
    # ------------------------------------------------------
    def nonlinear_term(self, t):
        """
        Compute nonlinear stiffness term for the u-equation:
            α + β ∫ (∂u/∂x)² dx
        This models geometric nonlinearity in the beam.
        """
        # Define function to integrate: (∂u/∂x)
        integrand = lambda x: self.diff1x_u(x, t)

        # Perform integration using custom utility
        result, *_ = integrate_derivative_form(
            df=integrand,
            ell=self.cfg.ell,
            form='squared'  # Compute integral of (∂u/∂x)²
        )

        # Return α + β ∫(∂u/∂x)² dx
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