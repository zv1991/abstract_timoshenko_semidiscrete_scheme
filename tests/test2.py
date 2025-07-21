# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # Numerical library for arrays and mathematical operations
from utils.auxiliary import integrate_derivative_form  # Computes ∫(∂u/∂x)² dx — used for nonlinear stiffness term
from tests.timoshenko_data import TimoshenkoTesterParent  # Base class handling symbolic/numeric init logic


# ======================================================
# CLASS: Testcase2 – Trigonometric Benchmark for Timoshenko System
# ======================================================

class Testcase2(TimoshenkoTesterParent):
    """
    Symbolic benchmark test case for the nonlinear Timoshenko beam model.

    This test case provides closed-form trigonometric expressions for:
    - Displacement field u(x, t)
    - Rotation field v(x, t)

    These allow verification of numerical methods via direct comparison
    to exact solutions and help test solver accuracy and stability.
    """

    # ------------------------------------------------------
    # CONSTRUCTOR: Initialize with configuration
    # ------------------------------------------------------
    def __init__(self, cfg):
        """
        Initialize the benchmark using external configuration.

        Parameters
        ----------
        cfg : object or dict-like
            Beam configuration object containing:
            tau, ell, alpha, beta, gamma, delta, a1, a2, lam_u, lam_v
        """
        self.cfg = cfg
        self.name = "test2"           # Identifier for logging and plot output
        self.known_solutions = True   # Enables exact solution-based solver validation
        super().__init__(cfg)         # Initialize base class
        self._prepare_data()          # Perform any symbolic setup or precaching

        # Store oscillation frequencies for spatial basis functions
        self.lam_u = cfg.lam_u
        self.lam_v = cfg.lam_v

    # ------------------------------------------------------
    # METHOD: Spatial Basis for Displacement u(x)
    # ------------------------------------------------------
    def h_u(self, x):
        return np.sin(self.lam_u * np.pi * x / self.cfg.ell)

    def d1h_u(self, x):
        # First spatial derivative of h_u(x)
        return (self.lam_u * np.pi / self.cfg.ell) * np.cos(self.lam_u * np.pi * x / self.cfg.ell)

    def d2h_u(self, x):
        # Second spatial derivative of h_u(x)
        factor = (self.lam_u * np.pi / self.cfg.ell) ** 2
        return -factor * np.sin(self.lam_u * np.pi * x / self.cfg.ell)

    # ------------------------------------------------------
    # METHOD: Spatial Basis for Rotation v(x)
    # ------------------------------------------------------
    def h_v(self, x):
        return np.sin(self.lam_v * np.pi * x / self.cfg.ell)

    def d1h_v(self, x):
        return (self.lam_v * np.pi / self.cfg.ell) * np.cos(self.lam_v * np.pi * x / self.cfg.ell)

    def d2h_v(self, x):
        factor = (self.lam_v * np.pi / self.cfg.ell) ** 2
        return -factor * np.sin(self.lam_v * np.pi * x / self.cfg.ell)

    # ------------------------------------------------------
    # METHOD: Temporal Basis for Displacement u(t)
    # ------------------------------------------------------
    def g_u(self, t):
        return t

    def d1g_u(self, t):
        return np.float64(0.0)

    def d2g_u(self, t):
        return np.float64(0.0)

    # ------------------------------------------------------
    # METHOD: Temporal Basis for Rotation v(t)
    # ------------------------------------------------------
    def g_v(self, t):
        return t

    def d1g_v(self, t):
        return np.float64(0.0)

    def d2g_v(self, t):
        return np.float64(0.0)

    # ------------------------------------------------------
    # METHOD: Exact Solution Functions u(x, t) and v(x, t)
    # ------------------------------------------------------
    def u(self, x, t):
        return self.h_u(x) * self.g_u(t)

    def v(self, x, t):
        return self.h_v(x) * self.g_v(t)

    # ------------------------------------------------------
    # METHOD: Derivatives of u(x, t)
    # ------------------------------------------------------
    def diff1t_u(self, x, t):
        return self.h_u(x) * self.d1g_u(t)

    def diff2t_u(self, x, t):
        return self.h_u(x) * self.d2g_u(t)

    def diff1x_u(self, x, t):
        return self.d1h_u(x) * self.g_u(t)

    def diff2x_u(self, x, t):
        return self.d2h_u(x) * self.g_u(t)

    # ------------------------------------------------------
    # METHOD: Derivatives of v(x, t)
    # ------------------------------------------------------
    def diff1t_v(self, x, t):
        return self.h_v(x) * self.d1g_v(t)

    def diff2t_v(self, x, t):
        return self.h_v(x) * self.d2g_v(t)

    def diff1x_v(self, x, t):
        return self.d1h_v(x) * self.g_v(t)

    def diff2x_v(self, x, t):
        return self.d2h_v(x) * self.g_v(t)

    # ------------------------------------------------------
    # METHOD: Nonlinear Term ∫(∂u/∂x)² dx
    # ------------------------------------------------------
    def integr_term(self, t):
        """
        Computes nonlinear stiffness term:
            ∫ (∂u/∂x)² dx

        Parameters
        ----------
        t : float
            Time at which to evaluate derivative

        Returns
        -------
        float
            Integral result at time t
        """
        integrand = lambda x: self.diff1x_u(x, t)
        result, *_ = integrate_derivative_form(df=integrand, ell=self.cfg.ell)
        return result

    # ------------------------------------------------------
    # METHOD: f₁(x, t) – RHS of Displacement Equation
    # ------------------------------------------------------
    def f1(self, x, t):
        """
        Compute RHS for displacement equation:

        f₁(x, t) = ∂²u/∂t² - (α + β ∫(∂u/∂x)² dx) ∂²u/∂x² + a₁ ∂v/∂x

        Parameters
        ----------
        x : float
        t : float

        Returns
        -------
        float
            Source term for u-equation
        """
        return (
            self.diff2t_u(x, t)
            - (self.cfg.alpha + self.cfg.beta * self.integr_term(t)) * self.diff2x_u(x, t)
            + self.cfg.a1 * self.diff1x_v(x, t)
        )

    # ------------------------------------------------------
    # METHOD: f₂(x, t) – RHS of Rotation Equation
    # ------------------------------------------------------
    def f2(self, x, t):
        """
        Compute RHS for rotation equation:

        f₂(x, t) = ∂²v/∂t² - γ ∂²v/∂x² + δ v - a₂ ∂u/∂x

        Parameters
        ----------
        x : float
        t : float

        Returns
        -------
        float
            Source term for v-equation
        """
        return (
            self.diff2t_v(x, t)
            - self.cfg.gamma * self.diff2x_v(x, t)
            + self.cfg.delta * self.v(x, t)
            - self.cfg.a2 * self.diff1x_u(x, t)
        )

    # ------------------------------------------------------
    # METHOD: Post-Init Hook (For Dataclasses Compatibility)
    # ------------------------------------------------------
    def __post_init__(self):
        """
        Optional post-initialization hook for dataclass usage.
        Ensures custom setup logic is re-applied.
        """
        self._prepare_data()