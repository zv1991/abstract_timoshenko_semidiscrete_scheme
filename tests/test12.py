# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # Numerical computing library: provides efficient arrays, trigonometric functions, etc.
from tests.timoshenko_data import TimoshenkoTesterParent  # Base class for symbolic test cases
# NOTE: TimoshenkoTesterParent provides symbolic PDE evaluation and test scaffolding utilities.


# ======================================================
# CLASS: Testcase12 – Unknown Benchmark Solution for Timoshenko System
# ======================================================

class Testcase12(TimoshenkoTesterParent):
    """
    Unknown benchmark test case for the nonlinear Timoshenko beam model.

    Purpose
    -------
    Validates solvers via the Method of Manufactured Solutions (MMS) without
    requiring an exact known solution — ideal for convergence and robustness tests.

    Notes
    -----
    - All spatial functions accept both scalars and arrays (via np.asarray).
    - Modal structure is imposed using sine and cosine functions.
    """

    # ------------------------------------------------------
    # CONSTRUCTOR: Initialize Parameters and Precompute Fields
    # ------------------------------------------------------
    def __init__(self, cfg):
        """
        Initialize test case with simulation configuration.

        Parameters
        ----------
        cfg : object
            Contains model coefficients, oscillation frequencies, domain size, and amplitude:
            - alpha, beta, gamma, delta, a1, a2
            - lam, lam1, A
            - ell, T, tau
        """
        self.cfg = cfg
        self.name = "test12"
        self.known_solutions = False  # No analytical time-dependent solution available

        # Oscillation parameters (used in sine/cosine arguments)
        self.lam = cfg.lam
        self.lam1 = cfg.lam1
        self.A = cfg.A  # Amplitude for u and v fields

        # Spatial and temporal frequency scaling
        self.d = self.lam * np.pi / cfg.ell    # Spatial frequency for initial conditions
        self.d1 = self.lam1 * np.pi / cfg.T    # Temporal frequency for manufactured forcing

        # Parent initialization
        super().__init__(cfg)
        self._prepare_data()  # Symbolically generate source/BC data

    # ------------------------------------------------------
    # INITIAL CONDITIONS AT t = 0
    # ------------------------------------------------------

    # ---------- METHOD: varphi0 ----------
    def varphi0(self, x):
        """Initial displacement u(x, 0) = A * sin(d * x)."""
        x = np.asarray(x)
        return self.A * np.sin(self.d * x)

    # ---------- METHOD: psi0 ----------
    def psi0(self, x):
        """Initial rotation v(x, 0) = A * sin(d * x)."""
        x = np.asarray(x)
        return self.A * np.sin(self.d * x)

    # ------------------------------------------------------
    # FIRST SPATIAL DERIVATIVES AT t = 0
    # ------------------------------------------------------

    # ---------- METHOD: d1varphi0 ----------
    def d1varphi0(self, x):
        """∂u/∂x at t = 0."""
        x = np.asarray(x)
        return self.A * self.d * np.cos(self.d * x)

    # ---------- METHOD: d1psi0 ----------
    def d1psi0(self, x):
        """∂v/∂x at t = 0."""
        x = np.asarray(x)
        return self.A * self.d * np.cos(self.d * x)

    # ------------------------------------------------------
    # SECOND SPATIAL DERIVATIVES AT t = 0
    # ------------------------------------------------------

    # ---------- METHOD: d2varphi0 ----------
    def d2varphi0(self, x):
        """∂²u/∂x² at t = 0."""
        x = np.asarray(x)
        return -self.A * self.d**2 * np.sin(self.d * x)

    # ---------- METHOD: d2psi0 ----------
    def d2psi0(self, x):
        """∂²v/∂x² at t = 0."""
        x = np.asarray(x)
        return -self.A * self.d**2 * np.sin(self.d * x)

    # ------------------------------------------------------
    # THIRD SPATIAL DERIVATIVES AT t = 0
    # ------------------------------------------------------

    # ---------- METHOD: d3varphi0 ----------
    def d3varphi0(self, x):
        """∂³u/∂x³ at t = 0."""
        x = np.asarray(x)
        return -self.A * self.d**3 * np.cos(self.d * x)

    # ---------- METHOD: d3psi0 ----------
    def d3psi0(self, x):
        """∂³v/∂x³ at t = 0."""
        x = np.asarray(x)
        return -self.A * self.d**3 * np.cos(self.d * x)

    # ------------------------------------------------------
    # TIME DERIVATIVES AT t = 0 (STATIC CASE)
    # ------------------------------------------------------

    # ---------- METHOD: varphi1 ----------
    def varphi1(self, x):
        """∂u/∂t at t = 0 — zero for static initial condition."""
        x = np.asarray(x)
        return np.float64(0.0)

    # ---------- METHOD: psi1 ----------
    def psi1(self, x):
        """∂v/∂t at t = 0 — zero for static initial condition."""
        x = np.asarray(x)
        return np.float64(0.0)

    # ---------- METHOD: d1varphi1 ----------
    def d1varphi1(self, x):
        """∂/∂x of ∂u/∂t at t = 0 — remains zero."""
        x = np.asarray(x)
        return np.float64(0.0)

    # ---------- METHOD: d1psi1 ----------
    def d1psi1(self, x):
        """∂/∂x of ∂v/∂t at t = 0 — remains zero."""
        x = np.asarray(x)
        return np.float64(0.0)

    # ------------------------------------------------------
    # FORCING TERMS (RHS FUNCTIONS)
    # ------------------------------------------------------

    # ---------- METHOD: f1 ----------
    def f1(self, x, t):
        """Manufactured source term in u-equation."""
        coeff1 = -self.d1**2
        coeff2 = self.d**2 * (
            self.cfg.alpha +
            ((self.A * self.lam * np.pi)**2 / (2.0 * self.cfg.ell)) *
            self.cfg.beta * np.cos(self.lam1 * t)**2
        )
        coeff3 = self.d * self.cfg.a1

        x = np.asarray(x)
        return (
            self.A * (coeff1 + coeff2) * np.cos(self.d1 * t) * np.sin(self.d * x) +
            self.A * coeff3 * np.cos(self.d1 * t) * np.cos(self.d * x)
        )

    # ---------- METHOD: f2 ----------
    def f2(self, x, t):
        """Manufactured source term in v-equation."""
        x = np.asarray(x)
        return -self.A * self.d * self.cfg.a2 * np.cos(self.d1 * t) * np.cos(self.d * x)

    # ---------- METHOD: d1f1 ----------
    def d1f1(self, x, t):
        """∂/∂x of u-forcing term."""
        coeff1 = -self.d1**2
        coeff2 = self.d**2 * (
            self.cfg.alpha +
            ((self.A * self.lam * np.pi)**2 / (2.0 * self.cfg.ell)) *
            self.cfg.beta * np.cos(self.lam1 * t)**2
        )
        coeff3 = self.d * self.cfg.a1

        x = np.asarray(x)
        return (
            self.A * self.d * (coeff1 + coeff2) * np.cos(self.d1 * t) * np.cos(self.d * x) -
            self.A * coeff3 * self.d * np.cos(self.d1 * t) * np.sin(self.d * x)
        )

    # ---------- METHOD: d1f2 ----------
    def d1f2(self, x, t):
        """∂/∂x of v-forcing term."""
        x = np.asarray(x)
        return self.A * self.d**2 * self.cfg.a2 * np.cos(self.d1 * t) * np.sin(self.d * x)

    # ------------------------------------------------------
    # DATACLASS POST-CONSTRUCTION HOOK (OPTIONAL)
    # ------------------------------------------------------

    # ---------- METHOD: __post_init__ ----------
    def __post_init__(self):
        """
        Post-construction setup for dataclass compatibility.
        Ensures symbolic data generation occurs even if __init__ is bypassed.
        """
        self._prepare_data()