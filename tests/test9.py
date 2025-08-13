# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # Numerical computing library: vectors/arrays, ufuncs (sin/cos), constants (pi)
from tests.timoshenko_data import TimoshenkoTesterParent  # Parent class: provides MMS/PDE test infrastructure


# ======================================================
# CLASS: Testcase9 – Unknown Benchmark Solution for Timoshenko System
# ======================================================

class Testcase9(TimoshenkoTesterParent):
    """
    Unknown benchmark test case for the nonlinear Timoshenko beam model.

    Purpose:
        Validates numerical solvers via the Method of Manufactured Solutions (MMS).
        This test case uses synthetic initial data but does not provide known closed-form
        solutions for the full PDE — useful for convergence or robustness testing.
    """

    # ------------------------------------------------------
    # CONSTRUCTOR: Initialize Configuration and Parameters
    # ------------------------------------------------------
    # === Method Title: __init__ — construct test case and precompute constants
    def __init__(self, cfg):
        """
        Initialize the benchmark test case using configuration parameters.

        Parameters
        ----------
        cfg : object
            Configuration containing:
            - Physical coefficients: alpha, beta, gamma, delta, a1, a2
            - Domain/time: ell, tau
            - Oscillation frequencies: lam_u, lam_v
        """
        self.cfg = cfg                         # Store configuration for use across methods
        self.name = "test9"                    # Identifier used by the testing framework
        self.known_solutions = False           # No exact time-evolving solution is provided

        # Oscillation parameters (copied locally for convenience/readability)
        self.lam_u = cfg.lam_u                 # Spatial frequency for u
        self.lam_v = cfg.lam_v                 # Spatial frequency for v

        # Arguments in trigonometric functions: c_* = λ_* π / ℓ
        # These appear frequently in sin/cos(c_* x) and their derivatives wrt x.
        self.c_u = self.lam_u * np.pi / self.cfg.ell
        self.c_v = self.lam_v * np.pi / self.cfg.ell

        super().__init__(cfg)                  # Initialize symbolic setup from parent class
        self._prepare_data()                   # Precompute symbolic forcing and boundary data

    # ------------------------------------------------------
    # INITIAL CONDITIONS (u, v) AT t = 0
    # ------------------------------------------------------
    # === Method Title: varphi0 — initial displacement u(x, 0)
    def varphi0(self, x):
        """Initial displacement u(x, 0)."""
        return np.float64(0.0)  # Scalar float64; NumPy broadcasts over vector x if given

    # === Method Title: psi0 — initial rotation v(x, 0)
    def psi0(self, x):
        """Initial rotation v(x, 0)."""
        return np.float64(0.0)

    # ------------------------------------------------------
    # FIRST SPATIAL DERIVATIVES AT t = 0
    # ------------------------------------------------------
    # === Method Title: d1varphi0 — ∂u/∂x at t = 0
    def d1varphi0(self, x):
        """First spatial derivative of u at t = 0 (zero for this test case)."""
        return np.float64(0.0)

    # === Method Title: d1psi0 — ∂v/∂x at t = 0
    def d1psi0(self, x):
        """First spatial derivative of v at t = 0 (zero for this test case)."""
        return np.float64(0.0)

    # ------------------------------------------------------
    # SECOND SPATIAL DERIVATIVES AT t = 0
    # ------------------------------------------------------
    # === Method Title: d2varphi0 — ∂²u/∂x² at t = 0
    def d2varphi0(self, x):
        """Second spatial derivative of u at t = 0 (zero for this test case)."""
        return np.float64(0.0)

    # === Method Title: d2psi0 — ∂²v/∂x² at t = 0
    def d2psi0(self, x):
        """Second spatial derivative of v at t = 0 (zero for this test case)."""
        return np.float64(0.0)

    # ------------------------------------------------------
    # THIRD SPATIAL DERIVATIVES AT t = 0
    # ------------------------------------------------------
    # === Method Title: d3varphi0 — ∂³u/∂x³ at t = 0
    def d3varphi0(self, x):
        """Third spatial derivative of u at t = 0 (zero for this test case)."""
        return np.float64(0.0)

    # === Method Title: d3psi0 — ∂³v/∂x³ at t = 0
    def d3psi0(self, x):
        """Third spatial derivative of v at t = 0 (zero for this test case)."""
        return np.float64(0.0)

    # ------------------------------------------------------
    # TIME DERIVATIVES (INITIAL VELOCITIES)
    # ------------------------------------------------------
    # === Method Title: varphi1 — initial time derivative ∂u/∂t at t = 0
    def varphi1(self, x):
        """
        Initial velocity for u.
        u_t(x,0) = (π/2) * sin(c_u * x).
        """
        return 0.5 * np.pi * np.sin(self.c_u * x)  # Vectorized over x

    # === Method Title: psi1 — initial time derivative ∂v/∂t at t = 0
    def psi1(self, x):
        """Initial time derivative ∂v/∂t at t = 0."""
        return 0.5 * np.pi * np.sin(self.c_v * x)

    # === Method Title: d1varphi1 — ∂/∂x of (∂u/∂t) at t = 0
    def d1varphi1(self, x):
        """
        Spatial derivative of initial velocity:
        d/dx[(π/2) sin(c_u x)] = (π/2) * c_u * cos(c_u x)
                              = (lam_u * π^2) / (2 * ell) * cos(c_u x).
        """
        return (self.lam_u * np.pi**2) / (2.0 * self.cfg.ell) * np.cos(self.c_u * x)

    # === Method Title: d1psi1 — ∂/∂x of (∂v/∂t) at t = 0
    def d1psi1(self, x):
        """
        Spatial derivative of initial velocity for v:
        d/dx[(π/2) sin(c_v x)] = (π/2) * c_v * cos(c_v x)
                              = (lam_v * π^2) / (2 * ell) * cos(c_v x).
        """
        return (self.lam_v * np.pi**2) / (2.0 * self.cfg.ell) * np.cos(self.c_v * x)

    # ------------------------------------------------------
    # FORCING TERMS (SOURCE FUNCTIONS)
    # ------------------------------------------------------
    # === Method Title: f1 — forcing in displacement equation (u-equation)
    def f1(self, x, t):
        """
        External forcing for the u-equation.

        Structure:
            f1 = [ (-π^2/4 + c_u^2*(alpha + beta*K*s^2)) * s * sin(c_u x) ]
                 + [ a1 * c_v * s * cos(c_v x) ]

        where:
            s  = sin(π t / 2)
            K  = (lam_u * π)^2 / (2 * ell)   # surrogate for |∂u/∂x|^2 scaling in MMS

        Notes:
            - We cache s and s^2 to reduce repeated trig evaluations.
            - temporal depends on t but not x; spatial dependence is only via sin/cos.
        """
        s = np.sin(0.5 * np.pi * t)          # time-dependent factor reused multiple times
        s2 = s * s                           # sin^2(π t / 2) for the nonlinear beta term
        temporal = (-0.25 * np.pi**2) + self.c_u**2 * (
            self.cfg.alpha + self.cfg.beta * (self.lam_u * np.pi)**2 / (2.0 * self.cfg.ell) * s2
        )
        return temporal * s * np.sin(self.c_u * x) + self.cfg.a1 * self.c_v * s * np.cos(self.c_v * x)

    # === Method Title: f2 — forcing in rotation equation (v-equation)
    def f2(self, x, t):
        """
        External forcing for the v-equation.

        Structure:
            f2 = [ (-π^2/4 + γ c_v^2 + δ) * s * sin(c_v x) ]
                 - [ a2 * c_u * s * cos(c_u x) ]

        where s = sin(π t / 2).
        """
        s = np.sin(0.5 * np.pi * t)          # shared temporal factor
        temporal_v = (-0.25 * np.pi**2) + self.cfg.gamma * self.c_v**2 + self.cfg.delta
        return temporal_v * s * np.sin(self.c_v * x) - self.cfg.a2 * self.c_u * s * np.cos(self.c_u * x)

    # === Method Title: d1f1 — ∂/∂x of forcing f1
    def d1f1(self, x, t):
        """
        First spatial derivative of f1.

        Identities used:
            d/dx[sin(c_u x)] = c_u cos(c_u x)
            d/dx[cos(c_v x)] = -c_v sin(c_v x)
        """
        s = np.sin(0.5 * np.pi * t)          # reuse temporal factor
        s2 = s * s
        temporal = (-0.25 * np.pi**2) + self.c_u**2 * (
            self.cfg.alpha + self.cfg.beta * (self.lam_u * np.pi)**2 / (2.0 * self.cfg.ell) * s2
        )
        return temporal * s * self.c_u * np.cos(self.c_u * x) - self.cfg.a1 * (self.c_v**2) * s * np.sin(self.c_v * x)

    # === Method Title: d1f2 — ∂/∂x of forcing f2
    def d1f2(self, x, t):
        """
        First spatial derivative of f2.

        Identities used:
            d/dx[sin(c_v x)] = c_v cos(c_v x)
            d/dx[cos(c_u x)] = -c_u sin(c_u x)
        """
        s = np.sin(0.5 * np.pi * t)
        temporal_v = (-0.25 * np.pi**2) + self.cfg.gamma * self.c_v**2 + self.cfg.delta
        return temporal_v * s * self.c_v * np.cos(self.c_v * x) + self.cfg.a2 * (self.c_u**2) * s * np.sin(self.c_u * x)

    # ------------------------------------------------------
    # POST INITIALIZATION (OPTIONAL DATACLASS SUPPORT)
    # ------------------------------------------------------
    # === Method Title: __post_init__ — optional hook for dataclass compatibility
    def __post_init__(self):
        """
        Hook for dataclass compatibility.
        Ensures post-construction preparation is called if needed.

        Note:
            This class is not a dataclass, so __post_init__ is not invoked automatically.
            It's kept for compatibility with frameworks that might call this hook explicitly.
        """
        self._prepare_data()  # Safe to call; parent typically guards idempotency