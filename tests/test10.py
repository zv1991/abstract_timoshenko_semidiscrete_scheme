# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # Numerical computing library: arrays, mathematical functions, exp, sin, cos
from tests.timoshenko_data import TimoshenkoTesterParent  # Parent class for test cases using symbolic PDE data


# ======================================================
# CLASS: Testcase10 – Unknown Benchmark Solution for Timoshenko System
# ======================================================

class Testcase10(TimoshenkoTesterParent):
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
    def __init__(self, cfg):
        """
        Initialize the benchmark test case using configuration parameters.

        Parameters
        ----------
        cfg : object
            Configuration containing:
            - Physical coefficients: alpha, beta, gamma, delta, a1, a2
            - Domain and time step: ell, tau
            - Oscillation frequencies: lam_u, lam_v
            - Gaussian shape parameters: A_*, c_*
        """
        self.cfg = cfg
        self.name = "test10"
        self.known_solutions = False  # No exact time-evolving solution is provided

        # Oscillation parameters
        self.lam_u = cfg.lam_u
        self.lam_v = cfg.lam_v

        # Gaussian shape parameters
        self.A_u = cfg.A_u  # Amplitude of Gaussian for u
        self.A_v = cfg.A_v  # Amplitude of Gaussian for v
        self.c_u = cfg.c_u  # Width of Gaussian for u
        self.c_v = cfg.c_v  # Width of Gaussian for v
        
        # Argument scalling in trigonometric functions
        self.d_u = self.lam_u * np.pi / self.cfg.ell
        self.d_v = self.lam_v * np.pi / self.cfg.ell
        
        super().__init__(cfg)        # Initialize symbolic setup from parent class
        self._prepare_data()         # Precompute symbolic forcing and boundary data

    # ------------------------------------------------------
    # GAUSSIAN PROFILES FOR INITIAL CONDITIONS
    # ------------------------------------------------------
    def gauss_u(self, x):
        """Gaussian envelope for u(x) centered at mid-domain."""
        return self.A_u * np.exp(-(2 * x - self.cfg.ell)**2 / self.c_u**2)

    def gauss_v(self, x):
        """Gaussian envelope for v(x) centered at mid-domain."""
        return self.A_v * np.exp(-(2 * x - self.cfg.ell)**2 / self.c_v**2)

    # ------------------------------------------------------
    # TRIGONOMETRIC MULTIPLIERS FOR MODAL STRUCTURE
    # ------------------------------------------------------
    def h_u(self, x):
        """Sine wave spatial profile for displacement u(x)."""
        return np.sin(self.d_u * x)

    def h_v(self, x):
        """Sine wave spatial profile for rotation v(x)."""
        return np.sin(self.d_v * x)

    # ------------------------------------------------------
    # INITIAL CONDITIONS (u, v) AT t = 0
    # ------------------------------------------------------
    def varphi0(self, x):
        """Initial displacement u(x, 0) = Gaussian * sine."""
        return self.gauss_u(x) * self.h_u(x)

    def psi0(self, x):
        """Initial rotation v(x, 0) = Gaussian * sine."""
        return self.gauss_v(x) * self.h_v(x)

    # ------------------------------------------------------
    # FIRST SPATIAL DERIVATIVES AT t = 0
    # ------------------------------------------------------
    def d1varphi0(self, x):
        """∂u/∂x at t = 0 (product rule applied to Gaussian * sine)."""
        return (
            self.gauss_u(x) * (
                - 4.0 * (2 * x - self.cfg.ell) / self.c_u**2 * self.h_u(x)
                + self.d_u * np.cos(self.d_u * x)
                )
        )

    def d1psi0(self, x):
        """∂v/∂x at t = 0 (product rule applied to Gaussian * sine)."""
        return (
            self.gauss_v(x) * (
                - 4.0 * (2 * x - self.cfg.ell) / self.c_v**2 * self.h_v(x)
                + self.d_v * np.cos(self.d_v * x)
                )
        )

    # ------------------------------------------------------
    # SECOND SPATIAL DERIVATIVES AT t = 0
    # ------------------------------------------------------
    def d2varphi0(self, x):
        """∂²u/∂x² at t = 0."""
        return (
            self.gauss_u(x) * (
                (16.0 * (2 * x - self.cfg.ell)**2 / self.c_u**4
                 - 8.0 / self.c_u**2
                 - self.d_u**2) * self.h_u(x)
                - 8.0 * self.d_u * (2 * x - self.cfg.ell) / self.c_u**2 * np.cos(self.d_u * x)
                )
        )

    def d2psi0(self, x):
        """∂²v/∂x² at t = 0."""
        return (
            self.gauss_v(x) * (
                (16.0 * (2 * x - self.cfg.ell)**2 / self.c_v**4
                 - 8.0 / self.c_v**2
                 - self.d_v**2) * self.h_v(x)
                - 8.0 * self.d_v * (2 * x - self.cfg.ell) / self.c_v**2 * np.cos(self.d_v * x)
                )
        )

    # ------------------------------------------------------
    # THIRD SPATIAL DERIVATIVES AT t = 0
    # ------------------------------------------------------
    def d3varphi0(self, x):
        """∂³u/∂x³ at t = 0."""
        return (
            self.gauss_u(x) * (
                (2 * x - self.cfg.ell) * (
                    -64.0 * (2 * x - self.cfg.ell)**2 / self.c_u**6
                    + 96.0 / self.c_u**4
                    + 12 * self.d_u**2 / self.c_u**2
                    ) * self.h_u(x)
                + (
                    48.0 * self.d_u * (2 * x - self.cfg.ell)**2 / self.c_u**4
                    - 24 * self.d_u / self.c_u ** 2
                    - self.d_u**3
                    ) * np.cos(self.d_u * x)
                )
        )

    def d3psi0(self, x):
        """∂³v/∂x³ at t = 0."""
        return (
            self.gauss_v(x) * (
                (2 * x - self.cfg.ell) * (
                    -64.0 * (2 * x - self.cfg.ell)**2 / self.c_v**6
                    + 96.0 / self.c_v**4
                    + 12 * self.d_v**2 / self.c_v**2
                    ) * self.h_v(x)
                + (
                    48.0 * self.d_v * (2 * x - self.cfg.ell)**2 / self.c_v**4
                    - 24 * self.d_v / self.c_v ** 2
                    - self.d_v**3
                    ) * np.cos(self.d_v * x)
                )
        )

    # ------------------------------------------------------
    # TIME DERIVATIVES (STATIC CASE)
    # ------------------------------------------------------
    def varphi1(self, x):
        """Initial time derivative ∂u/∂t at t = 0 — set to zero for static initial state."""
        return np.float64(0.0)

    def psi1(self, x):
        """Initial time derivative ∂v/∂t at t = 0 — set to zero for static initial state."""
        return np.float64(0.0)

    def d1varphi1(self, x):
        """∂/∂x of ∂u/∂t at t = 0 — zero for static initial state."""
        return np.float64(0.0)

    def d1psi1(self, x):
        """∂/∂x of ∂v/∂t at t = 0 — zero for static initial state."""
        return np.float64(0.0)

    # ------------------------------------------------------
    # FORCING TERMS (SOURCE FUNCTIONS)
    # ------------------------------------------------------
    def f1(self, x, t):
        """External forcing in the u-equation — zero for manufactured test."""
        return np.float64(0.0)

    def f2(self, x, t):
        """External forcing in the v-equation — zero for manufactured test."""
        return np.float64(0.0)
    
    # First spatial derivatives of forcing terms
    def d1f1(self, x, t):
        return np.float64(0.0)
    def d1f2(self, x, t):
        return np.float64(0.0)

    # ------------------------------------------------------
    # POST INITIALIZATION (OPTIONAL DATACLASS SUPPORT)
    # ------------------------------------------------------
    def __post_init__(self):
        """
        Hook for dataclass compatibility.
        Ensures post-construction preparation is called if needed.
        """
        self._prepare_data()