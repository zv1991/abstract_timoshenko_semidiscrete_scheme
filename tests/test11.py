# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # Numerical computing library: arrays, mathematical functions, exp, sin, cos
from tests.timoshenko_data import TimoshenkoTesterParent  # Parent class for test cases using symbolic PDE data
# NOTE: The parent class is assumed to provide test harness utilities and _prepare_data().
#       This testcase focuses on providing initial/boundary data and their derivatives.


# ======================================================
# CLASS: Testcase11 – Unknown Benchmark Solution for Timoshenko System
# ======================================================

class Testcase11(TimoshenkoTesterParent):
    """
    Unknown benchmark test case for the nonlinear Timoshenko beam model.

    Purpose
    -------
    Validates numerical solvers via the Method of Manufactured Solutions (MMS).
    This test case uses synthetic initial data but does not provide known closed-form
    solutions for the full PDE — useful for convergence or robustness testing.

    Notes
    -----
    - All functions accept either scalars or array-like x; np.asarray is used to
      enable vectorized evaluation in downstream solvers or test grids.
    - Trigonometric factors provide modal structure with spatial frequencies d_u, d_v.
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
            - Amplitude parameters: A_*
        """
        self.cfg = cfg
        self.name = "test11"
        self.known_solutions = False  # No exact time-evolving solution is provided

        # Oscillation parameters
        self.lam_u = cfg.lam_u
        self.lam_v = cfg.lam_v
        # NOTE: lam_* are dimensionless mode counts (used inside sine/cosine arguments).

        # Amplitude parameters
        self.A_u = cfg.A_u  # Amplitude for u
        self.A_v = cfg.A_v  # Amplitude for v
        
        # Argument scalling in trigonometric functions  [sic: "scalling" → spelling note only]
        self.d_u = self.lam_u * np.pi / self.cfg.ell  # precompute spatial frequency for u
        self.d_v = self.lam_v * np.pi / self.cfg.ell  # precompute spatial frequency for v
        # NOTE: d_* = (lambda*pi)/ell makes sin(d_* x) an integer number of half-waves over [0, ell].

        super().__init__(cfg)        # Initialize symbolic setup from parent class
        self._prepare_data()         # Precompute symbolic forcing and boundary data
        # IMPORTANT: _prepare_data() is assumed to set up test fixtures (forcing, BCs) for the framework.

    # ------------------------------------------------------
    # TRIGONOMETRIC MULTIPLIERS FOR MODAL STRUCTURE
    # ------------------------------------------------------

    # ---------- METHOD: sin_u ----------
    def sin_u(self, x):
        """Sine wave spatial profile for displacement u(x)."""
        x = np.asarray(x)  # ensure input is array-like for broadcasting
        # sin(d_u * x) where d_u = lam_u * pi / ell
        # Imposes a standing-wave-like spatial mode for u.
        return np.sin(self.d_u * x)

    # ---------- METHOD: sin_v ----------
    def sin_v(self, x):
        """Sine wave spatial profile for rotation v(x)."""
        x = np.asarray(x)  # ensure input is array-like for broadcasting
        # sin(d_v * x) where d_v = lam_v * pi / ell
        # Independent mode count for v allows phase/mode mismatches for robustness tests.
        return np.sin(self.d_v * x)
    
    # ---------- METHOD: cos_u ----------
    def cos_u(self, x):
        """Cosine wave spatial profile for displacement u(x)."""
        x = np.asarray(x)  # ensure input is array-like for broadcasting
        # cos(d_u * x)
        # Used when taking spatial derivatives of sine terms.
        return np.cos(self.d_u * x)

    # ---------- METHOD: cos_v ----------
    def cos_v(self, x):
        """Cosine wave spatial profile for rotation v(x)."""
        x = np.asarray(x)  # ensure input is array-like for broadcasting
        # cos(d_v * x)
        # Used when taking spatial derivatives of sine terms.
        return np.cos(self.d_v * x)

    # ------------------------------------------------------
    # INITIAL CONDITIONS (u, v) AT t = 0
    # ------------------------------------------------------

    # ---------- METHOD: varphi0 ----------
    def varphi0(self, x):
        """Initial displacement u(x, 0) = amplitude * sine."""
        x = np.asarray(x)  # ensure input is array-like for broadcasting
        # u(x,0) = A_u * sin(d_u * x)
        return self.A_u * self.sin_u(x)

    # ---------- METHOD: psi0 ----------
    def psi0(self, x):
        """Initial rotation v(x, 0) = amplitude * sine."""
        x = np.asarray(x)  # ensure input is array-like for broadcasting
        # v(x,0) = A_v * sin(d_v * x)
        return self.A_v * self.sin_v(x)

    # ------------------------------------------------------
    # FIRST SPATIAL DERIVATIVES AT t = 0
    # ------------------------------------------------------

    # ---------- METHOD: d1varphi0 ----------
    def d1varphi0(self, x):
        """∂u/∂x at t = 0."""
        x = np.asarray(x)  # ensure input is array-like for broadcasting
        # (A_u * sin(d_u x))' = A_u * d_u * cos(d_u x)
        return self.A_u * self.d_u * self.cos_u(x)

    # ---------- METHOD: d1psi0 ----------
    def d1psi0(self, x):
        """∂v/∂x at t = 0."""
        x = np.asarray(x)  # ensure input is array-like for broadcasting
        # (A_v * sin(d_v x))' = A_v * d_v * cos(d_v x)
        return self.A_v * self.d_v * self.cos_v(x)

    # ------------------------------------------------------
    # SECOND SPATIAL DERIVATIVES AT t = 0
    # ------------------------------------------------------

    # ---------- METHOD: d2varphi0 ----------
    def d2varphi0(self, x):
        """∂²u/∂x² at t = 0."""
        x = np.asarray(x)  # ensure input is array-like for broadcasting
        # (A_u * sin(d_u x))'' = -A_u * d_u^2 * sin(d_u x)
        return (- self.A_u * self.d_u**2 * self.sin_u(x))

    # ---------- METHOD: d2psi0 ----------
    def d2psi0(self, x):
        """∂²v/∂x² at t = 0."""
        x = np.asarray(x)  # ensure input is array-like for broadcasting
        # (A_v * sin(d_v x))'' = -A_v * d_v^2 * sin(d_v x)
        return (- self.A_v * self.d_v**2 * self.sin_v(x))

    # ------------------------------------------------------
    # THIRD SPATIAL DERIVATIVES AT t = 0
    # ------------------------------------------------------

    # ---------- METHOD: d3varphi0 ----------
    def d3varphi0(self, x):
        """∂³u/∂x³ at t = 0."""
        x = np.asarray(x)  # ensure input is array-like for broadcasting
        # (A_u * sin(d_u x))''' = -A_u * d_u^3 * cos(d_u x)
        return (- self.A_u * self.d_u**3 * self.cos_u(x))

    # ---------- METHOD: d3psi0 ----------
    def d3psi0(self, x):
        """∂³v/∂x³ at t = 0."""
        x = np.asarray(x)  # ensure input is array-like for broadcasting
        # (A_v * sin(d_v x))''' = -A_v * d_v^3 * cos(d_v x)
        return (- self.A_v * self.d_v**3 * self.cos_v(x))

    # ------------------------------------------------------
    # TIME DERIVATIVES (STATIC CASE)
    # ------------------------------------------------------

    # ---------- METHOD: varphi1 ----------
    def varphi1(self, x):
        """Initial time derivative ∂u/∂t at t = 0 — set to zero for static initial state."""
        x = np.asarray(x)  # ensure input is array-like for broadcasting
        # Static initial state assumption → zero time derivative
        # NOTE: Returns scalar 0.0 (not zeros_like). This mirrors your original behavior.
        return np.float64(0.0)

    # ---------- METHOD: psi1 ----------
    def psi1(self, x):
        """Initial time derivative ∂v/∂t at t = 0 — set to zero for static initial state."""
        x = np.asarray(x)  # ensure input is array-like for broadcasting
        # Static initial state assumption → zero time derivative
        # NOTE: Returns scalar 0.0 to match original return type.
        return np.float64(0.0)

    # ---------- METHOD: d1varphi1 ----------
    def d1varphi1(self, x):
        """∂/∂x of ∂u/∂t at t = 0 — zero for static initial state."""
        x = np.asarray(x)  # ensure input is array-like for broadcasting
        # Spatial derivative of a zero field remains zero
        # NOTE: Returns scalar 0.0 to preserve your API expectations.
        return np.float64(0.0)

    # ---------- METHOD: d1psi1 ----------
    def d1psi1(self, x):
        """∂/∂x of ∂v/∂t at t = 0 — zero for static initial state."""
        x = np.asarray(x)  # ensure input is array-like for broadcasting
        # Spatial derivative of a zero field remains zero
        # NOTE: Returns scalar 0.0 to preserve your API expectations.
        return np.float64(0.0)

    # ------------------------------------------------------
    # FORCING TERMS (SOURCE FUNCTIONS)
    # ------------------------------------------------------

    # ---------- METHOD: f1 ----------
    def f1(self, x, t):
        """External forcing in the u-equation — zero for manufactured test."""
        x = np.asarray(x)  # ensure input is array-like for broadcasting
        # No external forcing applied in this manufactured case
        # NOTE: Returns scalar 0.0 for consistency with original code.
        return np.float64(0.0)

    # ---------- METHOD: f2 ----------
    def f2(self, x, t):
        """External forcing in the v-equation — zero for manufactured test."""
        x = np.asarray(x)  # ensure input is array-like for broadcasting
        # No external forcing applied in this manufactured case
        # NOTE: Returns scalar 0.0 for consistency with original code.
        return np.float64(0.0)
    
    # ---------- METHOD: d1f1 ----------
    def d1f1(self, x, t):
        """First spatial derivative of u-forcing — zero (forcing is identically zero)."""
        x = np.asarray(x)  # ensure input is array-like for broadcasting
        # Gradient of a zero forcing term is zero
        # NOTE: Returns scalar 0.0 for consistency with original code.
        return np.float64(0.0)

    # ---------- METHOD: d1f2 ----------
    def d1f2(self, x, t):
        """First spatial derivative of v-forcing — zero (forcing is identically zero)."""
        x = np.asarray(x)  # ensure input is array-like for broadcasting
        # Gradient of a zero forcing term is zero
        # NOTE: Returns scalar 0.0 for consistency with original code.
        return np.float64(0.0)

    # ------------------------------------------------------
    # POST INITIALIZATION (OPTIONAL DATACLASS SUPPORT)
    # ------------------------------------------------------

    # ---------- METHOD: __post_init__ ----------
    def __post_init__(self):
        """
        Hook for dataclass compatibility.
        Ensures post-construction preparation is called if needed.
        """
        # If constructed via dataclass semantics, ensure required setup is performed
        self._prepare_data()
        # NOTE: Calling _prepare_data here (in addition to __init__) keeps behavior robust
        #       if dataclass-like construction paths are used by the test framework.