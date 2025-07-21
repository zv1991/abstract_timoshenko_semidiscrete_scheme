# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # Provides support for numerical operations and array processing
from numpy.polynomial.legendre import legval, legder  # Efficient Legendre polynomial evaluation and differentiation
import utils.auxiliary as aux  # Custom utilities: basis functions, normalization, integration, etc.
from tests.timoshenko_data import TimoshenkoTesterParent  # Base class for symbolic test case templates


# ======================================================
# CLASS: Testcase1 – Analytical Benchmark for Timoshenko System
# ======================================================

class Testcase1(TimoshenkoTesterParent):
    """
    Symbolic benchmark test case for the nonlinear Timoshenko beam model.

    This class defines closed-form expressions for displacement (u) and 
    rotation (v), including spatial and temporal derivatives. It's used
    to validate numerical solvers via exact analytical comparison.
    """

    # ==================================================
    # CONSTRUCTOR
    # ==================================================
    def __init__(self, cfg):
        """
        Initialize the symbolic benchmark using an external configuration object.

        Parameters
        ----------
        cfg : object
            Beam configuration object with attributes: 
            tau, ell, alpha, beta, gamma, delta, a1, a2
        """
        self.cfg = cfg
        self.name = "test1"           # Identifier for output tracking
        self.known_solutions = True   # Flags that this test includes exact solutions
        super().__init__(cfg)         # Pass configuration to parent initializer
        self._prepare_data()          # Trigger any custom setup logic

    # ==================================================
    # METHOD: Derivative of Normalized Shifted Legendre Polynomial
    # ==================================================
    def derivative_norm_shifted_legendre(self, m: int, ell: float, x: float | np.ndarray) -> float | np.ndarray:
        """
        Computes the first derivative of the normalized shifted Legendre polynomial P̂_m(x):

            d/dx [P̂_m(x)] = (2 / (ell * A_m * sqrt(ell))) * P'_m((2x / ell) - 1)

        This avoids summation by using the analytical derivative of the standard 
        Legendre polynomial and applying the chain rule for the shift.

        Parameters
        ----------
        m : int
            Degree of the polynomial
        ell : float
            Beam length or domain scaling
        x : float or np.ndarray
            Evaluation point(s)

        Returns
        -------
        float or np.ndarray
            First derivative of normalized shifted Legendre polynomial at x
        """
        z = (2 * x / ell) - 1  # Rescale x to [-1, 1] domain
        coeffs = np.zeros(m + 1)
        coeffs[m] = 1  # P_m(z)

        dcoeffs = legder(coeffs)  # Compute derivative coefficients
        Pm_prime_z = legval(z, dcoeffs)  # Evaluate at z

        A_m = aux.coeff_A[m]  # Normalization factor
        return (2 / (ell * A_m * np.sqrt(ell))) * Pm_prime_z

    # ==================================================
    # BASIS FUNCTION DEGREE CONFIGURATION
    # ==================================================
    m_u = 2  # Degree of polynomial for displacement
    m_v = 2  # Degree of polynomial for rotation

    # ==================================================
    # SPATIAL BASIS FUNCTIONS FOR DISPLACEMENT u(x)
    # ==================================================
    def h_u(self, x): 
        return aux.phi_m(self.m_u, self.cfg.ell, x)

    def d1h_u(self, x): 
        return aux.normalized_shifted_legendre(self.m_u, self.cfg.ell, x)

    def d2h_u(self, x): 
        return self.derivative_norm_shifted_legendre(self.m_u, self.cfg.ell, x)

    # ==================================================
    # SPATIAL BASIS FUNCTIONS FOR ROTATION v(x)
    # ==================================================
    def h_v(self, x): 
        return aux.phi_m(self.m_v, self.cfg.ell, x)

    def d1h_v(self, x): 
        return aux.normalized_shifted_legendre(self.m_v, self.cfg.ell, x)

    def d2h_v(self, x): 
        return self.derivative_norm_shifted_legendre(self.m_v, self.cfg.ell, x)

    # ==================================================
    # TEMPORAL BASIS FUNCTIONS FOR u(t) AND v(t)
    # ==================================================
    def g_u(self, t): return t
    def d1g_u(self, t): return np.float64(0.0)
    def d2g_u(self, t): return np.float64(0.0)

    def g_v(self, t): return t
    def d1g_v(self, t): return np.float64(0.0)
    def d2g_v(self, t): return np.float64(0.0)

    # ==================================================
    # EXACT SOLUTIONS: u(x, t) and v(x, t)
    # ==================================================
    def u(self, x, t): return self.h_u(x) * self.g_u(t)
    def v(self, x, t): return self.h_v(x) * self.g_v(t)

    # ==================================================
    # DERIVATIVES OF u(x, t)
    # ==================================================
    def diff1t_u(self, x, t): return self.h_u(x) * self.d1g_u(t)
    def diff2t_u(self, x, t): return self.h_u(x) * self.d2g_u(t)
    def diff1x_u(self, x, t): return self.d1h_u(x) * self.g_u(t)
    def diff2x_u(self, x, t): return self.d2h_u(x) * self.g_u(t)

    # ==================================================
    # DERIVATIVES OF v(x, t)
    # ==================================================
    def diff1t_v(self, x, t): return self.h_v(x) * self.d1g_v(t)
    def diff2t_v(self, x, t): return self.h_v(x) * self.d2g_v(t)
    def diff1x_v(self, x, t): return self.d1h_v(x) * self.g_v(t)
    def diff2x_v(self, x, t): return self.d2h_v(x) * self.g_v(t)

    # ==================================================
    # NONLINEAR TERM: ∫(∂u/∂x)² dx
    # ==================================================
    def integr_term(self, t):
        """
        Computes the nonlinear energy-like term used in f₁:

            ∫ (∂u/∂x)² dx

        Parameters
        ----------
        t : float
            Time at which to evaluate the spatial derivative

        Returns
        -------
        float
            Integral result at time t
        """
        integrand = lambda x: self.diff1x_u(x, t)
        result, *_ = aux.integrate_derivative_form(df=integrand, ell=self.cfg.ell)
        return result

    # ==================================================
    # RIGHT-HAND SIDE FUNCTION: f₁(x, t) – Displacement Equation
    # ==================================================
    def f1(self, x, t):
        """
        Evaluates the right-hand side of the displacement PDE:

            f₁(x, t) = ∂²u/∂t² - (α + β ∫(∂u/∂x)² dx) ∂²u/∂x² + a₁ ∂v/∂x

        Parameters
        ----------
        x : float
        t : float

        Returns
        -------
        float
            Source term value at (x, t)
        """
        return (
            self.diff2t_u(x, t)
            - (self.cfg.alpha + self.cfg.beta * self.integr_term(t)) * self.diff2x_u(x, t)
            + self.cfg.a1 * self.diff1x_v(x, t)
        )

    # ==================================================
    # RIGHT-HAND SIDE FUNCTION: f₂(x, t) – Rotation Equation
    # ==================================================
    def f2(self, x, t):
        """
        Evaluates the right-hand side of the rotation PDE:

            f₂(x, t) = ∂²v/∂t² - γ ∂²v/∂x² + δ v - a₂ ∂u/∂x

        Parameters
        ----------
        x : float
        t : float

        Returns
        -------
        float
            Source term value at (x, t)
        """
        return (
            self.diff2t_v(x, t)
            - self.cfg.gamma * self.diff2x_v(x, t)
            + self.cfg.delta * self.v(x, t)
            - self.cfg.a2 * self.diff1x_u(x, t)
        )

    # ==================================================
    # POST-INITIALIZATION HOOK (For dataclass compatibility)
    # ==================================================
    def __post_init__(self):
        """
        Called automatically after class construction for dataclass compatibility.
        Ensures necessary derived fields are initialized.
        """
        self._prepare_data()