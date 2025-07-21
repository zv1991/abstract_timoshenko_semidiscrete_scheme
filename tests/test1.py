# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # Provides numerical array and floating-point operations
from numpy.polynomial.legendre import legval, legder  # For evaluating and differentiating Legendre polynomials
import utils.auxiliary as aux  # Custom utility module: basis functions, integrators, normalizations, etc.
from tests.timoshenko_data import TimoshenkoTesterParent  # Abstract base class for symbolic test cases


# ======================================================
# CLASS: Testcase1 – Analytical Benchmark for Timoshenko System
# ======================================================

class Testcase1(TimoshenkoTesterParent):
    """
    Symbolic benchmark solution for the nonlinear Timoshenko beam model.

    Provides exact functions u(x, t) and v(x, t) and their derivatives,
    along with PDE right-hand sides f₁ and f₂ used to validate solvers.
    """

    # ==================================================
    # CONSTRUCTOR: Initialize benchmark configuration
    # ==================================================
    def __init__(self, cfg):
        """
        Initializes the benchmark case with configuration parameters.

        Parameters
        ----------
        cfg : object
            Configuration object containing model and numerical parameters.
            tau, ell, alpha, beta, gamma, delta, a1, a2, m_u, m_v
        """
        self.cfg = cfg
        self.name = "test1"             # Identifier used in filenames or logs
        self.known_solutions = True     # This case provides analytical solutions
        super().__init__(cfg)           # Call parent class constructor
        self._prepare_data()            # Optional setup logic (inherited or overridden)

        # Store polynomial degrees for u(x) and v(x) basis functions
        self.m_u = cfg.m_u
        self.m_v = cfg.m_v

    # ==================================================
    # METHOD: Derivative of Normalized Shifted Legendre Polynomial
    # ==================================================
    def derivative_norm_shifted_legendre(self, m: int, ell: float, x: float | np.ndarray) -> float | np.ndarray:
        """
        Computes the first derivative of normalized shifted Legendre polynomial 𝑃̂ₘ(x),
        shifted to [0, ell] from standard [-1, 1].

        d/dx 𝑃̂ₘ(x) = (2 / (ell * Aₘ * sqrt(ell))) * dPₘ/dz, where z = (2x / ell) - 1

        Parameters
        ----------
        m : int
            Degree of Legendre polynomial
        ell : float
            Length of spatial domain
        x : float or np.ndarray
            Evaluation points in [0, ell]

        Returns
        -------
        float or np.ndarray
            Derivative evaluated at x
        """
        z = (2 * x / ell) - 1  # Map x from [0, ell] to z ∈ [-1, 1]
        coeffs = np.zeros(m + 1)
        coeffs[m] = 1  # Construct Pₘ(z) as basis

        dcoeffs = legder(coeffs)           # Get coefficients of Pₘ'(z)
        Pm_prime_z = legval(z, dcoeffs)    # Evaluate derivative at z

        A_m = aux.coeff_A[m]  # Normalization factor for orthonormality
        return (2 / (ell * A_m * np.sqrt(ell))) * Pm_prime_z

    # ==================================================
    # METHOD: Spatial Basis for Displacement u(x)
    # ==================================================
    def h_u(self, x): return aux.phi_m(self.m_u, self.cfg.ell, x)
    def d1h_u(self, x): return aux.normalized_shifted_legendre(self.m_u, self.cfg.ell, x)
    def d2h_u(self, x): return self.derivative_norm_shifted_legendre(self.m_u, self.cfg.ell, x)

    # ==================================================
    # METHOD: Spatial Basis for Rotation v(x)
    # ==================================================
    def h_v(self, x): return aux.phi_m(self.m_v, self.cfg.ell, x)
    def d1h_v(self, x): return aux.normalized_shifted_legendre(self.m_v, self.cfg.ell, x)
    def d2h_v(self, x): return self.derivative_norm_shifted_legendre(self.m_v, self.cfg.ell, x)

    # ==================================================
    # METHOD: Temporal Basis for u(t) and v(t)
    # ==================================================
    def g_u(self, t): return t                      # Linear in time
    def d1g_u(self, t): return np.float64(0.0)      # Time-derivative of t is constant (used in weak form)
    def d2g_u(self, t): return np.float64(0.0)      # Second time-derivative of t is zero

    def g_v(self, t): return t
    def d1g_v(self, t): return np.float64(0.0)
    def d2g_v(self, t): return np.float64(0.0)

    # ==================================================
    # METHOD: Exact Solutions u(x, t) and v(x, t)
    # ==================================================
    def u(self, x, t): return self.h_u(x) * self.g_u(t)
    def v(self, x, t): return self.h_v(x) * self.g_v(t)

    # ==================================================
    # METHOD: Derivatives of u(x, t)
    # ==================================================
    def diff1t_u(self, x, t): return self.h_u(x) * self.d1g_u(t)
    def diff2t_u(self, x, t): return self.h_u(x) * self.d2g_u(t)
    def diff1x_u(self, x, t): return self.d1h_u(x) * self.g_u(t)
    def diff2x_u(self, x, t): return self.d2h_u(x) * self.g_u(t)

    # ==================================================
    # METHOD: Derivatives of v(x, t)
    # ==================================================
    def diff1t_v(self, x, t): return self.h_v(x) * self.d1g_v(t)
    def diff2t_v(self, x, t): return self.h_v(x) * self.d2g_v(t)
    def diff1x_v(self, x, t): return self.d1h_v(x) * self.g_v(t)
    def diff2x_v(self, x, t): return self.d2h_v(x) * self.g_v(t)

    # ==================================================
    # METHOD: Nonlinear Term ∫(∂u/∂x)² dx
    # ==================================================
    def integr_term(self, t):
        """
        Computes ∫(∂u/∂x)² dx — used in the nonlinear coefficient of the u-equation.

        Parameters
        ----------
        t : float
            Time instance

        Returns
        -------
        float
            Spatial integral value
        """
        integrand = lambda x: self.diff1x_u(x, t)
        result, *_ = aux.integrate_derivative_form(df=integrand, ell=self.cfg.ell)
        return result

    # ==================================================
    # METHOD: f1(x, t) – RHS of Displacement Equation
    # ==================================================
    def f1(self, x, t):
        """
        Right-hand side of the displacement PDE:

        f₁(x, t) = ∂²u/∂t² - (α + β ∫(∂u/∂x)² dx) ∂²u/∂x² + a₁ ∂v/∂x

        Parameters
        ----------
        x : float
        t : float

        Returns
        -------
        float
            Value of RHS at point (x, t)
        """
        return (
            self.diff2t_u(x, t)
            - (self.cfg.alpha + self.cfg.beta * self.integr_term(t)) * self.diff2x_u(x, t)
            + self.cfg.a1 * self.diff1x_v(x, t)
        )

    # ==================================================
    # METHOD: f2(x, t) – RHS of Rotation Equation
    # ==================================================
    def f2(self, x, t):
        """
        Right-hand side of the rotation PDE:

        f₂(x, t) = ∂²v/∂t² - γ ∂²v/∂x² + δ v - a₂ ∂u/∂x

        Parameters
        ----------
        x : float
        t : float

        Returns
        -------
        float
            Value of RHS at point (x, t)
        """
        return (
            self.diff2t_v(x, t)
            - self.cfg.gamma * self.diff2x_v(x, t)
            + self.cfg.delta * self.v(x, t)
            - self.cfg.a2 * self.diff1x_u(x, t)
        )

    # ==================================================
    # METHOD: Post Initialization Hook
    # ==================================================
    def __post_init__(self):
        """
        Optional dataclass-compatible hook to re-invoke custom setup.
        """
        self._prepare_data()