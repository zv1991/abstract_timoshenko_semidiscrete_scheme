# ======================================================
# MODULE IMPORTS
# ======================================================

from utils.auxiliary import integrate_derivative_form  # Computes ∫(∂u/∂x)² dx for nonlinear stiffness correction


# ======================================================
# TIMOSHENKO BENCHMARK BASE CLASS (CONFIG-INDEPENDENT)
# ======================================================

class TimoshenkoTesterParent:
    """
    Abstract base class for initializing data in the nonlinear Timoshenko beam model.

    This class supports two modes:
    ------------------------------
    1. Symbolic mode (with known analytical solutions)
    2. Numerical mode (using Taylor-expansion initial conditions)
    """

    # ------------------------------------------------------
    # CONSTRUCTOR: Inject configuration and initialize
    # ------------------------------------------------------
    def __init__(self, cfg):
        """
        Initialize with a config object containing material and solver parameters.

        Parameters
        ----------
        cfg : object
            Configuration object with attributes:
            - tau, ell: Time step and beam length
            - alpha, beta, gamma, delta: Physical constants
            - a1, a2: Coupling coefficients
        """
        self.cfg = cfg  # Store config for later use

    # ------------------------------------------------------
    # UTILITY: Second-order Taylor expansion of a function
    # ------------------------------------------------------
    def taylor_expansion(self, tau, f0, f1, f2):
        """
        Constructs a second-order Taylor approximation of a time-evolved function f(x, t).

        Parameters
        ----------
        tau : float
            Time step τ
        f0 : callable
            f(x, 0) - value at initial time
        f1 : callable
            ∂f/∂t at t = 0
        f2 : callable
            ∂²f/∂t² at t = 0

        Returns
        -------
        callable
            Approximated function f(x, τ)
        """
        return lambda x: f0(x) + tau * f1(x) + 0.5 * tau**2 * f2(x)

    # ------------------------------------------------------
    # CORE: Prepare initial data and derivatives
    # ------------------------------------------------------
    def _prepare_data(self):
        """
        Precomputes u, v, ∂u/∂x, ∂v/∂x at t = 0 and t = τ,
        along with source term components, depending on the mode.
        """
        tau = self.cfg.tau  # Time step from config

        if self.known_solutions:
            # =============================================
            # MODE 1: Use exact symbolic solution
            # =============================================

            # Expect subclass to provide `source_terms` method
            if hasattr(self, "source_terms"):
                terms = self.source_terms()
                self.f1 = terms["f1"]  # f₁ as list of symbolic components
                self.f2 = terms["f2"]  # f₂ as list of symbolic components
            else:
                raise NotImplementedError(
                    "`source_terms()` must be implemented in subclass when `known_solutions = True`"
                )

            # Displacement and rotation at t = 0 and t = τ
            self.u0 = lambda x: self.u(x, 0)
            self.u1 = lambda x: self.u(x, tau)
            self.v0 = lambda x: self.v(x, 0)
            self.v1 = lambda x: self.v(x, tau)

            # First spatial derivatives at t = 0 and τ
            self.du0 = lambda x: self.diff1x_u(x, 0)
            self.du1 = lambda x: self.diff1x_u(x, tau)
            self.dv0 = lambda x: self.diff1x_v(x, 0)
            self.dv1 = lambda x: self.diff1x_v(x, tau)

        else:
            # =============================================
            # MODE 2: Use Taylor expansion approximation
            # =============================================

            # Initial values
            self.u0 = lambda x: self.varphi0(x)
            self.v0 = lambda x: self.psi0(x)

            # First-order time derivatives
            varphi1 = lambda x: self.varphi1(x)
            psi1    = lambda x: self.psi1(x)

            # Spatial derivatives at t = 0
            self.du0 = lambda x: self.d1varphi0(x)
            self.dv0 = lambda x: self.d1psi0(x)

            # First derivatives of the time-derivative terms
            dvarphi1 = lambda x: self.d1varphi1(x)
            dpsi1    = lambda x: self.d1psi1(x)

            # Higher-order spatial derivatives
            d2varphi0 = lambda x: self.d2varphi0(x)
            d2psi0    = lambda x: self.d2psi0(x)
            d3varphi0 = lambda x: self.d3varphi0(x)
            d3psi0    = lambda x: self.d3psi0(x)
            
            # Source terms
            f1 = lambda x, t: self.f1(x, t)
            f2 = lambda x, t: self.f2(x, t)

            # Compute nonlinear stiffness contribution
            nonlinear_term, *_ = integrate_derivative_form(
                df=self.du0, ell=self.cfg.ell
            )

            # Second-order time derivatives from PDE model
            varphi2 = lambda x: (
                f1(x, 0)
                - self.cfg.a1 * self.dv0(x)
                + (self.cfg.alpha + self.cfg.beta * nonlinear_term) * d2varphi0(x)
            )

            psi2 = lambda x: (
                f2(x, 0)
                + self.cfg.a2 * self.du0(x)
                + self.cfg.gamma * d2psi0(x)
                - self.cfg.delta * self.v0(x)
            )

            # Approximate fields at t = τ using Taylor expansion
            self.u1 = self.taylor_expansion(tau, self.u0, varphi1, varphi2)
            self.v1 = self.taylor_expansion(tau, self.v0, psi1, psi2)

            # Derivatives of source terms for spatial projection
            d1f1 = lambda x, t: self.d1f1(x, t)
            d1f2 = lambda x, t: self.d1f2(x, t)

            # Second time derivatives of ∂u/∂x and ∂v/∂x
            dvarphi2 = lambda x: (
                d1f1(x, 0)
                - self.cfg.a1 * d2psi0(x)
                + (self.cfg.alpha + self.cfg.beta * nonlinear_term) * d3varphi0(x)
            )

            dpsi2 = lambda x: (
                d1f2(x, 0)
                + self.cfg.a2 * d2varphi0(x)
                + self.cfg.gamma * d3psi0(x)
                - self.cfg.delta * self.dv0(x)
            )

            # Taylor expansion for spatial derivatives at t = τ
            self.du1 = self.taylor_expansion(tau, self.du0, dvarphi1, dvarphi2)
            self.dv1 = self.taylor_expansion(tau, self.dv0, dpsi1, dpsi2)

    # ------------------------------------------------------
    # GETTER: Return all initialized symbolic/numeric data
    # ------------------------------------------------------
    def get_initial_data(self):
        """
        Returns all required fields and derivatives for solver input.

        Returns
        -------
        tuple
            (
                f1, f2        : list of source term components (symbolic) or callables
                u0, u1        : displacement at t = 0 and τ
                v0, v1        : rotation at t = 0 and τ
                du0, du1      : ∂u/∂x at t = 0 and τ
                dv0, dv1      : ∂v/∂x at t = 0 and τ
            )
        """
        return (
            self.f1, self.f2,
            self.u0, self.u1,
            self.v0, self.v1,
            self.du0, self.du1,
            self.dv0, self.dv1
        )