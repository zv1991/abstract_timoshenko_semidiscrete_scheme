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

    This class supports:
    --------------------
    1. Symbolic (exact) solution benchmarks
    2. Numerical (Taylor-expanded) initial conditions
    """

    # ------------------------------------------------------
    # METHOD: Constructor with configuration injection
    # ------------------------------------------------------
    def __init__(self, cfg):
        """
        Initialize with a config object containing material and solver parameters.

        Parameters
        ----------
        cfg : object
            Configuration namespace with fields:
            - tau, ell: Time step and beam length
            - alpha, beta, gamma, delta: Physical constants
            - a1, a2: Coupling coefficients
        """
        self.cfg = cfg  # Store configuration for later access

    # ------------------------------------------------------
    # METHOD: Second-order Taylor expansion utility
    # ------------------------------------------------------
    def taylor_expansion(self, tau, f0, f1, f2):
        """
        Constructs a second-order Taylor approximation for a time-evolved function f(x, t).

        Parameters
        ----------
        tau : float
            Time step τ.
        f0 : callable
            Value of f(x) at t = 0.
        f1 : callable
            First time derivative ∂f/∂t at t = 0.
        f2 : callable
            Second time derivative ∂²f/∂t² at t = 0.

        Returns
        -------
        callable
            Function approximating f(x, t = τ).
        """
        return lambda x: f0(x) + tau * f1(x) + 0.5 * tau**2 * f2(x)

    # ------------------------------------------------------
    # METHOD: Prepare u, v, ∂u/∂x, ∂v/∂x at t = 0 and t = τ
    # ------------------------------------------------------
    def _prepare_data(self):
        """
        Initializes displacement u and rotation v, and their spatial derivatives,
        at time t = 0 and first time step t = τ.
        """
        tau = self.cfg.tau  # Time step from configuration

        if self.known_solutions:
            # ========================================================
            # MODE 1: Exact symbolic solution available
            # ========================================================

            # Projected integrals must be precomputed in child (subclass)
            self.f1_integr = self.f1_integr
            self.f2_integr = self.f2_integr

            # Displacement and rotation at t = 0 and t = τ
            self.u0 = lambda x: self.u(x, 0)
            self.u1 = lambda x: self.u(x, tau)
            self.v0 = lambda x: self.v(x, 0)
            self.v1 = lambda x: self.v(x, tau)

            # First spatial derivatives
            self.du0 = lambda x: self.diff1x_u(x, 0)
            self.du1 = lambda x: self.diff1x_u(x, tau)
            self.dv0 = lambda x: self.diff1x_v(x, 0)
            self.dv1 = lambda x: self.diff1x_v(x, tau)

        else:
            # ========================================================
            # MODE 2: No symbolic solution — use Taylor expansion
            # ========================================================

            # Initial values at t = 0
            self.u0 = lambda x: self.varphi0(x)
            self.v0 = lambda x: self.psi0(x)

            # Source terms as functions of (x, t)
            self.f1 = lambda x, t: self.f1(x, t)
            self.f2 = lambda x, t: self.f2(x, t)

            # First time derivatives
            varphi1 = lambda x: self.varphi1(x)
            psi1    = lambda x: self.psi1(x)

            # Spatial derivatives at t = 0
            self.du0 = lambda x: self.d1varphi0(x)
            self.dv0 = lambda x: self.d1psi0(x)

            # First spatial derivatives of time derivatives
            dvarphi1 = lambda x: self.d1varphi1(x)
            dpsi1    = lambda x: self.d1psi1(x)

            # Higher spatial derivatives at t = 0
            d2varphi0 = lambda x: self.d2varphi0(x)
            d2psi0    = lambda x: self.d2psi0(x)
            d3varphi0 = lambda x: self.d3varphi0(x)
            d3psi0    = lambda x: self.d3psi0(x)

            # Compute ∫(∂u/∂x)² dx for nonlinear stiffness contribution
            nonlinear_term, *_ = integrate_derivative_form(
                df=self.du0, ell=self.cfg.ell
            )

            # Compute second time derivatives using the PDE system
            varphi2 = lambda x: (
                self.f1(x, 0)
                - self.cfg.a1 * self.dv0(x)
                + (self.cfg.alpha + self.cfg.beta * nonlinear_term) * d2varphi0(x)
            )

            psi2 = lambda x: (
                self.f2(x, 0)
                + self.cfg.a2 * self.du0(x)
                + self.cfg.gamma * d2psi0(x)
                - self.cfg.delta * self.v0(x)
            )

            # Taylor expansion for u(x, τ) and v(x, τ)
            self.u1 = self.taylor_expansion(tau, self.u0, varphi1, varphi2)
            self.v1 = self.taylor_expansion(tau, self.v0, psi1, psi2)

            # Derivatives of source terms for spatial correction
            d1f1 = lambda x, t: self.d1f1(x, t)
            d1f2 = lambda x, t: self.d1f2(x, t)

            # Compute second time derivatives of spatial derivatives
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

            # Taylor expansion for ∂u/∂x and ∂v/∂x at t = τ
            self.du1 = self.taylor_expansion(tau, self.du0, dvarphi1, dvarphi2)
            self.dv1 = self.taylor_expansion(tau, self.dv0, dpsi1, dpsi2)

    # ------------------------------------------------------
    # METHOD: Retrieve prepared data
    # ------------------------------------------------------
    def get_initial_data(self):
        """
        Collects all required field values and derivatives for use in the solver.

        Returns
        -------
        tuple
            (
                f1, f2        : source terms
                u0, u1        : displacement at t = 0 and τ
                v0, v1        : rotation at t = 0 and τ
                du0, du1      : ∂u/∂x at t = 0 and τ
                dv0, dv1      : ∂v/∂x at t = 0 and τ
            )
        """
        return (
            self.f1_integr, self.f2_integr,
            self.u0, self.u1,
            self.v0, self.v1,
            self.du0, self.du1,
            self.dv0, self.dv1
        )