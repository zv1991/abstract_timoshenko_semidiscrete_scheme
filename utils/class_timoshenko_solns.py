# ======================================================
# MODULE IMPORTS
# ======================================================

import utils.config as cfg                               # Global configuration (e.g., domain length, time step, material constants)
from utils.auxiliary import integrate_derivative_form    # Utility function for integral-based nonlinearity in PDE


# ======================================================
# TIMOSHENKO BENCHMARK CLASS
# ======================================================

class TimoshenkoSolutions:
    """
    Class for managing benchmark initial conditions and source terms
    for the nonlinear Timoshenko beam model.

    Supports:
    ---------
    - Exact symbolic solutions (if `known_solutions=True`)
    - Taylor-approximated solutions (if `known_solutions=False`)
    """

    def __init__(self, known_solutions: bool = True):
        """
        Constructor that sets up initial displacement, rotation, source terms,
        and their spatial derivatives based on whether exact solutions are known.

        Parameters
        ----------
        known_solutions : bool
            If True, use predefined symbolic solutions; otherwise, approximate using Taylor expansion.
        """
        self.known_solutions = known_solutions

        # Initialize placeholders for solution components
        self.f1 = self.f2 = None         # Source terms
        self.u0 = self.u1 = None         # Displacement at t = 0 and t = τ
        self.v0 = self.v1 = None         # Rotation at t = 0 and t = τ
        self.du0 = self.du1 = None       # ∂u/∂x at t = 0 and t = τ
        self.dv0 = self.dv1 = None       # ∂v/∂x at t = 0 and t = τ
        self.u = self.v = None           # Exact u(x,t), v(x,t) (if available)

        # Prepare data based on the setting
        self._prepare_data()

    # ------------------------------------------------------
    # TAYLOR EXPANSION HELPER
    # ------------------------------------------------------

    def taylor_expansion(self, tau, f0, f1, f2):
        """
        Returns a second-order Taylor expansion of f(x, t) evaluated at t = τ.

        Parameters
        ----------
        tau : float
            Time step size.
        f0 : callable
            Function value at t = 0.
        f1 : callable
            First time derivative at t = 0.
        f2 : callable
            Second time derivative at t = 0.

        Returns
        -------
        callable
            Function approximating f(x, τ)
        """
        return lambda x: f0(x) + tau * f1(x) + 0.5 * tau**2 * f2(x)

    # ------------------------------------------------------
    # INTERNAL INITIALIZATION
    # ------------------------------------------------------

    def _prepare_data(self):
        """
        Sets up source terms, displacement, rotation, and their derivatives
        either via exact solutions or Taylor approximations.
        """
        tau = cfg.tau  # Time step size

        if self.known_solutions:
            import utils.case_known_solns as ks

            # Exact symbolic solutions u(x,t), v(x,t)
            self.u = lambda x, t: ks.u(x, t)
            self.v = lambda x, t: ks.v(x, t)

            self.f1 = ks.f1
            self.f2 = ks.f2

            self.u0 = lambda x: self.u(x, 0)
            self.u1 = lambda x: self.u(x, tau)
            self.v0 = lambda x: self.v(x, 0)
            self.v1 = lambda x: self.v(x, tau)

            self.du0 = lambda x: ks.diff1x_u(x, 0)
            self.du1 = lambda x: ks.diff1x_u(x, tau)
            self.dv0 = lambda x: ks.diff1x_v(x, 0)
            self.dv1 = lambda x: ks.diff1x_v(x, tau)

        else:
            import utils.case_unk_soln as us

            # Basic fields at t=0 from symbolic definitions
            self.u0 = lambda x: us.lambdified_derivatives['varphi0'](x)
            self.v0 = lambda x: us.lambdified_derivatives['psi0'](x)

            self.f1 = lambda x, t: us.lambdified_derivatives['f1'](x, t)
            self.f2 = lambda x, t: us.lambdified_derivatives['f2'](x, t)

            d1f1 = lambda x, t: us.lambdified_derivatives['d1f1'](x, t)
            d1f2 = lambda x, t: us.lambdified_derivatives['d1f2'](x, t)

            # Time derivatives at t=0
            varphi1 = lambda x: us.lambdified_derivatives['varphi1'](x)
            psi1    = lambda x: us.lambdified_derivatives['psi1'](x)
            dvarphi1 = lambda x: us.lambdified_derivatives['d1varphi1'](x)
            dpsi1    = lambda x: us.lambdified_derivatives['d1psi1'](x)

            # Spatial derivatives at t=0
            self.du0 = lambda x: us.lambdified_derivatives['d1varphi0'](x)
            self.dv0 = lambda x: us.lambdified_derivatives['d1psi0'](x)

            d2varphi0 = lambda x: us.lambdified_derivatives['d2varphi0'](x)
            d2psi0    = lambda x: us.lambdified_derivatives['d2psi0'](x)
            d3varphi0 = lambda x: us.lambdified_derivatives['d3varphi0'](x)
            d3psi0    = lambda x: us.lambdified_derivatives['d3psi0'](x)

            # Nonlinear term for effective stiffness (via integral)
            nonlinear_term, *_ = integrate_derivative_form(df=self.du0, ell=cfg.ell)

            # Second-order time derivatives from governing PDE
            varphi2 = lambda x: (
                self.f1(x, 0)
                - cfg.a1 * self.dv0(x)
                + (cfg.alpha + cfg.beta * nonlinear_term) * d2varphi0(x)
            )
            psi2 = lambda x: (
                self.f2(x, 0)
                + cfg.a2 * self.du0(x)
                + cfg.gamma * d2psi0(x)
                - cfg.delta * self.v0(x)
            )

            # Apply second-order Taylor expansion for t = τ
            self.u1 = self.taylor_expansion(tau, self.u0, varphi1, varphi2)
            self.v1 = self.taylor_expansion(tau, self.v0, psi1, psi2)

            # First-order spatial derivatives at t = τ
            dvarphi2 = lambda x: (
                d1f1(x, 0)
                - cfg.a1 * d2psi0(x)
                + (cfg.alpha + cfg.beta * nonlinear_term) * d3varphi0(x)
            )
            dpsi2 = lambda x: (
                d1f2(x, 0)
                + cfg.a2 * d2varphi0(x)
                + cfg.gamma * d3psi0(x)
                - cfg.delta * self.dv0(x)
            )

            self.du1 = self.taylor_expansion(tau, self.du0, dvarphi1, dvarphi2)
            self.dv1 = self.taylor_expansion(tau, self.dv0, dpsi1, dpsi2)

    # ------------------------------------------------------
    # PUBLIC INTERFACE
    # ------------------------------------------------------

    def get_initial_data(self):
        """
        Returns:
        --------
        tuple of callables:
            f1, f2 : RHS source terms
            u0, u1 : displacement at t = 0, τ
            v0, v1 : rotation at t = 0, τ
            du0, du1 : ∂u/∂x at t = 0, τ
            dv0, dv1 : ∂v/∂x at t = 0, τ
        """
        return (
            self.f1, self.f2,
            self.u0, self.u1,
            self.v0, self.v1,
            self.du0, self.du1,
            self.dv0, self.dv1
        )