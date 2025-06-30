# ======================================================
# MODULE IMPORTS
# ======================================================

import utils.config as cfg  # Configuration module (contains constants like time step τ, domain length ℓ, material coefficients)
from utils.auxiliary import integrate_derivative_form  # Computes integral-based nonlinear correction term for PDE stiffness


# ======================================================
# TIMOSHENKO BENCHMARK CLASS
# ======================================================

class TimoshenkoSolutions:
    """
    Manages initial and boundary data for the nonlinear Timoshenko beam model.

    Supports:
    ---------
    - Exact symbolic solutions if `known_solutions=True`
    - Approximate Taylor expansions if `known_solutions=False`
    """

    def __init__(self, known_solutions: bool = True):
        """
        Initializes the data setup for Timoshenko beam simulation.

        Parameters
        ----------
        known_solutions : bool
            Whether to use exact symbolic solutions or Taylor approximations.
        """
        self.known_solutions = known_solutions

        # Placeholder attributes for:
        # - f1, f2: source terms (right-hand side)
        # - u0, u1: displacement at t = 0, τ
        # - v0, v1: rotation at t = 0, τ
        # - du0, du1: spatial derivative of u at t = 0, τ
        # - dv0, dv1: spatial derivative of v at t = 0, τ
        # - u, v: exact solution functions if known
        self.f1 = self.f2 = None
        self.u0 = self.u1 = None
        self.v0 = self.v1 = None
        self.du0 = self.du1 = None
        self.dv0 = self.dv1 = None
        self.u = self.v = None

        # Set up data using either exact solutions or Taylor approximation
        self._prepare_data()

    # ------------------------------------------------------
    # TAYLOR EXPANSION HELPER FUNCTION
    # ------------------------------------------------------

    def taylor_expansion(self, tau, f0, f1, f2):
        """
        Builds a second-order Taylor approximation of f(x, t) at t = τ.

        Parameters
        ----------
        tau : float
            Time step Δt (denoted τ).
        f0 : callable
            Function f(x, 0).
        f1 : callable
            First time derivative ∂f/∂t at t = 0.
        f2 : callable
            Second time derivative ∂²f/∂t² at t = 0.

        Returns
        -------
        callable
            Approximated f(x, τ)
        """
        return lambda x: f0(x) + tau * f1(x) + 0.5 * tau**2 * f2(x)

    # ------------------------------------------------------
    # INTERNAL INITIALIZATION METHOD
    # ------------------------------------------------------

    def _prepare_data(self):
        """
        Populates source terms, displacement, rotation fields, and their spatial derivatives
        using either symbolic or approximated (Taylor-expanded) expressions.
        """
        tau = cfg.tau  # Time step size from configuration

        if self.known_solutions:
            # --------------------------------------------------
            # USE KNOWN SYMBOLIC SOLUTIONS (EXACT)
            # --------------------------------------------------
            import utils.case_known_solns as ks

            # Exact solutions for u(x,t), v(x,t)
            self.u = lambda x, t: ks.u(x, t)
            self.v = lambda x, t: ks.v(x, t)

            # Source terms
            self.f1 = ks.f1
            self.f2 = ks.f2

            # Displacement and rotation at t = 0 and τ
            self.u0 = lambda x: self.u(x, 0)
            self.u1 = lambda x: self.u(x, tau)
            self.v0 = lambda x: self.v(x, 0)
            self.v1 = lambda x: self.v(x, tau)

            # First spatial derivatives ∂u/∂x and ∂v/∂x
            self.du0 = lambda x: ks.diff1x_u(x, 0)
            self.du1 = lambda x: ks.diff1x_u(x, tau)
            self.dv0 = lambda x: ks.diff1x_v(x, 0)
            self.dv1 = lambda x: ks.diff1x_v(x, tau)

        else:
            # --------------------------------------------------
            # TAYLOR APPROXIMATIONS BASED ON INITIAL DATA
            # --------------------------------------------------
            import utils.case_unk_soln as us

            # Initial conditions for displacement and rotation
            self.u0 = lambda x: us.lambdified_derivatives['varphi0'](x)
            self.v0 = lambda x: us.lambdified_derivatives['psi0'](x)

            # Source terms as time-dependent functions
            self.f1 = lambda x, t: us.lambdified_derivatives['f1'](x, t)
            self.f2 = lambda x, t: us.lambdified_derivatives['f2'](x, t)

            # First-order time derivatives at t = 0
            varphi1 = lambda x: us.lambdified_derivatives['varphi1'](x)
            psi1 = lambda x: us.lambdified_derivatives['psi1'](x)
            dvarphi1 = lambda x: us.lambdified_derivatives['d1varphi1'](x)
            dpsi1 = lambda x: us.lambdified_derivatives['d1psi1'](x)

            # First-order spatial derivatives at t = 0
            self.du0 = lambda x: us.lambdified_derivatives['d1varphi0'](x)
            self.dv0 = lambda x: us.lambdified_derivatives['d1psi0'](x)

            # Higher-order spatial derivatives for second time derivative
            d2varphi0 = lambda x: us.lambdified_derivatives['d2varphi0'](x)
            d2psi0 = lambda x: us.lambdified_derivatives['d2psi0'](x)
            d3varphi0 = lambda x: us.lambdified_derivatives['d3varphi0'](x)
            d3psi0 = lambda x: us.lambdified_derivatives['d3psi0'](x)

            # Compute nonlinear stiffness correction via integration
            nonlinear_term, *_ = integrate_derivative_form(df=self.du0, ell=cfg.ell)

            # Second-order time derivatives (∂²u/∂t², ∂²v/∂t²) using the Timoshenko PDEs
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

            # Compute Taylor-approximated displacement and rotation at t = τ
            self.u1 = self.taylor_expansion(tau, self.u0, varphi1, varphi2)
            self.v1 = self.taylor_expansion(tau, self.v0, psi1, psi2)

            # First spatial derivatives of time-derivatives at t = 0
            d1f1 = lambda x, t: us.lambdified_derivatives['d1f1'](x, t)
            d1f2 = lambda x, t: us.lambdified_derivatives['d1f2'](x, t)

            # Second-order time derivatives of spatial derivatives
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

            # Taylor-expanded spatial derivatives at t = τ
            self.du1 = self.taylor_expansion(tau, self.du0, dvarphi1, dvarphi2)
            self.dv1 = self.taylor_expansion(tau, self.dv0, dpsi1, dpsi2)

    # ------------------------------------------------------
    # PUBLIC INTERFACE
    # ------------------------------------------------------

    def get_initial_data(self):
        """
        Returns all initial and second-time-step data required to solve the PDE.

        Returns
        -------
        tuple of callables:
            f1, f2   : Source terms f₁(x, t), f₂(x, t)
            u0, u1   : Displacement u(x, 0), u(x, τ)
            v0, v1   : Rotation     v(x, 0), v(x, τ)
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