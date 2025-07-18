# ======================================================
# MODULE IMPORTS
# ======================================================

import setting.config as cfg  # Configuration constants, including:
#   - τ (time step)
#   - ℓ (beam length)
#   - Material coefficients (α, β, γ, δ, a₁, a₂)

from utils.auxiliary import integrate_derivative_form  # Computes the integral ∫(∂u/∂x)² dx
# Used for computing the nonlinear stiffness correction term in the beam equation


# ======================================================
# TIMOSHENKO BENCHMARK BASE CLASS
# ======================================================

class TimoshenkoTesterParent:
    """
    Base class to manage initial and boundary conditions for the nonlinear Timoshenko beam model.

    Two supported modes:
    ---------------------
    1. Exact symbolic solutions (analytical mode) if `self.known_solutions = True`
    2. Approximate Taylor-expanded data if `self.known_solutions = False`
    """

    # ------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------
    def __init__(self):
        """
        Constructor does not initialize data directly.
        Derived classes must:
            - Set `self.known_solutions`
            - Call `_prepare_data()` to populate required fields
        """
        pass

    # ------------------------------------------------------
    # METHOD: Second-Order Taylor Expansion
    # ------------------------------------------------------
    def taylor_expansion(self, tau, f0, f1, f2):
        """
        Constructs a second-order Taylor approximation of a function f(t, x) at time t = τ.

        Parameters
        ----------
        tau : float
            Time step size τ
        f0 : callable
            f(x, t=0)
        f1 : callable
            ∂f/∂t(x, t=0)
        f2 : callable
            ∂²f/∂t²(x, t=0)

        Returns
        -------
        callable
            Approximated value f(x, t=τ)
        """
        return lambda x: f0(x) + tau * f1(x) + 0.5 * tau**2 * f2(x)

    # ------------------------------------------------------
    # METHOD: Prepare Initial and Time-Shifted Data
    # ------------------------------------------------------
    def _prepare_data(self):
        """
        Prepares the problem data at t = 0 and t = τ based on availability of exact solutions.

        Behavior splits by mode:
            - If exact solutions are known: evaluate directly
            - Otherwise: construct using second-order Taylor expansions
        """
        tau = cfg.tau  # Load τ (time step) from config

        if self.known_solutions:
            # ==============================================
            # MODE 1: Use known symbolic solutions
            # ==============================================

            # Source terms (e.g., external forces or body loads)
            self.f1 = self.f1
            self.f2 = self.f2

            # Displacement (u) and rotation (v) at t = 0 and t = τ
            self.u0 = lambda x: self.u(x, 0)
            self.u1 = lambda x: self.u(x, tau)
            self.v0 = lambda x: self.v(x, 0)
            self.v1 = lambda x: self.v(x, tau)

            # First spatial derivatives (∂u/∂x, ∂v/∂x) at t = 0 and t = τ
            self.du0 = lambda x: self.diff1x_u(x, 0)
            self.du1 = lambda x: self.diff1x_u(x, tau)
            self.dv0 = lambda x: self.diff1x_v(x, 0)
            self.dv1 = lambda x: self.diff1x_v(x, tau)

        else:
            # ==============================================
            # MODE 2: Use Taylor expansion approximations
            # ==============================================

            # Initial conditions (displacement and rotation)
            self.u0 = lambda x: self.varphi0(x)
            self.v0 = lambda x: self.psi0(x)

            # Source terms (must be functions of both x and t)
            self.f1 = lambda x, t: self.f1(x, t)
            self.f2 = lambda x, t: self.f2(x, t)

            # First time derivatives at t = 0
            varphi1 = lambda x: self.varphi1(x)
            psi1    = lambda x: self.psi1(x)

            # First spatial derivatives at t = 0
            self.du0 = lambda x: self.d1varphi0(x)
            self.dv0 = lambda x: self.d1psi0(x)

            # First spatial derivatives of time-derivatives
            dvarphi1 = lambda x: self.d1varphi1(x)
            dpsi1    = lambda x: self.d1psi1(x)

            # Higher-order spatial derivatives at t = 0
            d2varphi0 = lambda x: self.d2varphi0(x)
            d2psi0    = lambda x: self.d2psi0(x)
            d3varphi0 = lambda x: self.d3varphi0(x)
            d3psi0    = lambda x: self.d3psi0(x)

            # Compute nonlinear stiffness term: ∫ (∂u/∂x)² dx
            nonlinear_term, *_ = integrate_derivative_form(df=self.du0, ell=cfg.ell)

            # Compute second time derivatives using the PDE system at t = 0
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

            # Use Taylor expansion to compute u and v at t = τ
            self.u1 = self.taylor_expansion(tau, self.u0, varphi1, varphi2)
            self.v1 = self.taylor_expansion(tau, self.v0, psi1, psi2)

            # Derivatives of source terms for computing second-time derivatives of ∂u/∂x, ∂v/∂x
            d1f1 = lambda x, t: self.d1f1(x, t)
            d1f2 = lambda x, t: self.d1f2(x, t)

            # Compute second time derivatives of spatial derivatives
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

            # Use Taylor expansion to compute ∂u/∂x and ∂v/∂x at t = τ
            self.du1 = self.taylor_expansion(tau, self.du0, dvarphi1, dvarphi2)
            self.dv1 = self.taylor_expansion(tau, self.dv0, dpsi1, dpsi2)

    # ------------------------------------------------------
    # METHOD: Retrieve Initialized Data Fields
    # ------------------------------------------------------
    def get_initial_data(self):
        """
        Accessor method to return initial and first-step data (at t = 0 and t = τ).

        Returns
        -------
        tuple :
            (
                f1, f2,       # Source functions
                u0, u1,       # Displacement at t = 0 and τ
                v0, v1,       # Rotation at t = 0 and τ
                du0, du1,     # ∂u/∂x at t = 0 and τ
                dv0, dv1      # ∂v/∂x at t = 0 and τ
            )
        """
        return (
            self.f1, self.f2,
            self.u0, self.u1,
            self.v0, self.v1,
            self.du0, self.du1,
            self.dv0, self.dv1
        )