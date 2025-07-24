# ======================================================
# MODULE IMPORTS
# ======================================================

import numpy as np  # Scientific computing library: arrays, vectorization, numerical methods

import utils.auxiliary as aux  # Utility functions for numerical integration and Galerkin projection

from tests.timoshenko_data import TimoshenkoTesterParent  # Abstract base class for symbolic solution framework


# ======================================================
# TEST CASE 0: ANALYTICAL SOLUTION FOR TIMOSHENKO SYSTEM
# ======================================================

class Testcase0(TimoshenkoTesterParent):
    """
    Symbolic benchmark for nonlinear Timoshenko beam model.
    Defines exact solutions and derives analytic source terms 
    suitable for Galerkin projection.
    """

    # ------------------------------------------------------
    # CONSTRUCTOR: INITIALIZE SYMBOLIC TEST CASE
    # ------------------------------------------------------
    def __init__(self, cfg):
        """
        Initialize and project time-dependent RHS onto Legendre basis.

        Parameters
        ----------
        cfg : object
            Configuration object with:
                - Domain parameters (ell, tau, t)
                - Beam coefficients (alpha, beta, gamma, delta, a1, a2)
                - Polynomial exponents (m1_u, m2_u, m1_v, m2_v)
                - Number of spectral basis functions (N)
                - Quadrature settings (quad_kwargs)
        """
        self.cfg = cfg
        self.name = "test0"
        self.known_solutions = True
        
        # Required before calling _prepare_data
        super().__init__(cfg)

        # Polynomial degrees for u and v
        self.m1_u, self.m2_u = cfg.m1_u, cfg.m2_u
        self.m1_v, self.m2_v = cfg.m1_v, cfg.m2_v

        # Discretization and quadrature settings
        self.N = cfg.N
        self.t = cfg.t
        self.quad_kwargs = cfg.quad_kwargs

        # Retrieve RHS components
        terms = self.source_terms()

        # -------------- Projection: RHS for u-equation (f₁) --------------
        nonlinear_and_coupling = lambda x, t_val: (
            (self.cfg.alpha + self.cfg.beta * self.integr_term(t_val)) * terms["f1"][1](x, t_val)
            - self.cfg.a1 * terms["f1"][2](x, t_val)
        )

        self.f1_integr = (
            aux.compute_time_dependent_integrals(
                terms["f1"][0],  # ∂²u/∂t²
                self.cfg.N,
                self.cfg.ell,
                self.t,
                multiplier="galerkin_basis",
                **self.quad_kwargs
            )
            +
            aux.compute_time_dependent_integrals(
                nonlinear_and_coupling,  # ∂u/∂x term with nonlinearity and -a₁v coupling
                self.cfg.N,
                self.cfg.ell,
                self.t,
                multiplier="norm_leg_poly",
                **self.quad_kwargs
            )
        )

        # -------------- Projection: RHS for v-equation (f₂) --------------
        self.f2_integr = aux.compute_time_dependent_integrals(
            lambda x, t: terms["f2"][0](x, t) + self.cfg.delta * terms["f2"][2](x, t),  # (∂²v/∂t² + δ·v)
            self.cfg.N,
            self.cfg.ell,
            self.t,
            multiplier="galerkin_basis",
            **self.quad_kwargs
        )

        self.f2_integr += aux.compute_time_dependent_integrals(
            lambda x, t: self.cfg.gamma * terms["f2"][1](x, t) + self.cfg.a2 * terms["f2"][3](x, t),  # (γ∂v/∂x + a₂u)
            self.cfg.N,
            self.cfg.ell,
            self.t,
            multiplier="norm_leg_poly",
            **self.quad_kwargs
        )
        
        # After f1_integr and f2_integr are available
        self._prepare_data()

    # ------------------------------------------------------
    # SPATIAL BASIS FUNCTIONS FOR u(x) AND v(x)
    # ------------------------------------------------------
    def h_u(self, x):
        """Return spatial profile h_u(x) = x^m1_u * (ℓ - x)^m2_u"""
        return x**self.m1_u * (self.cfg.ell - x)**self.m2_u

    def d1h_u(self, x):
        """Return ∂h_u/∂x computed via product rule"""
        return (
            x**(self.m1_u - 1) * (self.cfg.ell - x)**(self.m2_u - 1) *
            (self.m1_u * self.cfg.ell - (self.m1_u + self.m2_u) * x)
        )

    def h_v(self, x):
        """Return spatial profile h_v(x) = x^m1_v * (ℓ - x)^m2_v"""
        return x**self.m1_v * (self.cfg.ell - x)**self.m2_v

    def d1h_v(self, x):
        """Return ∂h_v/∂x computed via product rule"""
        return (
            x**(self.m1_v - 1) * (self.cfg.ell - x)**(self.m2_v - 1) *
            (self.m1_v * self.cfg.ell - (self.m1_v + self.m2_v) * x)
        )

    # ------------------------------------------------------
    # TEMPORAL BASIS FUNCTIONS FOR u(t) AND v(t)
    # ------------------------------------------------------
    def g_u(self, t):
        """Return temporal basis function for u(t); linear time evolution"""
        return t

    def d2g_u(self, t):
        """Return ∂²g_u/∂t² = 0"""
        return np.float64(0.0)

    def g_v(self, t):
        """Return temporal basis function for v(t); linear time evolution"""
        return t

    def d2g_v(self, t):
        """Return ∂²g_v/∂t² = 0"""
        return np.float64(0.0)

    # ------------------------------------------------------
    # EXACT SOLUTIONS FOR u(x, t) AND v(x, t)
    # ------------------------------------------------------
    def u(self, x, t):
        """Return exact solution u(x, t) = h_u(x) · g_u(t)"""
        return self.h_u(x) * self.g_u(t)

    def v(self, x, t):
        """Return exact solution v(x, t) = h_v(x) · g_v(t)"""
        return self.h_v(x) * self.g_v(t)

    # ------------------------------------------------------
    # DERIVATIVES OF u(x, t)
    # ------------------------------------------------------
    def diff1x_u(self, x, t):
        """Return ∂u/∂x = ∂h_u/∂x · g_u(t)"""
        return self.d1h_u(x) * self.g_u(t)

    def diff2t_u(self, x, t):
        """Return ∂²u/∂t² = h_u(x) · ∂²g_u/∂t²"""
        return self.h_u(x) * self.d2g_u(t)

    # ------------------------------------------------------
    # DERIVATIVES OF v(x, t)
    # ------------------------------------------------------
    def diff1x_v(self, x, t):
        """Return ∂v/∂x = ∂h_v/∂x · g_v(t)"""
        return self.d1h_v(x) * self.g_v(t)

    def diff2t_v(self, x, t):
        """Return ∂²v/∂t² = h_v(x) · ∂²g_v/∂t²"""
        return self.h_v(x) * self.d2g_v(t)

    # ------------------------------------------------------
    # NONLINEAR STIFFNESS TERM ∫(∂u/∂x)² dx
    # ------------------------------------------------------
    def integr_term(self, t):
        """
        Compute nonlinear term ∫ (∂u/∂x)² dx used in the u-equation.
        """
        integrand = lambda x: self.diff1x_u(x, t)
        result, *_ = aux.integrate_derivative_form(df=integrand, ell=self.cfg.ell)
        return result

    # ------------------------------------------------------
    # COMPONENTS OF SOURCE TERMS FOR f₁ AND f₂ IN WEAK FORM
    # ------------------------------------------------------
    def source_terms(self):
        """
        Return the functional components of the source terms f₁(x, t) and f₂(x, t)
        appearing in the weak formulation of the Timoshenko beam equations.
    
        Each returned component corresponds to a specific term in the weak form
        that must be projected against the test functions φₘ (and their gradients P̂ₘ).
    
        These callable components are necessary for computing time-dependent
        Galerkin inner products of the form:
            (term(x, t), φₘ)        for mass-type terms
            (term(x, t), P̂ₘ)        for stiffness/coupling terms
    
        -------------------------------------------------------------------
        f₁(x, t): Source term for the u-equation (displacement field)
        -------------------------------------------------------------------
        Strong form:
            f₁(x, t) = ∂²u/∂t² 
                     - (α + β ∫(∂u/∂x)² dx) ∂²u/∂x² 
                     + a₁ ∂v/∂x
    
        Weak form (after integration by parts):
            (f₁, φₘ) = (∂²u/∂t², φₘ)
                     + (α + β ∫(∂u/∂x)² dx) (∂u/∂x, P̂ₘ)
                     - a₁ (v, P̂ₘ)
    
        Required components:
            1. ∂²u/∂t² — acceleration term for φₘ projection
            2. ∂u/∂x   — nonlinear term for stiffness projection
            3. v(x, t) — coupling term, projected against P̂ₘ
    
        -------------------------------------------------------------------
        f₂(x, t): Source term for the v-equation (rotation field)
        -------------------------------------------------------------------
        Strong form:
            f₂(x, t) = ∂²v/∂t² 
                     - γ ∂²v/∂x² 
                     + δ v 
                     - a₂ ∂u/∂x
    
        Weak form (after integration by parts):
            (f₂, φₘ) = (∂²v/∂t², φₘ)
                     + γ (∂v/∂x, P̂ₘ)
                     + δ (v, φₘ)
                     + a₂ (u, P̂ₘ)
    
        Required components:
            1. ∂²v/∂t² — acceleration term for φₘ projection
            2. ∂v/∂x   — stiffness term for P̂ₘ projection
            3. v(x, t) — mass-like term for φₘ projection
            4. u(x, t) — coupling term, projected against P̂ₘ
    
        Returns
        -------
        dict of str -> list of callable
            {
                "f1": [∂²u/∂t², ∂u/∂x, v],
                "f2": [∂²v/∂t², ∂v/∂x, v, u]
            }
        """
        return {
            "f1": [
                lambda x, t: self.diff2t_u(x, t),  # Term 1: ∂²u/∂t² for (•, φₘ)
                lambda x, t: self.diff1x_u(x, t),  # Term 2: ∂u/∂x for (•, P̂ₘ)
                lambda x, t: self.v(x, t)          # Term 3: v(x, t) for (•, P̂ₘ)
            ],
            "f2": [
                lambda x, t: self.diff2t_v(x, t),  # Term 1: ∂²v/∂t² for (•, φₘ)
                lambda x, t: self.diff1x_v(x, t),  # Term 2: ∂v/∂x for (•, P̂ₘ)
                lambda x, t: self.v(x, t),         # Term 3: v(x, t) for (•, φₘ)
                lambda x, t: self.u(x, t)          # Term 4: u(x, t) for (•, P̂ₘ)
            ]
        }
    
        
    # ------------------------------------------------------
    # POST-INIT HOOK FOR DATACLASS INITIALIZATION
    # ------------------------------------------------------
    def __post_init__(self):
        """Ensure symbolic state is refreshed when __init__ is bypassed."""
        self._prepare_data()