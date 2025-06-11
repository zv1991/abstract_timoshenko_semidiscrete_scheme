# ======================================================
# MODULE IMPORTS
# ======================================================

import utils.config as cfg                           # Global configuration constants
from utils.symbolic_derivatives import SymbolicDerivatives as SD  # Symbolic derivative interface

# ======================================================
# INITIAL CONDITION SETUP
# ======================================================

def setup_initial_conditions():
    """
    Computes initial conditions for the PDE system using second-order Taylor expansion at t = 0.
    
    Taylor expansion approximation:
        u₁(x) ≈ u₀(x) + τ·u_t(x, 0) + (τ²/2)·u_tt(x, 0)
        v₁(x) ≈ v₀(x) + τ·v_t(x, 0) + (τ²/2)·v_tt(x, 0)

    Returns:
        dict: A dictionary with keys:
            'u_initial' : [u₀(x), u₁(x)]
            'v_initial' : [v₀(x), v₁(x)]
    """

    # --------------------------------------------------
    # u(x, t) and its time derivatives evaluated at t = 0
    # --------------------------------------------------

    # u₀(x): Initial displacement u(x, 0)
    varphi0 = lambda x: SD.u(x, 0)

    # u_t(x, 0): First time derivative at t = 0
    varphi1 = lambda x: SD.diff1t_u(x, 0)

    # u_tt(x, 0): Second time derivative via rearranged PDE f1 expression
    varphi2 = lambda x: (
        SD.f1(x, 0)
        - cfg.a1 * SD.diff1x_v(x, 0)
        + (cfg.alpha + cfg.beta * SD.integr_term(0)) * SD.diff2x_u(x, 0)
    )

    # --------------------------------------------------
    # v(x, t) and its time derivatives evaluated at t = 0
    # --------------------------------------------------

    # v₀(x): Initial displacement v(x, 0)
    psi0 = lambda x: SD.v(x, 0)

    # v_t(x, 0): First time derivative at t = 0
    psi1 = lambda x: SD.diff1t_v(x, 0)

    # v_tt(x, 0): Second time derivative via rearranged PDE f2 expression
    psi2 = lambda x: (
        SD.f2(x, 0)
        + cfg.a2 * SD.diff1x_u(x, 0)
        + cfg.gamma * SD.diff2x_v(x, 0)
        - cfg.delta * psi0(x)
    )

    # --------------------------------------------------
    # Construct Taylor-approximated initial data
    # --------------------------------------------------

    # u₀(x): Initial condition at t = 0
    u0 = lambda x: varphi0(x)

    # u₁(x): Approximation at t = τ
    u1 = lambda x: varphi0(x) + cfg.tau * varphi1(x) + 0.5 * cfg.tau**2 * varphi2(x)

    # v₀(x): Initial condition at t = 0
    v0 = lambda x: psi0(x)

    # v₁(x): Approximation at t = τ
    v1 = lambda x: psi0(x) + cfg.tau * psi1(x) + 0.5 * cfg.tau**2 * psi2(x)

    # --------------------------------------------------
    # Return packaged dictionary of initial conditions
    # --------------------------------------------------

    return {
        'u_initial': [u0, u1],
        'v_initial': [v0, v1]
    }