import utils.config as cfg        # Global configuration (domain length, time step, coefficients)
import utils.auxiliary as aux     # Math utilities (Legendre basis, etc.)
import utils.equations as eqs     # PDE solutions and derivatives

# -------------------------- First-Order Time Derivatives -------------------------- #
def diff_t_u(x, t):
    """
    First time derivative of u(x, t) = d/dt [t · φ₁(x)] = φ₁(x)
    """
    return aux.phi_m(1, cfg.ell, x)

def diff_t_v(x, t):
    """
    First time derivative of v(x, t) = d/dt [t · φ₁(x)] = φ₁(x)
    """
    return aux.phi_m(1, cfg.ell, x)

# -------------------------- Initial Condition Setup -------------------------- #
def setup_initial_conditions():
    """
    Computes initial condition functions u₀, u₁, v₀, v₁ using Taylor expansion at t = 0:
        u₁(x) ≈ u₀(x) + τ·u_t(x,0) + (τ²/2)·u_tt(x,0)
        v₁(x) ≈ v₀(x) + τ·v_t(x,0) + (τ²/2)·v_tt(x,0)
    
    Returns:
        dict: {'u_initial': [u₀, u₁], 'v_initial': [v₀, v₁]} as callable functions
    """

    # ------------------- u(x, 0) and its time derivatives ------------------- #
    varphi0 = lambda x: eqs.u(x, 0)     # u(x, 0)
    varphi1 = lambda x: diff_t_u(x, 0)  # u_t(x, 0)

    # u_tt(x, 0) from rearranged PDE f1 = u_tt - (...) => u_tt = f1 + ...
    varphi2 = lambda x: (
        eqs.f1(x, 0)
        - cfg.a1 * eqs.diff1x_v(x, 0)
        + (cfg.alpha + cfg.beta * eqs.integr_term(0)) * eqs.diff2x_u(x, 0)
    )

    # ------------------- v(x, 0) and its time derivatives ------------------- #
    psi0 = lambda x: eqs.v(x, 0)       # v(x, 0)
    psi1 = lambda x: diff_t_v(x, 0)    # v_t(x, 0)

    # v_tt(x, 0) from rearranged PDE f2 = v_tt - (...) => v_tt = f2 + ...
    psi2 = lambda x: (
        eqs.f2(x, 0)
        + cfg.a2 * eqs.diff1x_u(x, 0)
        + cfg.gamma * eqs.diff2x_v(x, 0)
        - cfg.delta * psi0(x)
    )

    # ------------------- Construct Initial Data via Taylor Series ------------------- #
    u0 = lambda x: varphi0(x)  # u₀(x) = u(x, 0)
    u1 = lambda x: varphi0(x) + cfg.tau * varphi1(x) + 0.5 * cfg.tau**2 * varphi2(x)  # u₁(x)

    v0 = lambda x: psi0(x)  # v₀(x) = v(x, 0)
    v1 = lambda x: psi0(x) + cfg.tau * psi1(x) + 0.5 * cfg.tau**2 * psi2(x)  # v₁(x)

    # Package results
    return {
        'u_initial': [u0, u1],  # [u₀(x), u₁(x)]
        'v_initial': [v0, v1]   # [v₀(x), v₁(x)]
    }