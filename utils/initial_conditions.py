import utils.config as cfg
from utils.auxLegendrePolynomials import phi_m
from utils.equations import u, v, diff1x_u, diff1x_v, diff2x_u, diff2x_v, f1, f2, integr_term

""" Initial conditions for u and v and their derivatives """
def diff_t_u(x, t): return phi_m(1, cfg.ell, x)
def diff_t_v(x, t): return phi_m(1, cfg.ell, x)

def setup_initial_conditions():
    varphi0 = lambda x: u(x, 0)
    varphi1 = lambda x: diff_t_u(x, 0)
    varphi2 = lambda x: f1(x, 0) - cfg.a1 * diff1x_v(x, 0) + (cfg.alpha + cfg.beta * integr_term(0)) * diff2x_u(x, 0)

    psi0 = lambda x: v(x, 0)
    psi1 = lambda x: diff_t_v(x, 0)
    psi2 = lambda x: f2(x, 0) + cfg.a2 * diff1x_u(x, 0) + cfg.gamma * diff2x_v(x, 0) - cfg.delta * psi0(x)
    
    """ Initial functions u₀ and u₁ using Taylor expansion """
    u0 = lambda x: varphi0(x)
    u1 = lambda x: varphi0(x) + cfg.tau * varphi1(x) + 0.5 * cfg.tau**2 * varphi2(x)

    v0 = lambda x: psi0(x)
    v1 = lambda x: psi0(x) + cfg.tau * psi1(x) + 0.5 * cfg.tau**2 * psi2(x)

    u_initial = [u0, u1]
    v_initial = [v0, v1]

    return {
        'u_initial': u_initial,
        'v_initial': v_initial
    }