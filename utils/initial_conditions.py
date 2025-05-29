import utils.config as cfg
import utils.auxiliary as aux
import utils.equations as eqs

""" Initial conditions for u and v and their derivatives """
def diff_t_u(x, t): return aux.phi_m(1, cfg.ell, x)
def diff_t_v(x, t): return aux.phi_m(1, cfg.ell, x)

def setup_initial_conditions():
    varphi0 = lambda x: eqs.u(x, 0)
    varphi1 = lambda x: diff_t_u(x, 0)
    varphi2 = lambda x: eqs.f1(x, 0) - cfg.a1 * eqs.diff1x_v(x, 0) + (cfg.alpha + cfg.beta * eqs.integr_term(0)) * eqs.diff2x_u(x, 0)

    psi0 = lambda x: eqs.v(x, 0)
    psi1 = lambda x: diff_t_v(x, 0)
    psi2 = lambda x: eqs.f2(x, 0) + cfg.a2 * eqs.diff1x_u(x, 0) + cfg.gamma * eqs.diff2x_v(x, 0) - cfg.delta * psi0(x)
    
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