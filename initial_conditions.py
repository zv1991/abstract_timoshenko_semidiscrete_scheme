from config import ell, alpha, beta, gamma, delta, a1, a2, tau
from symbolicApproach import u, v, diff1x_u, diff1x_v, diff2x_u, diff2x_v, integr_term, f1, f2
import auxiliary as aux

""" Initial conditions for u and v and their derivatives """
def diff_t_u(x, t): return aux.phi_m(1, ell, x)
def diff_t_v(x, t): return aux.phi_m(1, ell, x)

def setup_initial_conditions():
    varphi0 = lambda x: u(x, 0)
    varphi1 = lambda x: diff_t_u(x, 0)
    varphi2 = lambda x: f1(x, 0) - a1 * diff1x_v(x, 0) + (alpha + beta * integr_term(0)) * diff2x_u(x, 0)

    psi0 = lambda x: v(x, 0)
    psi1 = lambda x: diff_t_v(x, 0)
    psi2 = lambda x: f2(x, 0) + a2 * diff1x_u(x, 0) + gamma * diff2x_v(x, 0) - delta * psi0(x)
    
    """ Initial functions u₀ and u₁ using Taylor expansion """
    u0 = lambda x: varphi0(x)
    u1 = lambda x: varphi0(x) + tau * varphi1(x) + 0.5 * tau**2 * varphi2(x)

    v0 = lambda x: psi0(x)
    v1 = lambda x: psi0(x) + tau * psi1(x) + 0.5 * tau**2 * psi2(x)

    u_initial = [u0, u1]
    v_initial = [v0, v1]

    return {
        'u_initial': u_initial,
        'v_initial': v_initial
    }