import utils.config as cfg
from utils.auxGaussLegendreCoeff import coeff_A
from utils.auxLegendrePolynomials import normalized_shifted_legendre, phi_m

""" Benchmark exact solutions (used for testing and initial conditions) """
def u(x, t):
    #  Test function u(x, t) = t * phi_1(x)
    return t * phi_m(1, cfg.ell, x)

def v(x, t):
    #  Test function v(x, t) = t * phi_1(x)
    return t * phi_m(1, cfg.ell, x)

""" Initial conditions for u and v and their derivatives """

def diff2t_u(x, t): return 0
def diff2t_v(x, t): return 0

def diff2x_u(x, t):
    #  Second spatial derivative of u
    return (2 * normalized_shifted_legendre(0, cfg.ell, x)) / (cfg.ell * coeff_A(0) * coeff_A(1)) * t

def diff2x_v(x, t):
    #  Second spatial derivative of v
    return (2 * normalized_shifted_legendre(0, cfg.ell, x)) / (cfg.ell * coeff_A(0) * coeff_A(1)) * t

def diff1x_u(x, t):
    #  First spatial derivative of u
    return normalized_shifted_legendre(1, cfg.ell, x) * t

def diff1x_v(x, t):
    #  First spatial derivative of v
    return normalized_shifted_legendre(1, cfg.ell, x) * t

def integr_term(t): return t**2

""" Right-hand side for first equation """
def f1(x, t):
    #  RHS of first PDE
    return diff2t_u(x, t) - (cfg.alpha + cfg.beta * integr_term(t)) * diff2x_u(x, t) + cfg.a1 * diff1x_v(x, t)

def f2(x, t):
    #  RHS of second PDE
    return diff2t_v(x, t) - cfg.gamma * diff2x_v(x, t) + cfg.delta * v(x, t) - cfg.a2 * diff1x_u(x, t)