from config import ell, alpha, beta, gamma, delta, a1, a2
import auxiliary as aux

""" Benchmark exact solutions (used for testing and initial conditions) """
def u(x, t):
    #  Test function u(x, t) = t * phi_1(x)
    return t * aux.phi_m(1, ell, x)

def v(x, t):
    #  Test function v(x, t) = t * phi_1(x)
    return t * aux.phi_m(1, ell, x)

""" Initial conditions for u and v and their derivatives """

def diff2t_u(x, t): return 0
def diff2t_v(x, t): return 0

def diff2x_u(x, t):
    #  Second spatial derivative of u
    return (2 * aux.normalized_shifted_legendre(0, ell, x)) / (ell * aux.coeff_A(0) * aux.coeff_A(1)) * t

def diff2x_v(x, t):
    #  Second spatial derivative of v
    return (2 * aux.normalized_shifted_legendre(0, ell, x)) / (ell * aux.coeff_A(0) * aux.coeff_A(1)) * t

def diff1x_u(x, t):
    #  First spatial derivative of u
    return aux.normalized_shifted_legendre(1, ell, x) * t

def diff1x_v(x, t):
    #  First spatial derivative of v
    return aux.normalized_shifted_legendre(1, ell, x) * t

def integr_term(t): return t**2

""" Right-hand side for first equation """
def f1(x, t):
    #  RHS of first PDE
    return diff2t_u(x, t) - (alpha + beta * integr_term(t)) * diff2x_u(x, t) + a1 * diff1x_v(x, t)

def f2(x, t):
    #  RHS of second PDE
    return diff2t_v(x, t) - gamma * diff2x_v(x, t) + delta * v(x, t) - a2 * diff1x_u(x, t)