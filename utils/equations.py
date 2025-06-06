import utils.config as cfg     # Configuration constants (e.g., domain size, coefficients)
import utils.auxiliary as aux  # Auxiliary math tools (e.g., Legendre basis functions)

# ---------------------------- Benchmark Solutions ---------------------------- #
def u(x, t):
    """
    Benchmark solution for u(x, t) = t * φ₁(x)
    """
    return t * aux.phi_m(1, cfg.ell, x)

def v(x, t):
    """
    Benchmark solution for v(x, t) = t * φ₁(x)
    """
    return t * aux.phi_m(1, cfg.ell, x)

# ------------------------ Derivatives of u and v ----------------------------- #
def diff2t_u(x, t):
    """
    Second time derivative of u(x, t)
    Since u = t * φ₁(x), d²u/dt² = 0
    """
    return 0

def diff2t_v(x, t):
    """
    Second time derivative of v(x, t)
    """
    return 0

def diff2x_u(x, t):
    """
    Second spatial derivative of u(x, t) = t * φ₁(x)
    Result uses recurrence relations and derivative properties of Legendre polynomials.
    """
    A0 = aux.coeff_A(0)
    A1 = aux.coeff_A(1)
    return (2 * aux.normalized_shifted_legendre(0, cfg.ell, x)) / (cfg.ell * A0 * A1) * t

def diff2x_v(x, t):
    """
    Second spatial derivative of v(x, t)
    """
    A0 = aux.coeff_A(0)
    A1 = aux.coeff_A(1)
    return (2 * aux.normalized_shifted_legendre(0, cfg.ell, x)) / (cfg.ell * A0 * A1) * t

def diff1x_u(x, t):
    """
    First spatial derivative of u(x, t) = t * φ₁(x)
    dφ₁/dx = normalized_shifted_legendre(1)
    """
    return aux.normalized_shifted_legendre(1, cfg.ell, x) * t

def diff1x_v(x, t):
    """
    First spatial derivative of v(x, t)
    """
    return aux.normalized_shifted_legendre(1, cfg.ell, x) * t

# ---------------------- Nonlinear Time-Dependent Term ------------------------ #
def integr_term(t):
    """
    Time-dependent nonlinear scalar function used in PDEs: I(t) = t²
    """
    return t**2

# ----------------------------- RHS Definitions ------------------------------- #
def f1(x, t):
    """
    Right-hand side of the first PDE in the coupled system:
        f₁(x, t) = u_tt - (α + β·I(t))·u_xx + a₁·v_x
    """
    return (
        diff2t_u(x, t)
        - (cfg.alpha + cfg.beta * integr_term(t)) * diff2x_u(x, t)
        + cfg.a1 * diff1x_v(x, t)
    )

def f2(x, t):
    """
    Right-hand side of the second PDE in the coupled system:
        f₂(x, t) = v_tt - γ·v_xx + δ·v - a₂·u_x
    """
    return (
        diff2t_v(x, t)
        - cfg.gamma * diff2x_v(x, t)
        + cfg.delta * v(x, t)
        - cfg.a2 * diff1x_u(x, t)
    )