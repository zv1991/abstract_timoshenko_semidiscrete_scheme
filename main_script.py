import numpy as np
import auxiliary as aux  # Custom module containing utility functions
from auxiliary import galerkin_approx  # Direct import for convenience

# Temporal and spatial domain definitions
T = 1          # Time interval length
ell = 2        # Spatial interval length

# -------------------------------------------------------------------
# Benchmark exact solutions (used for testing and initial conditions)
# -------------------------------------------------------------------
def u(x, t):
    """Test function u(x, t) = t * phi_1(x)"""
    return t * aux.phi_m(1, ell, x)

def v(x, t):
    """Test function v(x, t) = t * phi_1(x)"""
    return t * aux.phi_m(1, ell, x)

# Equation parameters
alpha, beta, gamma, delta = 1, 1, 1, 1
a1, a2 = 1, 1

# -------------------------------------------------------------------
# Right-hand side for first equation
# -------------------------------------------------------------------
def diff2t_u(x, t): return 0

def diff2x_u(x, t):
    """Second spatial derivative of u"""
    return (2 * aux.normalized_shifted_legendre(0, ell, x)) / (ell * aux.coeff_A(0) * aux.coeff_A(1)) * t

def diff1x_v(x, t):
    """First spatial derivative of v"""
    return aux.normalized_shifted_legendre(1, ell, x) * t

def integr_term(t): return t ** 2

def f1(x, t):
    """RHS of first PDE"""
    return diff2t_u(x, t) - (alpha + beta * integr_term(t)) * diff2x_u(x, t) + a1 * diff1x_v(x, t)

# -------------------------------------------------------------------
# Right-hand side for second equation
# -------------------------------------------------------------------
def diff2t_v(x, t): return 0

def diff2x_v(x, t):
    """Second spatial derivative of v"""
    return (2 * aux.normalized_shifted_legendre(0, ell, x)) / (ell * aux.coeff_A(0) * aux.coeff_A(1)) * t

def diff1x_u(x, t):
    """First spatial derivative of u"""
    return aux.normalized_shifted_legendre(1, ell, x) * t

def f2(x, t):
    """RHS of second PDE"""
    return diff2t_v(x, t) - gamma * diff2x_v(x, t) + delta * v(x, t) - a2 * diff1x_u(x, t)

# -------------------------------------------------------------------
# Initial conditions for u and v and their derivatives
# -------------------------------------------------------------------
def diff_t_u(x, t): return aux.phi_m(1, ell, x)
def diff_t_v(x, t): return aux.phi_m(1, ell, x)

def varphi0(x): return u(x, 0)
def varphi1(x): return diff_t_u(x, 0)
def varphi2(x): return f1(x, 0) - a1 * diff1x_v(x, 0) + (alpha + beta * integr_term(0)) * diff2x_u(x, 0)

def psi0(x): return v(x, 0)
def psi1(x): return diff_t_v(x, 0)
def psi2(x): return f2(x, 0) + a2 * diff1x_u(x, 0) + gamma * diff2x_v(x, 0) - delta * psi0(x)

# -------------------------------------------------------------------
# Temporal discretization setup
# -------------------------------------------------------------------
n = 10
t = np.linspace(0, T, n + 1)
tau = T / n

# Initial functions u_0 and u_1 using Taylor expansion
def u0(x): return varphi0(x)
def u1(x): return varphi0(x) + tau * varphi1(x) + 0.5 * tau**2 * varphi2(x)

def v0(x): return psi0(x)
def v1(x): return psi0(x) + tau * psi1(x) + 0.5 * tau**2 * psi2(x)

# Number of basis functions (spectral order)
N = 5

# Allocate memory for modal coefficients
tild_u = np.zeros((n - 1, N))
tild_v = np.zeros((n - 1, N))
cond_numb_sys_u = np.zeros(n - 1)
cond_numb_sys_v = np.zeros(n - 1)

# Initial condition functions for projection
u_initial = [u0, u1]
v_initial = [v0, v1]

# Compute projections of source terms f1 and f2
f1_integr = aux.compute_time_dependent_integrals(f1, n, N, ell, t)
f2_integr = aux.compute_time_dependent_integrals(f2, n, N, ell, t)

# Compute projections of initial data and their spatial derivatives
init_data = aux.compute_initial_integrals(u_initial, v_initial, N, ell)
u0_integr, u1_integr = init_data['u_proj']
v0_integr, v1_integr = init_data['v_proj']
diff1u1 = init_data['diff1_u1']
diff1v1 = init_data['diff1_v1']
diff2u = init_data['diff2_u']
diff2v = init_data['diff2_v']

# Constant used in the v-equation formulation
a0 = 4 / (2 + delta * tau**2)

# Initial nonlinear term from energy-like integral
integral, _ = aux.adaptive_gauss_legendre_integrate_fprime_sq(u1, ell)
q_prev = alpha + beta * integral

# -------------------------------------------------------------------
# Main time-stepping loop (explicit Galerkin integration)
# -------------------------------------------------------------------
for k in range(n - 1):
    
    if k == 0:
        # Time step 1 (special treatment)
        b1 = (4 / ell**2) * (
            tau**2 * f1_integr[k]
            + 2 * u1_integr
            - a1 * tau**2 * diff1v1
            - u0_integr
            + 0.5 * tau**2 * q_prev * diff2u[k]
        )
        b2 = (2 * a0 / ell**2) * (
            tau**2 * f2_integr[k]
            + 2 * v1_integr
            + a2 * tau**2 * diff1u1
            - (1 + 0.5 * tau**2 * delta) * v0_integr
            + 0.5 * tau**2 * gamma * diff2v[k]
        )

    elif k == 1:
        # Time step 2 (uses tild_u[k-1])
        b1 = (4 / ell**2) * (
            tau**2 * f1_integr[k]
            + 0.5 * ell**2 * aux.galerkin_stencils(N, tild_u[k - 1])
            - 0.5 * a1 * tau**2 * ell * aux.galerkin_stencils(N, tild_v[k - 1], operator="first-order")
            - u1_integr
            + 0.5 * tau**2 * q_prev * diff2u[k]
        )
        b2 = (2 * a0 / ell**2) * (
            tau**2 * f2_integr[k]
            + 0.5 * ell**2 * aux.galerkin_stencils(N, tild_v[k - 1])
            + 0.5 * a2 * tau**2 * ell * aux.galerkin_stencils(N, tild_u[k - 1], operator="first-order")
            - (1 + 0.5 * tau**2 * delta) * v1_integr
            + 0.5 * tau**2 * gamma * diff2v[k]
        )
        
    else:
        # General time step (k ≥ 2)
        b1 = (
            (4 * tau**2 / ell**2) * f1_integr[k]
            + 2 * aux.galerkin_stencils(N, tild_u[k - 1])
            - (2 * a1 * tau**2 / ell) * aux.galerkin_stencils(N, tild_v[k - 1], operator="first-order")
        )
        b2 = (
            (2 * a0 * tau**2 / ell**2) * f2_integr[k]
            + a0 * aux.galerkin_stencils(N, tild_v[k - 1])
            + (a0 * a2 * tau**2 / ell) * aux.galerkin_stencils(N, tild_u[k - 1], operator="first-order")
        )

    # Condition number diagnostics
    cond_numb_sys_u[k] = aux.condition_number_associated_matrix(N, ell, 1, 0.5 * tau**2 * q_prev)
    cond_numb_sys_v[k] = aux.condition_number_associated_matrix(N, ell, 1 + 0.5 * tau**2 * delta, 0.5 * tau**2 * gamma)

    # Solve linear systems
    sol_u = aux.sys_soln(b1, N, 1, 0.5 * tau**2 * q_prev, ell)
    sol_v = aux.sys_soln(b2, N, 1 + 0.5 * tau**2 * delta, 0.5 * tau**2 * gamma, ell)

    # Apply correction for time steps ≥ 2
    if k >= 2:
        sol_u -= tild_u[k - 2]
        sol_v -= tild_v[k - 2]

    tild_u[k] = sol_u
    tild_v[k] = sol_v

    # Update nonlinear term q
    q_prev = alpha + beta * np.dot(tild_u[k], tild_u[k])

# -------------------------------------------------------------------
# Evaluation functions (recover physical values from modal coefficients)
# -------------------------------------------------------------------
def approx_u(N: int, k: int, ell: float, x: np.ndarray) -> np.ndarray:
    """Evaluate Galerkin approximation of u at timestep k (k ≥ 2)."""
    if k < 2:
        raise ValueError("Temporal index k must be ≥ 2")
    return galerkin_approx(N, ell, tild_u[k - 2], x)

def approx_v(N: int, k: int, ell: float, x: np.ndarray) -> np.ndarray:
    """Evaluate Galerkin approximation of v at timestep k (k ≥ 2)."""
    if k < 2:
        raise ValueError("Temporal index k must be ≥ 2")
    return galerkin_approx(N, ell, tild_v[k - 2], x)