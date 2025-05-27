import numpy as np  # For efficient numerical operations on arrays
from utils.auxLegendrePolynomials import normalized_shifted_legendre
from utils.auxComputIntegrals import adaptive_gauss_legendre, integrate_with_phi_m
from utils.auxComputDerivIntegrals import adaptive_gauss_legendre_integrate_fprime_leg

""" Legendre-Galerkin Integral Projections and Initial Condition Processing for PDE Solvers """

def compute_initial_integrals(u, v, N, ell):
    """
    Project initial condition functions onto basis functions and compute spatial derivatives.
    
    Parameters:
        u   : list of functions [u0, u1], representing u(x, t=0) and ∂u/∂t at t=0
        v   : list of functions [v0, v1], same as above for v
        N   : int, number of basis functions
        ell : float, scaling parameter for the domain
    
    Returns:
        A dictionary containing:
            - u_proj      : [∫ u0*φ_m dx, ∫ u1*φ_m dx] for m = 1..N
            - v_proj      : [∫ v0*φ_m dx, ∫ v1*φ_m dx] for m = 1..N
            - diff1_u1    : ∫ du1/dx * φ_m dx = ∫ -u1 * φ_m' dx for m = 1..N
            - diff1_v1    : same for v1
            - diff2_u     : [[∫ d²u0/dx² * φ_m dx], [∫ d²u1/dx² * φ_m dx]]
            - diff2_v     : same for v
    """
    # Initialize projection arrays
    u_proj = [np.zeros(N), np.zeros(N)]     # Projections of u0 and u1
    v_proj = [np.zeros(N), np.zeros(N)]     # Projections of v0 and v1
    diff1_u1 = np.zeros(N)                  # First derivative of u1 projected
    diff1_v1 = np.zeros(N)                  # First derivative of v1 projected
    diff2_u = np.zeros((2, N))              # Second derivative terms for u0 and u1
    diff2_v = np.zeros((2, N))              # Second derivative terms for v0 and v1

    for m in range(N):
        m_idx = m + 1  # φ_m index starts at 1 in auxiliary functions

        # --- Projections of initial values onto φ_m(x) ---
        u_proj[0][m], _ = integrate_with_phi_m(u[0], m_idx, ell)  # u0
        u_proj[1][m], _ = integrate_with_phi_m(u[1], m_idx, ell)  # u1
        v_proj[0][m], _ = integrate_with_phi_m(v[0], m_idx, ell)  # v0
        v_proj[1][m], _ = integrate_with_phi_m(v[1], m_idx, ell)  # v1

        # --- First derivative: ∂/∂x of u1 and v1, projected via weak form ---
        diff1_u1[m], _ = adaptive_gauss_legendre(
            lambda x: -u[1](x) * normalized_shifted_legendre(m_idx, ell, x), ell
        )
        diff1_v1[m], _ = adaptive_gauss_legendre(
            lambda x: -v[1](x) * normalized_shifted_legendre(m_idx, ell, x), ell
        )

        # --- Second derivative terms from weak form using ∫ f' * φ'_m dx ---
        for i in range(2):
            diff2_u[i][m], _ = adaptive_gauss_legendre_integrate_fprime_leg(
                lambda x, i=i: -u[i](x), m_idx, ell
            )
            diff2_v[i][m], _ = adaptive_gauss_legendre_integrate_fprime_leg(
                lambda x, i=i: -v[i](x), m_idx, ell
            )

    # Return all components in a structured dict
    return {
        'u_proj': u_proj,
        'v_proj': v_proj,
        'diff1_u1': diff1_u1,
        'diff1_v1': diff1_v1,
        'diff2_u': diff2_u,
        'diff2_v': diff2_v,
    }