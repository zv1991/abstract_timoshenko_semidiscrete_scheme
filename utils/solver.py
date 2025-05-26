import numpy as np
from config import ell, N, n, t, tau, alpha, beta, gamma, delta, a1, a2
import auxiliary as aux

def solve_system(data, f1, f2):
    # Initial condition functions for projection
    u_initial = data['u_initial']
    v_initial = data['v_initial']
    
    # Allocate memory for modal coefficients
    tild_u = np.zeros((n - 1, N))
    tild_v = np.zeros((n - 1, N))
    cond_u = np.zeros(n - 1)
    cond_v = np.zeros(n - 1)
    
    # Compute projections of source terms f1 and f2
    f1_integr = aux.compute_time_dependent_integrals(f1, n, N, ell, t)
    f2_integr = aux.compute_time_dependent_integrals(f2, n, N, ell, t)
    init_data = aux.compute_initial_integrals(u_initial, v_initial, N, ell)
    
    # Compute projections of initial data and their spatial derivatives
    u0_integr, u1_integr = init_data['u_proj']
    v0_integr, v1_integr = init_data['v_proj']
    diff1u1, diff1v1 = init_data['diff1_u1'], init_data['diff1_v1']
    diff2u, diff2v = init_data['diff2_u'], init_data['diff2_v']
    
    # Constant used in the v-equation formulation
    a0 = 4 / (2 + delta * tau**2)
    
    # Initial nonlinear term from energy-like integral
    integral, _ = aux.adaptive_gauss_legendre_integrate_fprime_sq(u_initial[1], ell)
    q_prev = alpha + beta * integral
    
    """ Main time-stepping loop (explicit Galerkin integration) """
    for k in range(n - 1):
        if k == 0:
            # Time step 2 (special treatment)
            
            # Compute right-hand side (RHS) of the u-equation
            b1 = (4 / ell**2) * (
                tau**2 * f1_integr[k]
                + 2 * u1_integr
                - a1 * tau**2 * diff1v1
                - u0_integr
                + 0.5 * tau**2 * q_prev * diff2u[k]
            )
            
            # Compute RHS of the v-equation
            b2 = (2 * a0 / ell**2) * (
                tau**2 * f2_integr[k]
                + 2 * v1_integr
                + a2 * tau**2 * diff1u1
                - (1 + 0.5 * tau**2 * delta) * v0_integr
                + 0.5 * tau**2 * gamma * diff2v[k]
            )
        elif k == 1:
            # Time step 3 (uses tild_u[k-1])
            
            # Compute RHS for u-equation using Galerkin approximation
            b1 = (4 / ell**2) * (
                tau**2 * f1_integr[k]
                + 0.5 * ell**2 * aux.galerkin_stencils(N, tild_u[k - 1])
                - 0.5 * a1 * tau**2 * ell * aux.galerkin_stencils(N, tild_v[k - 1], operator="first-order")
                - u1_integr
                + 0.5 * tau**2 * q_prev * diff2u[k]
            )
            
            # Compute RHS for v-equation similarly
            b2 = (2 * a0 / ell**2) * (
                tau**2 * f2_integr[k]
                + 0.5 * ell**2 * aux.galerkin_stencils(N, tild_v[k - 1])
                + 0.5 * a2 * tau**2 * ell * aux.galerkin_stencils(N, tild_u[k - 1], operator="first-order")
                - (1 + 0.5 * tau**2 * delta) * v1_integr
                + 0.5 * tau**2 * gamma * diff2v[k]
            )
        else:
            # General time step (k ≥ 4)
            
            # Compute RHS for u-equation using Galerkin approximation
            b1 = (
                (4 * tau**2 / ell**2) * f1_integr[k]
                + 2 * aux.galerkin_stencils(N, tild_u[k - 1])
                - (2 * a1 * tau**2 / ell) * aux.galerkin_stencils(N, tild_v[k - 1], operator="first-order")
            )
            
            # Compute RHS for v-equation similarly
            b2 = (
                (2 * a0 * tau**2 / ell**2) * f2_integr[k]
                + a0 * aux.galerkin_stencils(N, tild_v[k - 1])
                + (a0 * a2 * tau**2 / ell) * aux.galerkin_stencils(N, tild_u[k - 1], operator="first-order")
            )
        
        # Condition number diagnostics
        cond_u[k] = aux.condition_number_associated_matrix(N, ell, 1, 0.5 * tau**2 * q_prev)
        cond_v[k] = aux.condition_number_associated_matrix(N, ell, 1 + 0.5 * tau**2 * delta, 0.5 * tau**2 * gamma)
        
        # Solve linear systems
        tild_u[k] = aux.sys_soln(b1, N, 1, 0.5 * tau**2 * q_prev, ell)
        tild_v[k] = aux.sys_soln(b2, N, 1 + 0.5 * tau**2 * delta, 0.5 * tau**2 * gamma, ell)
        
        # Apply correction for time steps ≥ 2
        if k >= 2:
            tild_u[k] -= tild_u[k - 2]
            tild_v[k] -= tild_v[k - 2]
        
        # Update nonlinear term q
        q_prev = alpha + beta * np.dot(tild_u[k], tild_u[k])

    return tild_u, tild_v, cond_u, cond_v