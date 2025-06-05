import numpy as np  # For efficient numerical operations on arrays
import utils.config as cfg
import utils.auxiliary as aux

def solve_system(data, f1, f2):
    """
    Solves the coupled PDE system for modal coefficients using an explicit Galerkin method.
    
    Parameters:
        data (dict): Contains 'u_initial' and 'v_initial' as callable functions.
        f1, f2 (callables): Source term functions f1(x, t), f2(x, t).
    
    Returns:
        tild_u, tild_v: Arrays of modal coefficients at each time step.
        cond_u, cond_v: Arrays of condition numbers for u and v systems.
    """
    
    # Extract initial condition functions
    u_initial = data['u_initial']
    v_initial = data['v_initial']

    # Allocate storage for modal coefficients and condition numbers
    tild_u = np.zeros((cfg.n - 1, cfg.N))
    tild_v = np.zeros((cfg.n - 1, cfg.N))
    cond_u = np.zeros(cfg.n - 1)
    cond_v = np.zeros(cfg.n - 1)

    # Project source terms and initial data onto modal basis
    f1_integr = aux.compute_time_dependent_integrals(f1, cfg.N, cfg.ell, cfg.t)
    f2_integr = aux.compute_time_dependent_integrals(f2, cfg.N, cfg.ell, cfg.t)
    init_data = aux.compute_initial_integrals(u_initial, v_initial, cfg.N, cfg.ell)
    
    # Compute projections of initial data and spatial derivatives
    u0_integr, u1_integr = init_data['u_proj']
    v0_integr, v1_integr = init_data['v_proj']
    diff1u1 = init_data['diff1_u1']
    diff1v1 = init_data['diff1_v1']
    diff2u = init_data['diff2_u']
    diff2v = init_data['diff2_v']

    # Coefficient for v-equation system
    a0 = 4 / (2 + cfg.delta * cfg.tau**2)

    # Initialize nonlinear coefficient q
    integral, _ = aux.gauss_legendre_integrate_fprime_sq(u_initial[1], cfg.ell)
    q_prev = cfg.alpha + cfg.beta * integral

    # --- Time-stepping loop ---
    for k in range(cfg.n - 1):
        if k == 0:
            # --- Special Case: Time step 2 ---
            b1 = (4 / cfg.ell**2) * (
                cfg.tau**2 * f1_integr[k]
                + 2 * u1_integr
                - cfg.a1 * cfg.tau**2 * diff1v1
                - u0_integr
                + 0.5 * cfg.tau**2 * q_prev * diff2u[k]
            )
            b2 = (2 * a0 / cfg.ell**2) * (
                cfg.tau**2 * f2_integr[k]
                + 2 * v1_integr
                + cfg.a2 * cfg.tau**2 * diff1u1
                - (1 + 0.5 * cfg.tau**2 * cfg.delta) * v0_integr
                + 0.5 * cfg.tau**2 * cfg.gamma * diff2v[k]
            )

        elif k == 1:
            # --- Special Case: Time step 3 (uses tild_u[0], tild_v[0]) ---
            b1 = (4 / cfg.ell**2) * (
                cfg.tau**2 * f1_integr[k]
                + 0.5 * cfg.ell**2 * aux.galerkin_stencils(cfg.N, tild_u[k - 1])
                - 0.5 * cfg.a1 * cfg.tau**2 * cfg.ell * aux.galerkin_stencils(cfg.N, tild_v[k - 1], operator="first-order")
                - u1_integr
                + 0.5 * cfg.tau**2 * q_prev * diff2u[k]
            )
            b2 = (2 * a0 / cfg.ell**2) * (
                cfg.tau**2 * f2_integr[k]
                + 0.5 * cfg.ell**2 * aux.galerkin_stencils(cfg.N, tild_v[k - 1])
                + 0.5 * cfg.a2 * cfg.tau**2 * cfg.ell * aux.galerkin_stencils(cfg.N, tild_u[k - 1], operator="first-order")
                - (1 + 0.5 * cfg.tau**2 * cfg.delta) * v1_integr
                + 0.5 * cfg.tau**2 * cfg.gamma * diff2v[k]
            )
        else:
            # --- General Case: Time step k ≥ 2 ---
            b1 = (
                (4 * cfg.tau**2 / cfg.ell**2) * f1_integr[k]
                + 2 * aux.galerkin_stencils(cfg.N, tild_u[k - 1])
                - (2 * cfg.a1 * cfg.tau**2 / cfg.ell) * aux.galerkin_stencils(cfg.N, tild_v[k - 1], operator="first-order")
            )
            b2 = (
                (2 * a0 * cfg.tau**2 / cfg.ell**2) * f2_integr[k]
                + a0 * aux.galerkin_stencils(cfg.N, tild_v[k - 1])
                + (a0 * cfg.a2 * cfg.tau**2 / cfg.ell) * aux.galerkin_stencils(cfg.N, tild_u[k - 1], operator="first-order")
            )

        # --- Diagnostics: Compute condition numbers for system matrices ---
        cond_u[k] = aux.condition_number_associated_matrix(cfg.N, cfg.ell, 1, 0.5 * cfg.tau**2 * q_prev)
        cond_v[k] = aux.condition_number_associated_matrix(cfg.N, cfg.ell, 1 + 0.5 * cfg.tau**2 * cfg.delta, 0.5 * cfg.tau**2 * cfg.gamma)

        # Solve linear systems for modal coefficients
        tild_u[k] = aux.sys_soln(b1, cfg.N, 1, 0.5 * cfg.tau**2 * q_prev, cfg.ell)
        tild_v[k] = aux.sys_soln(b2, cfg.N, 1 + 0.5 * cfg.tau**2 * cfg.delta, 0.5 * cfg.tau**2 * cfg.gamma, cfg.ell)

        # Apply correction for time steps ≥ 2
        if k >= 2:
            tild_u[k] -= tild_u[k - 2]
            tild_v[k] -= tild_v[k - 2]

        # --- Update nonlinear term q for next iteration ---
        q_prev = cfg.alpha + cfg.beta * np.dot(tild_u[k], tild_u[k])

    return tild_u, tild_v, cond_u, cond_v
