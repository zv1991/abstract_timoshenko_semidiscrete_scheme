# --- Imports ---
import numpy as np  # NumPy: essential for array-based numerical computing, linear algebra, etc.

# Project-specific configuration and utility modules
import utils.config as cfg          # cfg: Contains simulation settings like domain length, time step size, constants, and discretization parameters
import utils.auxiliary as aux       # aux: Provides helper functions for integration, projections, Galerkin stencils, system solving, etc.


def solve_system(
    u_initial, v_initial, f1, f2,
    *, h=1e-3, derivmeth='nd', tol=1e-6, method='hglq',
    max_n=50, max_depth=20, n_points=10
):
    """
    Solves a coupled nonlinear PDE system using a Galerkin-based time-stepping approach.

    Parameters:
        u_initial (tuple): Initial functions (u0, u1) where each u_i is a function of x.
        v_initial (tuple): Initial functions (v0, v1) where each v_i is a function of x.
        f1, f2 (callable): Time-dependent source functions f1(x, t), f2(x, t).
        h (float): Step size for numerical differentiation (used in auxiliary computations).
        derivmeth (str): Method for derivative approximation (default is 'nd' for numerical differentiation).
        tol (float): Tolerance for adaptive quadrature.
        method (str): Quadrature integration method ('hglq', 'glq', 'scipy').
        max_n (int): Maximum quadrature order.
        max_depth (int): Maximum recursion depth for hierarchical integration.
        n_points (int): Number of evaluation points per integration subinterval.

    Returns:
        tuple: (tild_u, tild_v, cond_u, cond_v)
            - tild_u, tild_v (ndarray): Modal coefficients of u and v over time.
            - cond_u, cond_v (ndarray): Condition numbers of the system matrices at each time step.
    """

    # --- Define keyword arguments for quadrature routines ---
    quad_kwargs = dict(
        tol=tol,
        method=method,
        max_n=max_n,
        max_depth=max_depth,
        n_points=n_points
    )

    # --- Initialize storage arrays for solution and diagnostics ---
    tild_u = np.zeros((cfg.n - 1, cfg.N))  # Modal coefficients of u
    tild_v = np.zeros((cfg.n - 1, cfg.N))  # Modal coefficients of v
    cond_u = np.zeros(cfg.n - 1)           # Condition numbers for u matrix at each step
    cond_v = np.zeros(cfg.n - 1)           # Condition numbers for v matrix at each step

    # --- Precompute time-dependent source function projections ---
    f1_integr = aux.compute_time_dependent_integrals(f1, cfg.N, cfg.ell, cfg.t, **quad_kwargs)
    f2_integr = aux.compute_time_dependent_integrals(f2, cfg.N, cfg.ell, cfg.t, **quad_kwargs)

    # --- Compute initial condition projections and spatial derivatives ---
    init_data = aux.compute_initial_integrals(
        u_initial, v_initial, cfg.N, cfg.ell, h=h, derivmeth=derivmeth, **quad_kwargs
    )
    u0_integr, u1_integr = init_data['u_proj']
    v0_integr, v1_integr = init_data['v_proj']
    diff1u1 = init_data['diff1_u1']
    diff1v1 = init_data['diff1_v1']
    diff2u = init_data['diff2_u']
    diff2v = init_data['diff2_v']

    # --- Coefficient for v-equation matrix ---
    a0 = 4 / (2 + cfg.delta * cfg.tau**2)

    # --- Initialize q (nonlinear term) from ∫(u1')² ---
    integral, _ = aux.integrate_fprime_sq(u_initial[1], cfg.ell)
    q_prev = cfg.alpha + cfg.beta * integral

    # --- Time-stepping loop ---
    for k in range(cfg.n - 1):

        # --- Construct RHS for Galerkin system at each step ---
        if k == 0:
            # Step 2: Explicit handling of first time step
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
            # Step 3: Use Galerkin stencils from step 2
            b1 = (4 / cfg.ell**2) * (
                cfg.tau**2 * f1_integr[k]
                + 0.5 * cfg.ell**2 * aux.galerkin_stencils(cfg.N, tild_u[k - 1])
                - 0.5 * cfg.a1 * cfg.tau**2 * cfg.ell *
                  aux.galerkin_stencils(cfg.N, tild_v[k - 1], operator="first-order")
                - u1_integr
                + 0.5 * cfg.tau**2 * q_prev * diff2u[k]
            )
            b2 = (2 * a0 / cfg.ell**2) * (
                cfg.tau**2 * f2_integr[k]
                + 0.5 * cfg.ell**2 * aux.galerkin_stencils(cfg.N, tild_v[k - 1])
                + 0.5 * cfg.a2 * cfg.tau**2 * cfg.ell *
                  aux.galerkin_stencils(cfg.N, tild_u[k - 1], operator="first-order")
                - (1 + 0.5 * cfg.tau**2 * cfg.delta) * v1_integr
                + 0.5 * cfg.tau**2 * cfg.gamma * diff2v[k]
            )

        else:
            # General step: use previous modal data for recursion
            b1 = (
                (4 * cfg.tau**2 / cfg.ell**2) * f1_integr[k]
                + 2 * aux.galerkin_stencils(cfg.N, tild_u[k - 1])
                - (2 * cfg.a1 * cfg.tau**2 / cfg.ell) *
                  aux.galerkin_stencils(cfg.N, tild_v[k - 1], operator="first-order")
            )
            b2 = (
                (2 * a0 * cfg.tau**2 / cfg.ell**2) * f2_integr[k]
                + a0 * aux.galerkin_stencils(cfg.N, tild_v[k - 1])
                + (a0 * cfg.a2 * cfg.tau**2 / cfg.ell) *
                  aux.galerkin_stencils(cfg.N, tild_u[k - 1], operator="first-order")
            )

        # --- Compute condition numbers of matrices at this step ---
        cond_u[k] = aux.condition_number_associated_matrix(
            cfg.N, cfg.ell, 1, 0.5 * cfg.tau**2 * q_prev
        )
        cond_v[k] = aux.condition_number_associated_matrix(
            cfg.N, cfg.ell, 1 + 0.5 * cfg.tau**2 * cfg.delta, 0.5 * cfg.tau**2 * cfg.gamma
        )

        # --- Solve linear systems to get modal coefficients ---
        tild_u[k] = aux.sys_soln(b1, cfg.N, 1, 0.5 * cfg.tau**2 * q_prev, cfg.ell)
        tild_v[k] = aux.sys_soln(
            b2, cfg.N, 1 + 0.5 * cfg.tau**2 * cfg.delta, 0.5 * cfg.tau**2 * cfg.gamma, cfg.ell
        )

        # --- Apply leapfrog-type correction ---
        if k >= 2:
            tild_u[k] -= tild_u[k - 2]
            tild_v[k] -= tild_v[k - 2]

        # --- Update q for next step using new solution ---
        q_prev = cfg.alpha + cfg.beta * np.dot(tild_u[k], tild_u[k])  # Nonlinear update

    return tild_u, tild_v, cond_u, cond_v