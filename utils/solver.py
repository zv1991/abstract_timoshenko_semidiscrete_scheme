# --- Imports ---
import numpy as np  # Used for array creation, dot product, and numerical operations

# Local utility modules
import utils.config as cfg          # Configuration file holding constants like tau, N, ell, delta, etc.
import utils.auxiliary as aux       # Contains helper functions: numerical integration, projections, Galerkin operations

def solve_system(data, f1, f2, *, h=1e-3, derivmeth='nd', tol=1e-6, method='hglq',
                 max_n=50, max_depth=20, n_points=10):
    """
    Solve a coupled PDE system using an explicit Galerkin method.

    Parameters:
        data (dict): Dictionary containing 'u_initial' and 'v_initial' as callable initial functions.
        f1, f2 (callable): Source term functions f1(x, t), f2(x, t).
        h (float, optional): Step size for numerical differentiation (default 1e-3).
        derivmeth (str, optional): Method for derivatives, e.g., 'nd' (default).
        tol (float, optional): Absolute error tolerance for quadrature (default 1e-6).
        method (str, optional): Quadrature method ('glq', 'hglq', 'scipy') (default 'hglq').
        max_n (int, optional): Maximum order for Gauss-Legendre quadrature (default 50).
        max_depth (int, optional): Maximum recursion depth for hierarchical Gauss-Legendre (default 20).
        n_points (int, optional): Number of points per interval for hierarchical Gauss-Legendre (default 10).

    Returns:
        tuple:
            tild_u (ndarray): Modal coefficients for u over time (shape: (time_steps, modes)).
            tild_v (ndarray): Modal coefficients for v over time.
            cond_u (ndarray): Condition numbers of u-system matrices at each time step.
            cond_v (ndarray): Condition numbers of v-system matrices at each time step.
    """
    # --- Preprocess integration parameters ---
    quad_kwargs = dict(tol=tol, method=method, max_n=max_n, max_depth=max_depth, n_points=n_points)

    # --- Initial conditions ---
    u_initial = data['u_initial']
    v_initial = data['v_initial']

    # --- Allocate output arrays ---
    tild_u = np.zeros((cfg.n - 1, cfg.N))
    tild_v = np.zeros((cfg.n - 1, cfg.N))
    cond_u = np.zeros(cfg.n - 1)
    cond_v = np.zeros(cfg.n - 1)

    # --- Project source terms onto modal basis at each time step ---
    f1_integr = aux.compute_time_dependent_integrals(f1, cfg.N, cfg.ell, cfg.t, **quad_kwargs)
    f2_integr = aux.compute_time_dependent_integrals(f2, cfg.N, cfg.ell, cfg.t, **quad_kwargs)

    # --- Compute projections of initial conditions and spatial derivatives ---
    init_data = aux.compute_initial_integrals(
        u_initial, v_initial, cfg.N, cfg.ell,
        h=h, derivmeth=derivmeth, **quad_kwargs
    )
    u0_integr, u1_integr = init_data['u_proj']  # u initial position and velocity projections
    v0_integr, v1_integr = init_data['v_proj']  # v initial position and velocity projections
    diff1u1 = init_data['diff1_u1']             # First spatial derivative of u at t=0
    diff1v1 = init_data['diff1_v1']             # First spatial derivative of v at t=0
    diff2u = init_data['diff2_u']               # Second spatial derivative of u over modes
    diff2v = init_data['diff2_v']               # Second spatial derivative of v over modes

    # --- Precompute scalar constants for matrix terms ---
    a0 = 4 / (2 + cfg.delta * cfg.tau**2)

    # --- Initialize nonlinear term q using integral of squared spatial derivative of initial function u ---
    integral, _ = aux.integrate_fprime_sq(u_initial[1], cfg.ell)
    q_prev = cfg.alpha + cfg.beta * integral

    # --- Main time-stepping loop ---
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
            # --- Special Case: Time step 3 ---
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
            # --- General Case: Time step greater than or equal to 4 ---
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

        # --- Diagnostics: compute condition numbers for matrices being solved ---
        cond_u[k] = aux.condition_number_associated_matrix(cfg.N, cfg.ell, 1, 0.5 * cfg.tau**2 * q_prev)
        cond_v[k] = aux.condition_number_associated_matrix(cfg.N, cfg.ell, 1 + 0.5 * cfg.tau**2 * cfg.delta, 0.5 * cfg.tau**2 * cfg.gamma)

        # --- Solve linear systems for u and v modal coefficients ---
        tild_u[k] = aux.sys_soln(b1, cfg.N, 1, 0.5 * cfg.tau**2 * q_prev, cfg.ell)
        tild_v[k] = aux.sys_soln(b2, cfg.N, 1 + 0.5 * cfg.tau**2 * cfg.delta, 0.5 * cfg.tau**2 * cfg.gamma, cfg.ell)

        # --- Apply correction for k >= 2 to account for previous time step values ---
        if k >= 2:
            tild_u[k] -= tild_u[k - 2]
            tild_v[k] -= tild_v[k - 2]

        # --- Update nonlinear term q for next iteration using squared norm of current modal coeffs of u ---
        q_prev = cfg.alpha + cfg.beta * np.dot(tild_u[k], tild_u[k])

    return tild_u, tild_v, cond_u, cond_v