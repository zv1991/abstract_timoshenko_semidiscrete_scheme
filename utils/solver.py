# --------------------------------------------------------------------------- #
"""
Numerical Solver for a Coupled Nonlinear PDE System using Legendre–Galerkin Methods.

This solver projects a system of PDEs onto a modal basis of shifted Legendre polynomials
over the domain [0, ell]. It uses time-stepping and nonlinear feedback with 
leapfrog-type correction. It’s ideal for systems requiring spectral accuracy
in space and moderate time stepping.

Key Features:
-------------
- Modal Galerkin projection on Legendre basis (modal accuracy)
- Supports numerical or analytical spatial derivatives
- Handles nonlinearities involving ‖u₁′‖²
- Leapfrog-style temporal integration
- Supports adaptive quadrature via several backends
- Tracks condition numbers of system matrices to assess numerical stability
"""
# --------------------------------------------------------------------------- #

import numpy as np  # Core numerical library for arrays, linear algebra, etc.

# External configuration with domain parameters, coefficients, and time horizon
import utils.config as cfg

# Helper functions for projections, Galerkin stencils, quadrature, etc.
import utils.auxiliary as aux


def solve_system(
    u_initial, v_initial, f1, f2,
    n, N,  # n: number of time steps, N: number of Galerkin basis modes
    *, du_initial=None, dv_initial=None,
    h=1e-3, derivmeth='nd', tol=1e-6, method='hglq',
    max_n=50, max_depth=20, n_points=10
):
    """
    Solves a nonlinear coupled PDE system via modal Galerkin projection.

    Parameters
    ----------
    u_initial, v_initial : list of callable
        Initial states: [u₀(x), u₁(x)] and [v₀(x), v₁(x)]
    f1, f2 : callable
        Time-dependent source terms: f₁(x, t), f₂(x, t)
    n : int
        Number of time steps (≥ 1)
    N : int
        Number of modal Galerkin basis functions (≥ 2)
    du_initial, dv_initial : list of callable or None
        Optional spatial derivatives: [du₀, du₁], etc.
    h : float
        Step size for finite difference if no analytical derivative is given
    derivmeth : {'nd', 'sfd'}
        Derivative computation method
    tol : float
        Tolerance for numerical quadrature
    method : str
        Quadrature method: 'hglq', 'glq', or 'scipy'
    max_n, max_depth, n_points : int
        Parameters for adaptive quadrature integration

    Returns
    -------
    tild_u, tild_v : ndarray
        Modal solution coefficients for u and v across all time steps
    cond_u, cond_v : ndarray
        Condition numbers of the linear system matrices at each time step
    """

    # -------- Input Validation -------- #
    if not isinstance(n, int) or n < 1:
        raise ValueError("Parameter 'n' must be an integer ≥ 1.")
    if not isinstance(N, int) or N < 2:
        raise ValueError("Parameter 'N' must be an integer ≥ 2.")

    # -------- Time Grid and Step Size -------- #
    tau = cfg.T / n                                # Time step size
    t = np.linspace(0, cfg.T, n + 1)               # Discretized time vector

    # -------- Validation for Initial Inputs -------- #
    def is_valid_func_list(lst):
        return isinstance(lst, list) and all(callable(f) for f in lst)

    def is_valid_deriv_list(lst):
        return isinstance(lst, list) and all(callable(f) or f is None for f in lst)

    # Ensure initial inputs are lists of functions
    u_initial = list(u_initial)
    v_initial = list(v_initial)

    if not (is_valid_func_list(u_initial) and is_valid_func_list(v_initial)):
        raise ValueError("u_initial and v_initial must be lists of callable functions.")

    # Fill missing derivative info with None
    du_initial = [None] * len(u_initial) if du_initial is None else du_initial
    dv_initial = [None] * len(v_initial) if dv_initial is None else dv_initial

    if not (is_valid_deriv_list(du_initial) and is_valid_deriv_list(dv_initial)):
        raise ValueError("du_initial and dv_initial must be lists of callables or None.")

    if not (len(u_initial) == len(v_initial) == len(du_initial) == len(dv_initial)):
        raise ValueError("All initial and derivative lists must be of the same length.")

    # -------- Quadrature Settings -------- #
    quad_kwargs = dict(
        tol=tol,
        method=method,
        max_n=max_n,
        max_depth=max_depth,
        n_points=n_points
    )

    # -------- Allocate Solution and Diagnostic Arrays -------- #
    tild_u = np.zeros((n - 1, N))  # Modal coefficients of u
    tild_v = np.zeros((n - 1, N))  # Modal coefficients of v
    cond_u = np.zeros(n - 1)       # Condition numbers of u matrices
    cond_v = np.zeros(n - 1)       # Condition numbers of v matrices

    # -------- Project Time-Dependent Source Terms -------- #
    f1_integr = aux.compute_time_dependent_integrals(f1, N, cfg.ell, t, **quad_kwargs)
    f2_integr = aux.compute_time_dependent_integrals(f2, N, cfg.ell, t, **quad_kwargs)

    # -------- Project Initial Conditions and Derivatives -------- #
    init_data = aux.compute_initial_integrals(
        u_initial, v_initial, N, cfg.ell,
        du=du_initial, dv=dv_initial, h=h,
        derivmeth=derivmeth, **quad_kwargs
    )

    u0_integr, u1_integr = init_data['u_proj']
    v0_integr, v1_integr = init_data['v_proj']
    diff1u1 = init_data['diff1_u1']   # ⟨u₁′, φₘ⟩
    diff1v1 = init_data['diff1_v1']   # ⟨v₁′, φₘ⟩
    diff2u = init_data['diff2_u']     # ⟨u″, φₘ⟩
    diff2v = init_data['diff2_v']     # ⟨v″, φₘ⟩

    # -------- Compute Initial Nonlinear Term ‖u₁′‖² -------- #
    integral, _ = aux.integrate_derivative_form(
        f=u_initial[1] if du_initial[1] is None else None,
        df=du_initial[1] if du_initial[1] is not None else None,
        ell=cfg.ell, m=None, form='squared',
        h=h, derivmeth=derivmeth, **quad_kwargs
    )
    q_prev = cfg.alpha + cfg.beta * integral  # Initial nonlinearity factor

    # ===================== TIME LOOP ===================== #
    for k in range(n - 1):

        # -------- Right-Hand Side Construction -------- #
        if k == 0:
            # First step (special handling)
            b1 = (4 / cfg.ell**2) * (
                tau**2 * f1_integr[k] + 2 * u1_integr
                - cfg.a1 * tau**2 * diff1v1
                - u0_integr + 0.5 * tau**2 * q_prev * diff2u[k]
            )
            b2 = (8 / (2 + cfg.delta * tau**2) / cfg.ell**2) * (
                tau**2 * f2_integr[k] + 2 * v1_integr
                + cfg.a2 * tau**2 * diff1u1
                - (1 + 0.5 * tau**2 * cfg.delta) * v0_integr
                + 0.5 * tau**2 * cfg.gamma * diff2v[k]
            )

        elif k == 1:
            # Second step: includes modal stencils from u and v
            b1 = (4 / cfg.ell**2) * (
                tau**2 * f1_integr[k]
                + 0.5 * cfg.ell**2 * aux.galerkin_stencils(N, tild_u[k - 1])
                - 0.5 * cfg.a1 * tau**2 * cfg.ell *
                  aux.galerkin_stencils(N, tild_v[k - 1], operator="first-order")
                - u1_integr + 0.5 * tau**2 * q_prev * diff2u[k]
            )
            b2 = (8 / (2 + cfg.delta * tau**2) / cfg.ell**2) * (
                tau**2 * f2_integr[k]
                + 0.5 * cfg.ell**2 * aux.galerkin_stencils(N, tild_v[k - 1])
                + 0.5 * cfg.a2 * tau**2 * cfg.ell *
                  aux.galerkin_stencils(N, tild_u[k - 1], operator="first-order")
                - (1 + 0.5 * tau**2 * cfg.delta) * v1_integr
                + 0.5 * tau**2 * cfg.gamma * diff2v[k]
            )

        else:
            # General case: fully recursive
            b1 = (
                (4 * tau**2 / cfg.ell**2) * f1_integr[k]
                + 2 * aux.galerkin_stencils(N, tild_u[k - 1])
                - (2 * cfg.a1 * tau**2 / cfg.ell) *
                  aux.galerkin_stencils(N, tild_v[k - 1], operator="first-order")
            )
            b2 = (
                (8 * tau**2 / (2 + cfg.delta * tau**2) / cfg.ell**2) * f2_integr[k]
                + (4 / (2 + cfg.delta * tau**2)) *
                  aux.galerkin_stencils(N, tild_v[k - 1])
                + (4 * cfg.a2 * tau**2 / (2 + cfg.delta * tau**2) / cfg.ell) *
                  aux.galerkin_stencils(N, tild_u[k - 1], operator="first-order")
            )

        # -------- Solve Linear Galerkin Systems -------- #
        cond_u[k] = aux.condition_number_associated_matrix(N, cfg.ell, 1, 0.5 * tau**2 * q_prev)
        cond_v[k] = aux.condition_number_associated_matrix(
            N, cfg.ell,
            1 + 0.5 * tau**2 * cfg.delta,
            0.5 * tau**2 * cfg.gamma
        )

        tild_u[k] = aux.sys_soln(b1, N, 1, 0.5 * tau**2 * q_prev, cfg.ell)
        tild_v[k] = aux.sys_soln(b2, N,
                                 1 + 0.5 * tau**2 * cfg.delta,
                                 0.5 * tau**2 * cfg.gamma,
                                 cfg.ell)

        # -------- Leapfrog Correction -------- #
        if k >= 2:
            tild_u[k] -= tild_u[k - 2]
            tild_v[k] -= tild_v[k - 2]

        # -------- Update Nonlinearity: qₖ = α + β * ‖uₖ′‖² -------- #
        q_prev = cfg.alpha + cfg.beta * np.dot(tild_u[k], tild_u[k])

    return tild_u, tild_v, cond_u, cond_v