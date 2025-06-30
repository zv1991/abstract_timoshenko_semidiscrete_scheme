# --------------------------------------------------------------------------- #
"""
Solves a coupled nonlinear PDE system using a Legendre–Galerkin 
time-stepping scheme and modal basis representation.

The method projects spatial and temporal components onto a basis
of normalized shifted Legendre polynomials φₘ over the domain [0, ell].

Features:
---------
- Uses modal Galerkin projections for initial conditions, source terms,
  and spatial derivatives.
- Supports analytical first derivatives (du_initial, dv_initial) if provided.
- Computes second derivative projections ⟨uᵢ″, φₘ⟩ and ⟨vᵢ″, φₘ⟩ via
  integration by parts.
- Nonlinear term includes squared norm of u₁′.
- Supports adaptive quadrature via 'hglq', 'glq', or 'scipy'.
- Applies leapfrog-type correction in time loop.
- Tracks condition numbers of system matrices at each time step.
"""
# --------------------------------------------------------------------------- #

# ------------------------- Imports ------------------------- #

import numpy as np  # Efficient numerical operations on arrays

# Simulation configuration: time grid, constants, domain parameters
import utils.config as cfg

# Utility functions: projection, quadrature, Galerkin stencils
import utils.auxiliary as aux


def solve_system(
    u_initial, v_initial, f1, f2,
    *, du_initial=None, dv_initial=None,
    h=1e-3, derivmeth='nd', tol=1e-6, method='hglq',
    max_n=50, max_depth=20, n_points=10
):
    """
    Solves a nonlinear system of PDEs using Legendre–Galerkin projection.

    Parameters
    ----------
    u_initial, v_initial : list of callable
        Initial functions: [u₀(x), u₁(x)], [v₀(x), v₁(x)].
    f1, f2 : callable
        Time-dependent source functions f₁(x, t), f₂(x, t).
    du_initial, dv_initial : list of callable or None
        Optional analytical first derivatives: [du₀(x), du₁(x)], etc.
    h : float
        Step size for numerical derivatives (if analytical ones are not given).
    derivmeth : {'nd', 'sfd'}
        Derivative approximation method: 'nd' = numdifftools, 'sfd' = finite differences.
    tol : float
        Integration tolerance for quadrature.
    method : str
        Quadrature method: 'hglq', 'glq', or 'scipy'.
    max_n, max_depth, n_points : int
        Adaptive quadrature parameters.

    Returns
    -------
    tild_u, tild_v : ndarray
        Modal coefficients of u and v at each time step.
    cond_u, cond_v : ndarray
        Condition numbers of system matrices for u and v.
    """

    # ------------------------- Input Validation ------------------------- #

    def is_valid_func_list(lst):
        """Check if input is a list of callable objects."""
        return isinstance(lst, list) and all(callable(f) for f in lst)

    def is_valid_deriv_list(lst):
        """Check if input is a list of callables or None."""
        return isinstance(lst, list) and all(callable(f) or f is None for f in lst)

    u_initial = list(u_initial)
    v_initial = list(v_initial)

    if not (is_valid_func_list(u_initial) and is_valid_func_list(v_initial)):
        raise ValueError("u_initial and v_initial must be lists of callable functions.")

    du_initial = [None] * len(u_initial) if du_initial is None else du_initial
    dv_initial = [None] * len(v_initial) if dv_initial is None else dv_initial

    if not (is_valid_deriv_list(du_initial) and is_valid_deriv_list(dv_initial)):
        raise ValueError("du_initial and dv_initial must be lists of callables or None.")

    if not (len(u_initial) == len(v_initial) == len(du_initial) == len(dv_initial)):
        raise ValueError("All initial and derivative lists must have the same length.")

    # --------------------- Quadrature Configuration --------------------- #

    quad_kwargs = dict(
        tol=tol,
        method=method,
        max_n=max_n,
        max_depth=max_depth,
        n_points=n_points
    )

    # ------------------------- Initialize Storage ------------------------- #

    tild_u = np.zeros((cfg.n - 1, cfg.N))  # Modal coefficients for u
    tild_v = np.zeros((cfg.n - 1, cfg.N))  # Modal coefficients for v
    cond_u = np.zeros(cfg.n - 1)           # Condition number for u's system matrix
    cond_v = np.zeros(cfg.n - 1)           # Condition number for v's system matrix

    # -------- Time-Dependent Source Term Projections (Galerkin integrals) -------- #

    f1_integr = aux.compute_time_dependent_integrals(f1, cfg.N, cfg.ell, cfg.t, **quad_kwargs)
    f2_integr = aux.compute_time_dependent_integrals(f2, cfg.N, cfg.ell, cfg.t, **quad_kwargs)

    # --------------------- Initial Projections (u₀, u₁, etc.) --------------------- #

    init_data = aux.compute_initial_integrals(
        u_initial, v_initial, cfg.N, cfg.ell,
        du=du_initial, dv=dv_initial, h=h, derivmeth=derivmeth, **quad_kwargs
    )

    u0_integr, u1_integr = init_data['u_proj']
    v0_integr, v1_integr = init_data['v_proj']
    diff1u1 = init_data['diff1_u1']
    diff1v1 = init_data['diff1_v1']
    diff2u = init_data['diff2_u']
    diff2v = init_data['diff2_v']

    # ------------------- Initial Nonlinear Term (‖u₁′‖²) ------------------- #

    integral, _ = aux.integrate_derivative_form(
        f=u_initial[1] if du_initial[1] is None else None,
        df=du_initial[1] if du_initial[1] is not None else None,
        ell=cfg.ell, m=None, form='squared',
        h=h, derivmeth=derivmeth, **quad_kwargs
    )
    q_prev = cfg.alpha + cfg.beta * integral  # q₁ = α + β * ‖u₁′‖²

    # ---------------------------- Time Loop ---------------------------- #

    for k in range(cfg.n - 1):

        # ---------- Time-Stepping: Right-hand sides b1, b2 ---------- #

        if k == 0:
            # First time step
            b1 = (4 / cfg.ell**2) * (
                cfg.tau**2 * f1_integr[k] + 2 * u1_integr
                - cfg.a1 * cfg.tau**2 * diff1v1
                - u0_integr + 0.5 * cfg.tau**2 * q_prev * diff2u[k]
            )
            b2 = (8 / (2 + cfg.delta * cfg.tau**2) / cfg.ell**2) * (
                cfg.tau**2 * f2_integr[k] + 2 * v1_integr
                + cfg.a2 * cfg.tau**2 * diff1u1
                - (1 + 0.5 * cfg.tau**2 * cfg.delta) * v0_integr
                + 0.5 * cfg.tau**2 * cfg.gamma * diff2v[k]
            )

        elif k == 1:
            # Second time step
            b1 = (4 / cfg.ell**2) * (
                cfg.tau**2 * f1_integr[k]
                + 0.5 * cfg.ell**2 * aux.galerkin_stencils(cfg.N, tild_u[k - 1])
                - 0.5 * cfg.a1 * cfg.tau**2 * cfg.ell *
                aux.galerkin_stencils(cfg.N, tild_v[k - 1], operator="first-order")
                - u1_integr + 0.5 * cfg.tau**2 * q_prev * diff2u[k]
            )
            b2 = (8 / (2 + cfg.delta * cfg.tau**2) / cfg.ell**2) * (
                cfg.tau**2 * f2_integr[k]
                + 0.5 * cfg.ell**2 * aux.galerkin_stencils(cfg.N, tild_v[k - 1])
                + 0.5 * cfg.a2 * cfg.tau**2 * cfg.ell *
                aux.galerkin_stencils(cfg.N, tild_u[k - 1], operator="first-order")
                - (1 + 0.5 * cfg.tau**2 * cfg.delta) * v1_integr
                + 0.5 * cfg.tau**2 * cfg.gamma * diff2v[k]
            )

        else:
            # General case for k ≥ 2 (fully recursive)
            b1 = (
                (4 * cfg.tau**2 / cfg.ell**2) * f1_integr[k]
                + 2 * aux.galerkin_stencils(cfg.N, tild_u[k - 1])
                - (2 * cfg.a1 * cfg.tau**2 / cfg.ell) *
                aux.galerkin_stencils(cfg.N, tild_v[k - 1], operator="first-order")
            )
            b2 = (
                (8 * cfg.tau**2 / (2 + cfg.delta * cfg.tau**2) / cfg.ell**2) * f2_integr[k]
                + (4 / (2 + cfg.delta * cfg.tau**2)) *
                aux.galerkin_stencils(cfg.N, tild_v[k - 1])
                + (4 * cfg.a2 * cfg.tau**2 / (2 + cfg.delta * cfg.tau**2) / cfg.ell) *
                aux.galerkin_stencils(cfg.N, tild_u[k - 1], operator="first-order")
            )

        # ---------- Solve Galerkin Linear Systems ---------- #

        cond_u[k] = aux.condition_number_associated_matrix(
            cfg.N, cfg.ell, 1, 0.5 * cfg.tau**2 * q_prev
        )
        cond_v[k] = aux.condition_number_associated_matrix(
            cfg.N, cfg.ell,
            1 + 0.5 * cfg.tau**2 * cfg.delta,
            0.5 * cfg.tau**2 * cfg.gamma
        )

        tild_u[k] = aux.sys_soln(b1, cfg.N, 1, 0.5 * cfg.tau**2 * q_prev, cfg.ell)
        tild_v[k] = aux.sys_soln(
            b2, cfg.N,
            1 + 0.5 * cfg.tau**2 * cfg.delta,
            0.5 * cfg.tau**2 * cfg.gamma,
            cfg.ell
        )

        # ---------- Leapfrog Correction ---------- #
        if k >= 2:
            tild_u[k] -= tild_u[k - 2]
            tild_v[k] -= tild_v[k - 2]

        # ---------- Update Nonlinear Coefficient qₖ = α + β * ‖uₖ′‖² ---------- #
        q_prev = cfg.alpha + cfg.beta * np.dot(tild_u[k], tild_u[k])

    return tild_u, tild_v, cond_u, cond_v