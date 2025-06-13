# =========================
# IMPORT MODULES
# =========================

# Configuration parameters (e.g., domain size, simulation duration, resolution, and physical coefficients)
import utils.config as cfg

# Symbolic derivatives and exact solutions for u(x, t), v(x, t), and their source terms
from utils.symbolic_derivatives import SymbolicDerivatives as SD

# Solver class for the Timoshenko beam system using the Galerkin method
from utils.class_timoshenko import TimoshenkoModelSolver


# =========================
# UTILITY: TAYLOR EXPANSION
# =========================

def taylor_expansion(func0, func1, func2):
    """
    Constructs a second-order Taylor series approximation of a function evaluated at time τ.
    
    This is useful for approximating initial values at time τ using the known values and 
    derivatives at time t = 0:
    
        f(τ) ≈ f(0) + τ·f'(0) + (τ²/2)·f''(0)

    Args:
        func0 (callable): Function representing f(x, 0), the value of f at t = 0
        func1 (callable): Function representing f'(x, 0), the first time derivative of f at t = 0
        func2 (callable): Function representing f''(x, 0), the second time derivative of f at t = 0

    Returns:
        callable: A lambda function representing f(x, τ) as a second-order Taylor approximation
    """

    # Return the approximated function at time τ
    return lambda x: (
        func0(x)                             # f(0)
        + cfg.tau * func1(x)                 # τ·f'(0)
        + 0.5 * cfg.tau**2 * func2(x)        # (τ²/2)·f''(0)
    )


# =========================
# INITIAL DATA CONSTRUCTOR
# =========================

def get_initial_data():
    """
    Constructs the symbolic source terms and initial conditions for the 
    Timoshenko beam system using symbolic derivatives from the SD class.
    
    The function provides:
      - Exact source terms `f1`, `f2` from symbolic PDE expressions
      - Initial values u(x,0), v(x,0)
      - Approximated values at time τ using 2nd-order Taylor expansion

    Returns:
        f1 (callable): Source term for u-equation (function of x, t)
        f2 (callable): Source term for v-equation (function of x, t)
        u0 (callable): Initial condition u(x, 0)
        u1 (callable): Taylor-approximated u(x, τ)
        v0 (callable): Initial condition v(x, 0)
        v1 (callable): Taylor-approximated v(x, τ)
    """
    
    # ----------------------------------------
    # Symbolic exact solutions for displacements
    # ----------------------------------------
    u = lambda x, t: SD.u(x, t)  # Exact solution for u(x, t)
    v = lambda x, t: SD.v(x, t)  # Exact solution for v(x, t)

    # ----------------------------------------
    # Symbolic source (forcing) terms
    # ----------------------------------------
    f1 = lambda x, t: SD.f1(x, t)  # Forcing term for u-equation
    f2 = lambda x, t: SD.f2(x, t)  # Forcing term for v-equation

    # ----------------------------------------
    # Initial conditions for u(x, t)
    # ----------------------------------------
    varphi0 = lambda x: u(x, 0)  # Initial displacement u(x, 0)
    varphi1 = lambda x: SD.diff1t_u(x, 0)  # Initial velocity ∂u/∂t at t = 0

    # Compute ∂²u/∂t² using PDE rearrangement at t = 0
    varphi2 = lambda x: (
        f1(x, 0)
        - cfg.a1 * SD.diff1x_v(x, 0)  # Coupling with ∂v/∂x
        + (cfg.alpha + cfg.beta * SD.integr_term(0)) * SD.diff2x_u(x, 0)  # Damping & stiffness
    )

    # ----------------------------------------
    # Initial conditions for v(x, t)
    # ----------------------------------------
    psi0 = lambda x: v(x, 0)  # Initial displacement v(x, 0)
    psi1 = lambda x: SD.diff1t_v(x, 0)  # Initial velocity ∂v/∂t at t = 0

    # Compute ∂²v/∂t² using PDE rearrangement at t = 0
    psi2 = lambda x: (
        f2(x, 0)
        + cfg.a2 * SD.diff1x_u(x, 0)  # Coupling with ∂u/∂x
        + cfg.gamma * SD.diff2x_v(x, 0)  # Stiffness term
        - cfg.delta * psi0(x)  # Damping term
    )

    # ----------------------------------------
    # Construct Taylor-expanded approximations at t = τ
    # ----------------------------------------
    u0 = varphi0  # u(x, 0)
    u1 = taylor_expansion(varphi0, varphi1, varphi2)  # u(x, τ) ≈ 2nd-order Taylor expansion

    v0 = psi0  # v(x, 0)
    v1 = taylor_expansion(psi0, psi1, psi2)  # v(x, τ) ≈ 2nd-order Taylor expansion

    # Return source terms and initial data
    return f1, f2, u0, u1, v0, v1


# =========================
# OBTAIN FORCING TERMS AND INITIAL DATA
# =========================

f1, f2, u0, u1, v0, v1 = get_initial_data()


# =========================
# SOLVER INITIALIZATION
# =========================

# Create and configure the solver instance
TimoshenkoModelObject = TimoshenkoModelSolver(
    ell=cfg.ell,                # Spatial domain length
    T=cfg.T,                    # Final simulation time
    alpha=cfg.alpha, beta=cfg.beta,  # u-equation parameters
    gamma=cfg.gamma, delta=cfg.delta,  # v-equation parameters
    a1=cfg.a1, a2=cfg.a2,       # Coupling coefficients between u and v
    n=cfg.n, N=cfg.N,           # Discretization: number of space/time intervals
    f1=f1, f2=f2,               # Source (forcing) functions
    u0=u0, u1=u1,               # Initial conditions for u
    v0=v0, v1=v1                # Initial conditions for v
)

# =========================
# GALERKIN RECONSTRUCTION
# =========================

# Compute Galerkin approximation for u(x, t) over a uniform spatial discretization
# unif_prt_spc = 4 means evaluating at 5 equally spaced points in space
gal_approx_u = TimoshenkoModelObject.galerkin_approx_u(4)
cond_u = TimoshenkoModelObject.cond_u

# (Optional) You can similarly call:
# gal_approx_v = TimoshenkoModelObject.galerkin_approx_v(unif_prt_spc=4)
# or
# single_value_u = TimoshenkoModelObject.galerkin_approx_u(x_val=1.0, k=10)