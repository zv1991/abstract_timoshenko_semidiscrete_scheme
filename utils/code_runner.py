# =========================
# IMPORT MODULES
# =========================

# Physical and numerical configuration constants (domain size, time, coefficients, resolution)
from utils.config import ell, T, alpha, beta, gamma, delta, a1, a2, n, N

# Provides symbolic representations of derivatives for the benchmark (exact) solution and source terms
from utils.symbolic_derivatives import SymbolicDerivatives as SD

# Provides symbolic initial conditions for the Timoshenko beam model
import utils.initial_conditions_symb as ic

# Main solver class that implements the Galerkin method for the Timoshenko system
from utils.class_timoshenko import TimoshenkoModelSolver

# Provides numerical operations and array manipulation
import numpy as np


# =========================
# DEFINE EXACT SOLUTIONS (for testing and forcing terms)
# =========================

# Exact benchmark solutions for displacement fields u(x, t) and v(x, t)
u = lambda x, t: SD.u(x, t)
v = lambda x, t: SD.v(x, t)

# Corresponding right-hand side (forcing) functions derived symbolically
f1 = lambda x, t: SD.f1(x, t)  # Source term for equation governing u
f2 = lambda x, t: SD.f2(x, t)  # Source term for equation governing v


# =========================
# INITIAL CONDITIONS
# =========================

# Retrieve symbolic initial conditions: displacement and velocity for both fields
data = ic.setup_initial_conditions()
u_initial = data['u_initial']  # [u0, u1] where u0 = u(x, 0), u1 = ∂u/∂t(x, 0)
v_initial = data['v_initial']  # [v0, v1] where v0 = v(x, 0), v1 = ∂v/∂t(x, 0)

# Define callable initial condition functions for u and v
u0 = lambda x: u_initial[0](x)  # Initial displacement u(x, 0)
u1 = lambda x: u_initial[1](x)  # Initial velocity ∂u/∂t(x, 0)
v0 = lambda x: v_initial[0](x)  # Initial displacement v(x, 0)
v1 = lambda x: v_initial[1](x)  # Initial velocity ∂v/∂t(x, 0)


# =========================
# SPATIAL GRID (optional preview/testing grid)
# =========================

# Generate a coarse spatial grid for testing reconstruction (5 points across the domain)
x = np.linspace(0, ell, 5)


# =========================
# SOLVER INITIALIZATION
# =========================

# Instantiate the solver with all required parameters, forcing terms, and initial conditions
TimoshenkoModelObject = TimoshenkoModelSolver(
    ell=ell,
    T=T,
    alpha=alpha, beta=beta, gamma=gamma, delta=delta,
    a1=a1, a2=a2,
    n=n, N=N,
    f1=f1, f2=f2,
    u0=u0, u1=u1,
    v0=v0, v1=v1
)


# =========================
# GALERKIN RECONSTRUCTION
# =========================

# Compute Galerkin approximation for u(x, t) over a uniform spatial discretization
# unif_prt_spc = 4 means evaluating at 5 equally spaced points in space
gal_approx_u = TimoshenkoModelObject.galerkin_approx_u(x_val=1.0)
print(gal_approx_u)
cond_u = TimoshenkoModelObject.cond_u

# (Optional) You can similarly call:
# gal_approx_v = TimoshenkoModelObject.galerkin_approx_v(unif_prt_spc=4)
# or
# single_value_u = TimoshenkoModelObject.galerkin_approx_u(x_val=1.0, k=10)