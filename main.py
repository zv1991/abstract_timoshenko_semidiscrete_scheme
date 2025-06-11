# ============================
# Main Script for Solving PDE System using Spectral Galerkin Method
# ============================

# --- Import project-specific modules for initial conditions, symbolic source terms, and PDE solver ---

# Module for setting up initial displacement and velocity functions for variables u and v
import utils.initial_conditions_symb as ic

# Class providing symbolic expressions for the source terms f1(x, t) and f2(x, t)
from utils.symbolic_derivatives import SymbolicDerivatives as SD

# Module implementing the Galerkin solver for the coupled PDE system
import utils.solver as soln


# --- Step 1: Generate initial condition data for both variables u and v ---

# `setup_initial_conditions()` returns a dictionary containing callable initial conditions:
# data["u0"], data["v0"] : initial displacements for u and v
# data["u1"], data["v1"] : initial velocities for u and v
data = ic.setup_initial_conditions()


# --- Step 2: Solve the coupled PDE system using the Galerkin method ---

# `solve_system()` integrates the PDE system over time using modal decomposition.
# It takes initial data and symbolic source functions, and returns:
# - tild_u, tild_v : lists of modal coefficient vectors at each time step
# - cond_u, cond_v : lists of condition numbers of the mass matrices for u and v
tild_u, tild_v, cond_u, cond_v = soln.solve_system(
    data,              # Dictionary of initial condition functions
    SD.f1,             # Source term function for u, f1(x, t)
    SD.f2              # Source term function for v, f2(x, t)
)


# --- Step 3 (Optional): Diagnostics and result inspection ---

# Uncomment the following lines for post-simulation analysis or debugging:

# Print final modal coefficients for u and v (useful for comparing modes or checking convergence)
# print("Final modal coefficients for u:", tild_u[-1])
# print("Final modal coefficients for v:", tild_v[-1])

# Print maximum condition numbers encountered during simulation
# These give insight into numerical stability and the quality of the mass matrix inversion
# print("Max condition number of u system:", max(cond_u))
# print("Max condition number of v system:", max(cond_v))