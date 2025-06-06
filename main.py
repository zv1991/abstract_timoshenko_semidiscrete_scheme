# --- Import project modules for initial conditions, equations, and solver ---
import utils.initial_conditions as ic  # Provides setup_initial_conditions() with u and v initial data
import utils.equations as eqs          # Provides source term functions f1(x, t), f2(x, t)
import utils.solver as soln            # Contains solve_system() implementing Galerkin PDE solver

# --- Prepare initial data: dictionaries of initial displacement and velocity functions for u and v ---
data = ic.setup_initial_conditions()

# --- Solve the PDE system: returns modal coefficients and matrix condition numbers over time ---
tild_u, tild_v, cond_u, cond_v = soln.solve_system(data, eqs.f1, eqs.f2)

# --- Optional: diagnostics or analysis ---
# print("Final modal coefficients for u:", tild_u[-1])
# print("Final modal coefficients for v:", tild_v[-1])
# print("Max condition number of u system:", np.max(cond_u))
# print("Max condition number of v system:", np.max(cond_v))