import utils.initial_conditions as ic
import utils.equations as eqs
import utils.solver as soln

# Prepare initial data and source terms
data = ic.setup_initial_conditions()

# Run the solver
tild_u, tild_v, cond_u, cond_v = soln.solve_system(data, eqs.f1, eqs.f2)