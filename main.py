from utils.initial_conditions import setup_initial_conditions
from utils.equations import f1, f2
from utils.solver import solve_system

# Prepare initial data and source terms
data = setup_initial_conditions()

# Run the solver
tild_u, tild_v, cond_u, cond_v = solve_system(data, f1, f2)