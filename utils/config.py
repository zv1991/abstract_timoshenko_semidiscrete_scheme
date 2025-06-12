import numpy as np  # For efficient numerical computations and array operations

# ---------------------------- Domain Parameters ----------------------------- #
T = 1.0          # Total simulation time (length of the time interval)
ell = 2.0        # Length of the spatial domain [0, ell]

# -------------------------- Equation Coefficients --------------------------- #
# Coefficients for the coupled PDE/ODE system (customize per physical model)
alpha = 1.0
beta = 1.0
gamma = 1.0
delta = 1.0
a1 = 1.0
a2 = 1.0

# ------------------------- Temporal Discretization -------------------------- #
n = 10                              # Number of uniform time steps
t = np.linspace(0, T, n + 1)        # Time grid: n+1 points including endpoints
tau = T / n                         # Time step size Δt = T / n

# ------------------------- Spectral Method Settings ------------------------- #
N = 40  # Number of Legendre basis functions (spectral order)
