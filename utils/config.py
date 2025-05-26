import numpy as np

""" Temporal and spatial domain definitions """
T = 1.0      # Time interval length
ell = 2.0    # Spatial interval length

""" Equation coefficients """
alpha, beta, gamma, delta = 1.0, 1.0, 1.0, 1.0
a1, a2 = 1.0, 1.0

""" Temporal discretization setup """
n = 10
t = np.linspace(0, T, n + 1)
tau = T / n

""" Number of basis functions (spectral order) """
N = 5