import numpy as np

""" Temporal and spatial domain definitions """
T = 1      # Time interval length
ell = 2    # Spatial interval length

""" Equation coefficients """
alpha, beta, gamma, delta = 1, 1, 1, 1
a1, a2 = 1, 1

""" Temporal discretization setup """
n = 10
t = np.linspace(0, T, n + 1)
tau = T / n

""" Number of basis functions (spectral order) """
N = 5