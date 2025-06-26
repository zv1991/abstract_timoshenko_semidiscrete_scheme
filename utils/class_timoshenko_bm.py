import numpy as np  # For efficient numerical operations, especially on arrays

# Configuration parameters for the beam problem.
# Includes time step size `tau`, domain length `ell`, coefficients (a1, a2, alpha, etc.), and time grid `t`.
import utils.config as cfg

# Symbolic derivatives and expressions for displacement, rotation, and their derivatives.
from utils.symbolic_derivatives import SymbolicDerivatives as SD

# ----------------------------------------
# Benchmark Class Definition
# ----------------------------------------