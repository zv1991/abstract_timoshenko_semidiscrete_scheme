# ======================================================
# MODULE: registry.py
# PURPOSE: Centralized mapping of test names to configuration modules and symbolic testcases.
# Provides dynamic retrieval of configuration and benchmark classes based on a string identifier.
# ======================================================

# ======================================================
# CONFIGURATION MODULES — PHYSICAL & NUMERICAL PARAMETERS
# ======================================================
# Each configuration module defines:
#   - Physical coefficients: α, β, γ, δ, a₁, a₂
#   - Domain size (ℓ), total simulation time (T)
#   - Discretization parameters: basis functions (N), time steps (n), time step size (τ)
#   - Symbolic tuning: spatial frequency, Gaussian width/amplitude, etc.

import setting.config_test0 as cfg0  # Testcase0 — Constant or linear field (debug baseline)
import setting.config_test1 as cfg1  # Testcase1 — Linear time scaling with Legendre modes
import setting.config_test2 as cfg2  # Testcase2 — Nonlinear system with sinusoidal forcing
import setting.config_test3 as cfg3  # Testcase3 — Oscillatory benchmark: sinusoids in x and t
import setting.config_test4 as cfg4  # Testcase4 — Variant of Testcase3 with tunable wave freq
import setting.config_test5 as cfg5  # Testcase5 — Polynomial time × Legendre spatial basis
import setting.config_test6 as cfg6  # Testcase6 — Exponentially growing sinusoidal system
import setting.config_test7 as cfg7  # Testcase7 — Unknown manufactured solution test case

# ======================================================
# SYMBOLIC TEST CASE CLASSES — Exact Solutions & Sources
# ======================================================
# Each class implements symbolic expressions for:
#   - Fields: u(x, t), v(x, t)
#   - Derivatives: ∂u/∂x, ∂²u/∂t², ...
#   - Sources: f₁(x, t), f₂(x, t)
#   - Initial and boundary conditions

from tests.test0 import Testcase0  # Basic symbolic case (constant or linear fields)
from tests.test1 import Testcase1  # Mild trigonometric + time-varying amplitude
from tests.test2 import Testcase2  # Sinusoidal displacement, nonlinear damping
from tests.test3 import Testcase3  # Oscillatory x–t sine functions
from tests.test4 import Testcase4  # Frequency-parametric extension of Testcase3
from tests.test5 import Testcase5  # Spatial Legendre modes × polynomial time dynamics
from tests.test6 import Testcase6  # Exponential time-amplitude sinusoidal fields
from tests.test7 import Testcase7  # Non-symbolic reference test for solver validation

# ======================================================
# FUNCTION: get_testcase
# PURPOSE : Dispatcher that returns the (cfg, testcase) pair for a given test identifier.
# ======================================================

def get_testcase(name: str):
    """
    Retrieve configuration and symbolic test class for a given test name.

    Parameters
    ----------
    name : str
        Identifier for the test case (e.g., "test0", ..., "test7").

    Returns
    -------
    tuple
        (cfg, testcase) where:
            cfg : module
                Configuration module with physical and numerical settings.
            testcase : Testcase object
                Instance of the corresponding symbolic test case class.

    Raises
    ------
    ValueError
        If the name is not found in the registry.
    """

    # --------------------------------------------------
    # REGISTRY: Mapping from test names to (cfg, class)
    # --------------------------------------------------
    test_registry = {
        "test0": lambda: (cfg0, Testcase0(cfg0)),
        "test1": lambda: (cfg1, Testcase1(cfg1)),
        "test2": lambda: (cfg2, Testcase2(cfg2)),
        "test3": lambda: (cfg3, Testcase3(cfg3)),
        "test4": lambda: (cfg4, Testcase4(cfg4)),
        "test5": lambda: (cfg5, Testcase5(cfg5)),
        "test6": lambda: (cfg6, Testcase6(cfg6)),
        "test7": lambda: (cfg7, Testcase7(cfg7)),
        # Extend here for new test cases:
        # "test8": lambda: (cfg8, Testcase8(cfg8)),
    }

    # --------------------------------------------------
    # VALIDATION: Check if test name exists
    # --------------------------------------------------
    if name not in test_registry:
        raise ValueError(
            f"Unknown test name: '{name}'. "
            f"Available options are: {list(test_registry.keys())}"
        )

    # --------------------------------------------------
    # DISPATCH: Return selected configuration and testcase
    # --------------------------------------------------
    return test_registry[name]()