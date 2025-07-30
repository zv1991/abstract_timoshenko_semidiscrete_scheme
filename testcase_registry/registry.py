# ======================================================
# MODULE: registry.py
# PURPOSE: Centralized mapping of test names to configuration modules and symbolic testcases.
# Provides dynamic retrieval of configuration and benchmark classes based on a string identifier.
# ======================================================


# ======================================================
# CONFIGURATION MODULES — PHYSICAL & NUMERICAL PARAMETERS
# ======================================================
# Each imported configuration defines:
#   - Physical coefficients: α, β, γ, δ, a₁, a₂
#   - Spatial domain length (ℓ) and simulation time (T)
#   - Discretization parameters: number of basis functions (N), number of time steps (n), time step size (τ)
#   - Test-specific symbolic tuning parameters (e.g., spatial frequency, amplitude scaling)

import setting.config_test0 as cfg0  # cfg0: Testcase0 — Simple constant/linear solutions (debug baseline)
import setting.config_test1 as cfg1  # cfg1: Testcase1 — Differences of the Legendre polynomials solution with linear temporal multiplier
import setting.config_test2 as cfg2  # cfg2: Testcase2 — Nonlinear system with sinusoidal forcing
import setting.config_test3 as cfg3  # cfg3: Testcase3 — Oscillatory benchmark using sinusoids in x and t
import setting.config_test4 as cfg4  # cfg4: Testcase4 — Variant of Testcase3 with tunable wave frequency
import setting.config_test5 as cfg5  # cfg5: Testcase5 — Legendre spatial basis with polynomial-in-time solutions
import setting.config_test6 as cfg6  # cfg6: Testcase6 — Sinusoidal in x, exponential amplitude growth in t


# ======================================================
# SYMBOLIC TEST CASE CLASSES — Exact Solutions & Sources
# ======================================================
# Each class defines symbolic expressions used for:
#   - Displacement field u(x, t) and rotation field v(x, t)
#   - Spatial and temporal derivatives (∂u/∂x, ∂²u/∂t², etc.)
#   - Source terms f₁(t, x) and f₂(t, x)
#   - Boundary and initial conditions

from tests.test0 import Testcase0  # Testcase0: Basic constant/polynomial field (sanity check)
from tests.test1 import Testcase1  # Testcase1: Mild trigonometric benchmark with temporal variation
from tests.test2 import Testcase2  # Testcase2: Sinusoidal solution with nonlinearity in u-equation
from tests.test3 import Testcase3  # Testcase3: Oscillatory test with spatial and temporal sine waves
from tests.test4 import Testcase4  # Testcase4: Testcase3 variant with parameterized spatial frequencies
from tests.test5 import Testcase5  # Testcase5: Analytical test using Legendre spatial modes and time polynomials
from tests.test6 import Testcase6  # Testcase6: Sinusoidal solution with exponential time dynamics (amplitude-growing)


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
        Identifier of the test case to load. Must be one of:
        'test0', 'test1', ..., 'test6'.

    Returns
    -------
    cfg : module
        Configuration module defining numerical and physical parameters.
    test : object
        Instance of the corresponding Testcase class with symbolic expressions.

    Raises
    ------
    ValueError
        If the test name is not in the supported registry.
    """

    # ----------------------------------------------
    # Registry: maps test name strings to loader lambdas
    # Each entry returns a tuple: (cfg_module, Testcase instance)
    # ----------------------------------------------
    test_registry = {
        "test0": lambda: (cfg0, Testcase0(cfg0)),
        "test1": lambda: (cfg1, Testcase1(cfg1)),
        "test2": lambda: (cfg2, Testcase2(cfg2)),
        "test3": lambda: (cfg3, Testcase3(cfg3)),
        "test4": lambda: (cfg4, Testcase4(cfg4)),
        "test5": lambda: (cfg5, Testcase5(cfg5)),
        "test6": lambda: (cfg6, Testcase6(cfg6)),
        # To extend the suite, add new entries here:
        # "test7": lambda: (cfg7, Testcase7(cfg7)),
    }

    # ----------------------------------------------
    # Validate user input against supported test cases
    # ----------------------------------------------
    if name not in test_registry:
        raise ValueError(
            f"Unknown test name: '{name}'. "
            f"Available options are: {list(test_registry.keys())}"
        )

    # ----------------------------------------------
    # Return configuration and benchmark instance
    # ----------------------------------------------
    return test_registry[name]()