# ======================================================
# MODULE: registry.py
# PURPOSE: Centralized mapping of test names to configuration modules and symbolic testcases.
# Provides dynamic retrieval of configuration and benchmark classes based on a string identifier.
# ======================================================


# ======================================================
# CONFIGURATION MODULE IMPORTS — Physical & Numerical Settings
# ======================================================
# Each config module defines:
#   - Physical coefficients (alpha, beta, gamma, delta, a1, a2)
#   - Domain settings (ell, T)
#   - Numerical parameters (N, n, tau)
#   - Oscillatory/Gaussian benchmark parameters

import setting.config_test0 as cfg0  # Testcase0: Constant/linear fields — simple baseline
import setting.config_test1 as cfg1  # Testcase1: Linear time scaling + Legendre modes
import setting.config_test2 as cfg2  # Testcase2: Nonlinear system with sinusoidal input
import setting.config_test3 as cfg3  # Testcase3: Sinusoidal spatial/temporal fields
import setting.config_test4 as cfg4  # Testcase4: Adjustable frequency extension of Testcase3
import setting.config_test5 as cfg5  # Testcase5: Legendre spatial × polynomial temporal
import setting.config_test6 as cfg6  # Testcase6: Exponential-in-time sinusoidal fields
import setting.config_test7 as cfg7  # Testcase7: Oscillating Gaussian-modulated sine fields


# ======================================================
# TESTCASE CLASS IMPORTS — Symbolic Solution Definitions
# ======================================================
# Each class defines:
#   - Exact fields u(x,t), v(x,t)
#   - Derivatives and source terms
#   - Initial and boundary conditions

from tests.test0 import Testcase0  # Basic test with constant/linear symbolic fields
from tests.test1 import Testcase1  # Trig functions with time scaling
from tests.test2 import Testcase2  # Sinusoidal benchmark with nonlinear terms
from tests.test3 import Testcase3  # Sinusoids in space and time
from tests.test4 import Testcase4  # Frequency-tunable extension of test3
from tests.test5 import Testcase5  # Polynomial-time Legendre-spatial solution
from tests.test6 import Testcase6  # Time-exponential sinusoidal behavior
from tests.test7 import Testcase7  # Gaussian spatial envelope × time-oscillatory wave


# ======================================================
# FUNCTION: get_testcase
# PURPOSE : Dispatcher for returning (cfg, testcase) by name
# ======================================================

def get_testcase(name: str):
    """
    Dispatcher to fetch the configuration module and symbolic testcase instance
    for a given benchmark identifier string (e.g., "test0", "test1", ...).

    Parameters
    ----------
    name : str
        String identifier corresponding to the test case.

    Returns
    -------
    tuple
        (cfg, testcase)
        - cfg      : Configuration module containing physical and numerical settings
        - testcase : Instantiated class with symbolic solutions and source terms

    Raises
    ------
    ValueError
        If the name is not found in the registered test dictionary.
    """

    # --------------------------------------------------
    # REGISTRY: Mapping of test names to lambda functions
    # Each lambda lazily constructs the (cfg, testcase) pair
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
        # Template for future additions:
        # "test8": lambda: (cfg8, Testcase8(cfg8)),
    }

    # --------------------------------------------------
    # VALIDATION: Check if requested test name is valid
    # --------------------------------------------------
    if name not in test_registry:
        raise ValueError(
            f"Unknown test name: '{name}'. "
            f"Available options are: {list(test_registry.keys())}"
        )

    # --------------------------------------------------
    # DISPATCH: Construct and return the (cfg, testcase) pair
    # --------------------------------------------------
    return test_registry[name]()