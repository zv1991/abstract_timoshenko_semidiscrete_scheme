# ======================================================
# MODULE: registry.py
# PURPOSE: Centralized mapping of test names to configuration modules and symbolic testcases.
# Provides dynamic retrieval of configuration and benchmark classes based on a string identifier.
# ======================================================

# (Optional reading) This module exposes a single public function:
#   get_testcase(name: str) -> (cfg_module, testcase_instance)
# which returns the configuration module and an instantiated testcase for the
# requested benchmark. The mapping itself is stored in a module-level constant
# to avoid re-allocating the dictionary on every call.

# ======================================================
# CONFIGURATION MODULE IMPORTS — Physical & Numerical Settings
# ======================================================
# Each config module defines:
#   - Physical coefficients (alpha, beta, gamma, delta, a1, a2)
#   - Domain settings (ell, T)
#   - Numerical parameters (N, n, tau)
#   - Oscillatory/Gaussian benchmark parameters

import setting.config_test0 as cfg0    # Testcase0: Constant/linear fields — simple baseline
import setting.config_test1 as cfg1    # Testcase1: Linear time scaling + Legendre modes
import setting.config_test2 as cfg2    # Testcase2: Nonlinear system with sinusoidal input
import setting.config_test3 as cfg3    # Testcase3: Sinusoidal spatial/temporal fields
import setting.config_test4 as cfg4    # Testcase4: Adjustable frequency extension of Testcase3
import setting.config_test5 as cfg5    # Testcase5: Legendre spatial × polynomial temporal
import setting.config_test6 as cfg6    # Testcase6: Exponential-in-time sinusoidal fields
import setting.config_test7 as cfg7    # Testcase7: Oscillating Gaussian-modulated sine fields
import setting.config_test8 as cfg8    # Testcase8: Oscillating Gaussian-modulated sine initial condition
import setting.config_test9 as cfg9    # Testcase9: Unknown solution: sinusoidal spatial/temporal fields
import setting.config_test10 as cfg10  # Testcase10: Unknown solution: oscillating Gaussian-modulated sine initial condition
import setting.config_test11 as cfg11  # Testcase11: Unknown solution: sinusoidal (oscillating) spatial initial condition

# ======================================================
# TESTCASE CLASS IMPORTS — Symbolic Solution Definitions
# ======================================================
# Each class defines:
#   - Exact fields u(x,t), v(x,t)
#   - Derivatives and source terms
#   - Initial and boundary conditions

from tests.test0 import Testcase0    # Basic test with constant/linear symbolic fields
from tests.test1 import Testcase1    # Trig functions with time scaling
from tests.test2 import Testcase2    # Sinusoidal benchmark with nonlinear terms
from tests.test3 import Testcase3    # Sinusoids in space and time
from tests.test4 import Testcase4    # Frequency-tunable extension of test3
from tests.test5 import Testcase5    # Polynomial-time Legendre-spatial solution
from tests.test6 import Testcase6    # Time-exponential sinusoidal behavior
from tests.test7 import Testcase7    # Gaussian spatial envelope × time-oscillatory wave
from tests.test8 import Testcase8    # Oscillating Gaussian-modulated sine initial condition; a "wave packet" test.
from tests.test9 import Testcase9    # Sinusoids in space
from tests.test10 import Testcase10  # Oscillating Gaussian-modulated sine initial condition; a "wave packet" test.
from tests.test11 import Testcase11  # Sinusoids in space

# ======================================================
# PUBLIC EXPORTS
# ======================================================
# Keep the module's public API explicit to avoid leaking internals via `from registry import *`.
__all__ = ["get_testcase"]

# ======================================================
# REGISTRY (module-level) — maps string keys to lazy constructors
# ======================================================
# Using lambdas defers testcase construction until requested and avoids
# building the dictionary on every get_testcase() call.
_TEST_REGISTRY = {
    "test0":  lambda: (cfg0,  Testcase0(cfg0)),  # Pair: (config module, testcase instance)
    "test1":  lambda: (cfg1,  Testcase1(cfg1)),
    "test2":  lambda: (cfg2,  Testcase2(cfg2)),
    "test3":  lambda: (cfg3,  Testcase3(cfg3)),
    "test4":  lambda: (cfg4,  Testcase4(cfg4)),
    "test5":  lambda: (cfg5,  Testcase5(cfg5)),
    "test6":  lambda: (cfg6,  Testcase6(cfg6)),
    "test7":  lambda: (cfg7,  Testcase7(cfg7)),
    "test8":  lambda: (cfg8,  Testcase8(cfg8)),
    "test9":  lambda: (cfg9,  Testcase9(cfg9)),
    "test10": lambda: (cfg10, Testcase10(cfg10)),
    "test11": lambda: (cfg11, Testcase11(cfg11)),
    # Template for future additions:
    # "testX": lambda: (cfgX, TestcaseX(cfgX)),
}

# ======================================================
# FUNCTION: get_testcase
# PURPOSE : Dispatcher for returning (cfg, testcase) by name
# ======================================================

# === Method Title: get_testcase — registry dispatcher returning (cfg_module, testcase_instance)
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
    # VALIDATION: Check if requested test name is valid
    # --------------------------------------------------
    # Use dict.get to avoid KeyError and to allow a single lookup.
    factory = _TEST_REGISTRY.get(name)
    if factory is None:
        # Sort for a deterministic, friendly error message across Python versions.
        available = ", ".join(sorted(_TEST_REGISTRY.keys()))  # e.g., "test0, test1, ..., test11"
        raise ValueError(
            f"Unknown test name: '{name}'. Available options are: {available}"
        )

    # --------------------------------------------------
    # DISPATCH: Construct and return the (cfg, testcase) pair
    # --------------------------------------------------
    # Calling the lambda performs lazy instantiation at request time.
    return factory()