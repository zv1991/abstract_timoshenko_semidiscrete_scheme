# ======================================================
# MODULE: registry.py
# PURPOSE: Centralized mapping of test names to configuration modules and symbolic testcases.
#          Provides dynamic retrieval of configuration and benchmark classes based on a string identifier.
# ======================================================

# ======================================================
# MODULE OVERVIEW
# ======================================================
# This module exposes a single public function:
#     get_testcase(name: str) -> (cfg_module, testcase_instance)
#
# The internal registry maps string identifiers (e.g., "test1") to a tuple of:
#     - Configuration module with physical and numerical parameters
#     - Instantiated test class with symbolic data (fields, derivatives, etc.)
#
# To avoid redundant allocations, a module-level dictionary is preconstructed using lambdas
# for lazy instantiation of each test pair.

# ======================================================
# CONFIGURATION MODULE IMPORTS — Physical & Numerical Settings
# ======================================================
# Each module (config_testX) defines:
#     - Physical coefficients: alpha, beta, gamma, delta, a1, a2
#     - Domain/time info: ell, T
#     - Discretization: N (spatial), n, tau (time)
#     - Oscillatory parameters (lam, lam1, A)

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
import setting.config_test10 as cfg10  # Testcase10: Unknown solution: Gaussian-initial sine wave
import setting.config_test11 as cfg11  # Testcase11: Unknown solution: sinusoidal spatial-only fields
import setting.config_test12 as cfg12  # Testcase12: Sinusoidal space + time-dependent amplitude

# ======================================================
# TESTCASE CLASS IMPORTS — Symbolic Solution Definitions
# ======================================================
# Each class provides:
#     - Symbolic fields: u(x,t), v(x,t)
#     - Derivatives: ∂u/∂x, ∂²v/∂x², etc.
#     - Forcing/source terms
#     - Initial/boundary condition logic

from tests.test0 import Testcase0
from tests.test1 import Testcase1
from tests.test2 import Testcase2
from tests.test3 import Testcase3
from tests.test4 import Testcase4
from tests.test5 import Testcase5
from tests.test6 import Testcase6
from tests.test7 import Testcase7
from tests.test8 import Testcase8
from tests.test9 import Testcase9
from tests.test10 import Testcase10
from tests.test11 import Testcase11
from tests.test12 import Testcase12

# ======================================================
# PUBLIC EXPORTS — Explicit control of exposed API
# ======================================================

__all__ = ["get_testcase"]

# ======================================================
# INTERNAL REGISTRY — Maps string keys to lazy constructors
# ======================================================
# Each lambda defers construction until called — minimizes load time and memory usage.

_TEST_REGISTRY = {
    "test0":  lambda: (cfg0,  Testcase0(cfg0)),
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
    "test12": lambda: (cfg12, Testcase12(cfg12)),
    # Template for future extensions:
    # "testX": lambda: (cfgX, TestcaseX(cfgX)),
}

# ======================================================
# FUNCTION: get_testcase
# PURPOSE : Retrieve config + testcase instance by string identifier
# ======================================================

# === Method Title: get_testcase — Registry dispatcher for (config, testcase) pairs ===
def get_testcase(name: str):
    """
    Retrieve the configuration module and symbolic test class for a given test name.

    Parameters
    ----------
    name : str
        String identifier of the test case (e.g., "test3", "test7", etc.)

    Returns
    -------
    tuple
        (cfg_module, testcase_instance)
        - cfg_module         : Imported module defining coefficients, domain, and time config
        - testcase_instance  : Instantiated class defining symbolic fields, derivatives, and sources

    Raises
    ------
    ValueError
        If the provided test name is not registered.
    """
    # --------------------------------------
    # Validate the requested name
    # --------------------------------------
    factory = _TEST_REGISTRY.get(name)
    if factory is None:
        available = ", ".join(sorted(_TEST_REGISTRY.keys()))
        raise ValueError(
            f"Unknown test name: '{name}'. Available options: {available}"
        )

    # --------------------------------------
    # Return the (cfg, testcase) pair
    # --------------------------------------
    return factory()  # Lazy instantiation from lambda