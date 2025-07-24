# ===============================================================
# MODULE IMPORTS
# Purpose : Load all required standard, scientific, external, and
#           project-specific modules used throughout the codebase.
# ===============================================================

# ===============================================================
# STANDARD LIBRARY MODULES
# ===============================================================
import os                                   # For file and directory path operations
import warnings                             # Enables non-blocking runtime warnings
import importlib.util                       # Load Python module dynamically from a file path

# ===============================================================
# CORE SCIENTIFIC LIBRARIES
# ===============================================================
import numpy as np
# Core numerical library: provides array operations, broadcasting, linear algebra, and more

from scipy.special import legendre
# Returns the unshifted Legendre polynomial of specified degree as a callable function

from numpy.polynomial.legendre import leggauss
# Computes Gauss–Legendre quadrature nodes and weights for integration over [-1, 1]

from scipy.sparse import identity, diags, csr_matrix
# Constructs identity matrices, diagonals, and compressed sparse row matrices efficiently

from numpy.linalg import cond
# Computes condition number of a matrix, a measure of numerical sensitivity (stability)

# ===============================================================
# OPTIONAL EXTERNAL TOOL: Numerical Differentiation
# ===============================================================
try:
    import numdifftools as nd
    # Provides numerical derivatives via high-accuracy finite differences
except ImportError:
    raise ImportError(
        "This module requires 'numdifftools'. Please install it using one of the following:\n"
        "  pip install numdifftools\n"
        "or\n"
        "  conda install -c conda-forge numdifftools"
    )

# --------------------------------------------------------------------------- #
""" 
Coefficients arising from inner products of Legendre polynomials and their role 
in the Gauss–Legendre spectral method. This module handles high-precision 
precomputation and retrieval of recurrence coefficients Aₘ, Bₘ, and Cₘ 
for Legendre polynomials.
"""
# --------------------------------------------------------------------------- #

# ===============================================================
# FUNCTION: generate_precomputed_coefficients
# Purpose : Generate and save high-precision values of coeff_A, coeff_B, and coeff_C.
# Output  : A standalone Python module with callable accessors.
# ===============================================================
def generate_precomputed_coefficients(N=100, precision=64, output_dir=None):
    """
    Generate high-precision recurrence coefficients Aₘ, Bₘ, and Cₘ
    used in Legendre polynomial expansions, then write them into
    a self-contained Python module for fast future access.

    Parameters:
        N (int): Maximum index for coeff_B and coeff_C. coeff_A computed up to N+1.
        precision (int): Decimal digit precision for mpmath.
        output_dir (str or None): Folder to save the module. Defaults to '../precomp_coeffs'.
    """
    from mpmath import mp  # High-precision arithmetic from mpmath

    # -------------------------------
    # Resolve output directory path
    # -------------------------------
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "precomp_coeffs"))

    mp.dps = precision  # Set global decimal precision for mpmath

    # -------------------------------
    # Coefficient formulas (Aₘ, Bₘ, Cₘ)
    # -------------------------------

    def compute_A(m):
        """Compute Aₘ = 1 / sqrt(2m + 1)"""
        return 1 / mp.sqrt(2 * m + 1)

    def compute_B(m):
        """Compute Bₘ = Aₘ₋₁ * Aₘ² * Aₘ₊₁"""
        return compute_A(m - 1) * compute_A(m)**2 * compute_A(m + 1)

    def compute_C(m):
        """Compute Cₘ = 2 * Aₘ₋₁² * Aₘ₊₁²"""
        return 2 * compute_A(m - 1)**2 * compute_A(m + 1)**2

    # -------------------------------
    # Generate coefficient dictionaries
    # -------------------------------
    coeff_A = {m: compute_A(m) for m in range(1, N + 2)}   # A: [1, N+1]
    coeff_B = {m: compute_B(m) for m in range(2, N)}       # B: [2, N-1]
    coeff_C = {m: compute_C(m) for m in range(1, N + 1)}   # C: [1, N]

    # -------------------------------
    # Create output directory if needed
    # -------------------------------
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "precomputed_coeffs.py")

    # -------------------------------
    # Write coefficients into Python file
    # -------------------------------
    with open(file_path, "w") as f:
        f.write("# Precomputed high-precision coefficients with callable accessors\n")
        f.write("from mpmath import mpf\n\n")

        # Write ranges and metadata
        f.write(f"_A_range = (1, {N + 1})\n")
        f.write(f"_B_range = (2, {N - 1})\n")
        f.write(f"_C_range = (1, {N})\n")
        f.write(f"_PRECISION = {precision}  # Decimal digits used by mpmath\n\n")

        # Write coefficient dictionaries
        for name, table in [('A', coeff_A), ('B', coeff_B), ('C', coeff_C)]:
            f.write(f"coeff_table_{name} = {{\n")
            for m, val in table.items():
                f.write(f"    {m}: mpf('{val}'),\n")  # Store as string to preserve precision
            f.write("}\n\n")

        # Write accessor functions with bounds checking
        for name in ['A', 'B', 'C']:
            f.write(f"def get_coeff_{name}(m):\n")
            f.write(f"    if not (_{name}_range[0] <= m <= _{name}_range[1]):\n")
            f.write(f"        raise ValueError(f'm must be in range {{_{name}_range}} for coeff_{name}')\n")
            f.write(f"    return coeff_table_{name}[m]\n\n")
            f.write(f"coeff_{name} = get_coeff_{name}\n\n")

    print(f"[INFO] Precomputed coefficients saved to: {file_path}")


# ===============================================================
# FUNCTION: load_coefficients
# Purpose : Load or regenerate high-precision Legendre coefficients.
# Returns : NumPy arrays of Aₘ, Bₘ, Cₘ coefficients.
# ===============================================================
def load_coefficients(N: int = 100, precision: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load high-precision Legendre coefficients (A, B, C) from disk,
    regenerating if higher N or precision is requested.

    Parameters:
        N (int): Maximum index for Bₘ and Cₘ. Aₘ computed up to N + 1.
        precision (int): Decimal digit precision for mpmath.

    Returns:
        tuple of np.ndarray: (coeff_A, coeff_B, coeff_C) with 1-based indexing
    """
    from mpmath import mp  # Used for precision control

    # -------------------------------
    # Locate precomputed module
    # -------------------------------
    precomp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "precomp_coeffs"))
    precomp_file = os.path.join(precomp_dir, "precomputed_coeffs.py")

    # -------------------------------
    # Generate coefficients if file missing
    # -------------------------------
    if not os.path.exists(precomp_file):
        print("[INFO] No precomputed file found. Generating coefficients...")
        generate_precomputed_coefficients(N=N, precision=precision, output_dir=precomp_dir)

    # -------------------------------
    # Dynamically load the coefficient module
    # -------------------------------
    spec = importlib.util.spec_from_file_location("precomputed_coeffs", precomp_file)
    precomp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(precomp)

    # -------------------------------
    # Read metadata from loaded module
    # -------------------------------
    stored_N = precomp._C_range[1]
    stored_precision = precomp._PRECISION

    # -------------------------------
    # Regenerate if required
    # -------------------------------
    if N > stored_N or precision > stored_precision:
        print(f"[INFO] Requested N={N}, precision={precision} exceeds stored N={stored_N}, precision={stored_precision}.")
        print("[INFO] Regenerating precomputed coefficients...")
        generate_precomputed_coefficients(N=N, precision=precision, output_dir=precomp_dir)
        spec.loader.exec_module(precomp)  # Reload after regeneration

    mp.dps = precision  # Ensure precision for downstream operations
    effective_N = min(N, precomp._C_range[1])  # Clamp to available range

    # -------------------------------
    # Extract coefficients into 1-based lists
    # -------------------------------
    A = [None] + [float(precomp.coeff_A(m)) for m in range(1, effective_N + 2)]
    B = [None, None] + [float(precomp.coeff_B(m)) for m in range(2, effective_N)]
    C = [None] + [float(precomp.coeff_C(m)) for m in range(1, effective_N + 1)]

    # -------------------------------
    # Convert to NumPy arrays
    # -------------------------------
    coeff_A = np.array(A, dtype=object)
    coeff_B = np.array(B, dtype=object)
    coeff_C = np.array(C, dtype=object)

    return coeff_A, coeff_B, coeff_C

# ===============================================================
# FUNCTION: safe_load_coefficients
# Purpose : Load coefficients using explicit N and optional precision.
# Note    : Ensures regeneration if N < DEFAULT_N threshold.
# ===============================================================

def safe_load_coefficients(numb_coeffs: int, precision: int = 64):
    """
    Safely load or generate numerical coefficients.

    This function ensures the number of coefficients used (N) is no less than a 
    minimum default value (DEFAULT_N), providing stability for downstream 
    calculations. It bypasses configuration objects and directly uses the given 
    arguments.

    Parameters:
        numb_coeffs (int): Number of coefficients to load or generate.
        precision (int, optional): The precision for coefficient generation, 
                                   typically controls the number of decimal digits. 
                                   Defaults to 64.

    Returns:
        tuple: A tuple containing three arrays of coefficients:
               coeff_A, coeff_B, coeff_C.
    """
    
    DEFAULT_N = 100  # Define the minimum number of coefficients to ensure safety.
    
    # Ensure at least DEFAULT_N coefficients are used.
    # This avoids under-generation and maintains algorithmic compatibility.
    N = max(numb_coeffs, DEFAULT_N)

    # Call the coefficient loader/generator function.
    # 'load_coefficients' should return three coefficient arrays: A, B, and C.
    # This function must be implemented elsewhere in your codebase.
    return load_coefficients(N=N, precision=precision)


# ===============================================================
# DEMONSTRATION / EXECUTION
# Purpose : Load a specified number of coefficients directly using
#           the safe_load_coefficients utility.
# ===============================================================

# Load 500 coefficients using the default precision of 64 decimal digits.
# If the requested number is below the threshold (DEFAULT_N = 100), it would auto-adjust.
# Since 500 > 100, the function will proceed with 500.
coeff_A, coeff_B, coeff_C = safe_load_coefficients(numb_coeffs=500)

# ---------------------------------------------------------------
# Optional: Preview the first 5 values of each coefficient set.
# Useful for debugging or verifying content of the loaded arrays.
# ---------------------------------------------------------------
# print("First 5 Coefficients from A:", coeff_A[:5])
# print("First 5 Coefficients from B:", coeff_B[:5])
# print("First 5 Coefficients from C:", coeff_C[:5])

# --------------------------------------------------------------------------- #
""" Functions related to Legendre polynomials, including their
    differences as employed in spectral basis constructions """
# --------------------------------------------------------------------------- #

def shifted_legendre(m, ell, x):
    """
    Compute the shifted Legendre polynomial P_m(x) in [0, ell].
    """
    x = np.clip(np.asarray(x), 0, ell)  # Ensure x is within valid range
    x_mapped = 2 * x / ell - 1  # Transform x from [0, ell] to [-1, 1]
    return legendre(m)(x_mapped)

# def shifted_legendre(m: int, x, ell) -> float | np.ndarray:
#     """
#     Computes the shifted Legendre polynomial P̃_m(x) over the interval [0, ell]
#     using Bonnet's recursion formula. Internally, the domain is mapped to [-1, 1],
#     the standard domain for Legendre polynomials.

#     Parameters:
#         m (int): Degree of the shifted Legendre polynomial (non-negative).
#         x (ndarray, list, float, or int): Input value(s) in the interval [0, ell].
#                                           Accepts scalars, lists, or NumPy arrays.
#         ell (float or int): Interval length (must be > 0); cast internally to float64.

#     Returns:
#         float or np.ndarray: 
#             - A float if a scalar x was provided.
#             - A NumPy array if x was vector-like.
#             Represents P̃_m(x) evaluated at the given x.
#     """

#     # --- Input Normalization ---
#     ell = np.float64(ell)                      # Convert ell to float64 for consistency
#     x = np.asarray(x, dtype=np.float64)        # Ensure x is a float64 NumPy array (handles scalars, lists, arrays)
#     x = np.clip(x, 0, ell)                     # Ensure all x values lie within [0, ell]

#     is_scalar_input = x.ndim == 0              # Track if input was originally scalar for output formatting

#     # --- Domain Mapping ---
#     x_mapped = 2 * x / ell - 1                 # Shift x from [0, ell] to standard Legendre domain [-1, 1]

#     # --- Base Cases ---
#     if m == 0:
#         result = np.ones_like(x_mapped)        # P̃_0(x) = 1
#     elif m == 1:
#         result = x_mapped                      # P̃_1(x) = x_mapped
#     else:
#         # --- Initialize Recursion for Bonnet's Formula ---
#         P_m_minus_1 = np.ones_like(x_mapped)   # P̃_0(x)
#         P_m_curr = x_mapped                    # P̃_1(x)

#         # --- Bonnet's Recursion ---
#         # Recursively compute P̃_m(x) for m ≥ 2:
#         # P̃_{k+1}(x) = ((2k + 1)x P̃_k(x) - k P̃_{k-1}(x)) / (k + 1)
#         for k in range(1, m):
#             P_m_next = ((2 * k + 1) * x_mapped * P_m_curr - k * P_m_minus_1) / (k + 1)
#             P_m_minus_1, P_m_curr = P_m_curr, P_m_next  # Update for next iteration

#         result = P_m_curr  # Final result after m iterations

#     # --- Return Formatting ---
#     return result.item() if is_scalar_input else result  # Return scalar if input was scalar, else array

# def shifted_legendre(n: int, ell: float, x: np.ndarray) -> np.ndarray:
#     """
#     Evaluate the n-th shifted Legendre polynomial on the interval [0, ell].

#     The shifted Legendre polynomial is defined on the domain [0, ell], 
#     but standard Legendre polynomials are defined on the domain [-1, 1]. 
#     This function performs the necessary transformation to map the input x 
#     from [0, ell] to [-1, 1] and evaluates the polynomial using its coefficients.

#     Args:
#         n (int): Degree of the polynomial (non-negative integer).
#         ell (float): The length of the interval [0, ell].
#         x (np.ndarray): Points in the domain [0, ell] where the polynomial should be evaluated.

#     Returns:
#         np.ndarray: Values of the shifted Legendre polynomial evaluated at the points x.
#     """
    
#     # Step 1: Map the input x from the domain [0, ell] to the domain [-1, 1]
#     x_mapped = 2 * x / ell - 1
    
#     # Step 2: Get the coefficients of the standard Legendre polynomial P_n (highest degree first)
#     coeffs = np.array(legendre(n).coef)  # Extract the coefficients of the Legendre polynomial P_n
    
#     # Step 3: Evaluate the polynomial using jax.numpy.polyval, which performs the evaluation using the coefficients
#     # The `coeffs` array represents the polynomial in the form:
#     #   coeffs[0] * x^n + coeffs[1] * x^(n-1) + ... + coeffs[n-1] * x + coeffs[n]
#     return np.polyval(coeffs, x_mapped)  # Return the evaluated polynomial at the mapped points x

def normalized_shifted_legendre(m, ell, x):
    """
    Computes the normalized shifted Legendre polynomial:
    
        P_m^*(x) = shifted_legendre(m, ell, x) / (A_m * sqrt(ell))
    
    where A_m is a normalization coefficient.

    Parameters:
        m   : Degree of the polynomial
        ell : Scaling factor
        x   : Input value (or array of values)

    Returns:
        Normalized shifted Legendre polynomial evaluated at x
    """
    # Compute the normalization coefficient A_m
    A_m = coeff_A[m]
    
    # Compute the standard shifted Legendre polynomial
    P_m_x = shifted_legendre(m, ell, x)
    
    # Compute the normalized polynomial
    return P_m_x / (A_m * np.sqrt(ell))

def phi_m(m: int, ell: float, x: np.ndarray) -> np.ndarray:
    """
    Compute the m-th Galerkin basis function φ_m(x) defined as:
        φ_m(x) = (sqrt(ell) / 2) * A_m * [P_{m+1}(x) - P_{m-1}(x)]

    Parameters:
    - m (int): Basis function index (m >= 1).
    - ell (float): Length of the interval [0, ell].
    - x (np.ndarray): Input points where φ_m is evaluated.

    Returns:
    - np.ndarray: Evaluated φ_m(x) at each x.
    """
    
    # Ensure m is a valid index (Galerkin basis functions are defined for m >= 1)
    if m < 1:
        raise ValueError("m must be >= 1.")
    
    # Compute the coefficient A_m
    A_m = coeff_A[m]
    
    # Compute shifted Legendre polynomials P_{m+1} and P_{m-1} at x
    P_plus = shifted_legendre(m + 1, ell, x)
    P_minus = shifted_legendre(m - 1, ell, x)
    
    # Evaluate φ_m(x) using the defined formula
    phi_vals = (np.sqrt(ell) / 2) * A_m * (P_plus - P_minus)
    
    return phi_vals

# =============================================================================
# Global Variables (Must be defined externally in the environment)
# =============================================================================

# coeff_B: np.ndarray
#   Recurrence coefficients B_k for the second-derivative term (length ≥ N+3)
# coeff_C: np.ndarray
#   Recurrence coefficients C_k for the main diagonal entries (length ≥ N+3)


# =============================================================================
# Function: sys_soln
# =============================================================================
# Title:
#   Spectral-Galerkin Linear System Solver
# -----------------------------------------------------------------------------
# Purpose:
#   Solves a symmetric, banded linear system arising from the spectral-Galerkin 
#   discretization of a linear PDE. It uses an efficient custom solver that:
#     - Exploits symmetry and structure in the banded matrix
#     - Performs forward elimination to reduce the system
#     - Uses backward substitution to reconstruct the solution
# -----------------------------------------------------------------------------
# Inputs:
#   f    : np.ndarray, right-hand side vector (length N)
#   N    : int, number of equations (must be ≥ 2)
#   a    : float, coefficient for the identity operator
#   b    : float, coefficient for the second-derivative operator
#   ell  : float, physical domain scaling factor (e.g., length of domain)
# -----------------------------------------------------------------------------
# Returns:
#   w    : np.ndarray, solution vector of shape (N,)
# -----------------------------------------------------------------------------
# External Requirements:
#   - Global arrays: coeff_B, coeff_C (defined with length ≥ N+3)
# =============================================================================

def sys_soln(f: np.ndarray, N: int, a: float, b: float, ell: float) -> np.ndarray:
    """
    Solves a symmetric banded linear system using a spectral-Galerkin discretization.

    Parameters:
        f (np.ndarray): Right-hand side vector of shape (N,)
        N (int): Number of equations (must be ≥ 2)
        a (float): Identity matrix coefficient
        b (float): Second-derivative operator coefficient
        ell (float): Physical length scale of the domain

    Returns:
        np.ndarray: Solution vector of shape (N,)

    Raises:
        ValueError: If N < 2
    """

    # =========================================================================
    # STEP 1: Validate Input
    # =========================================================================
    if N < 2:
        raise ValueError("N must be at least 2 for the system to be solvable.")

    # =========================================================================
    # STEP 2: Allocate Arrays for Diagonal, RHS Transform, and Solution
    # =========================================================================
    d = np.empty(N)  # Main diagonal values after transformation
    z = np.empty(N)  # Transformed RHS vector during forward elimination
    w = np.empty(N)  # Final solution vector

    # =========================================================================
    # STEP 3: Initialize Diagonal and RHS Vectors for First Two Entries
    # =========================================================================
    diag_scale = (4 * b) / (a * ell ** 2)  # Common scaling term derived from PDE

    # Initialize the first two diagonal entries using recurrence coefficient C
    d[0] = coeff_C[1] + diag_scale
    d[1] = coeff_C[2] + diag_scale

    # Copy the first two RHS entries directly
    z[0] = f[0]
    z[1] = f[1]

    # =========================================================================
    # STEP 4: Forward Elimination
    # =========================================================================
    # Recursively reduce the system using recurrence relations
    # Each step eliminates a lower diagonal component by modifying d and z

    half_N = (N + 1) // 2  # Half length, ensures correct indexing for even/odd N

    for j in range(2, half_N + 1):
        idx = 2 * (j - 1)  # Even index (0-based): 2, 4, 6, ...

        if idx < N:
            # Update diagonal for even index using recurrence and symmetry
            d[idx] = coeff_C[idx + 1] + diag_scale - (coeff_B[idx] ** 2) / d[idx - 2]
            # Update RHS for even index
            z[idx] = f[idx] + (coeff_B[idx] * z[idx - 2]) / d[idx - 2]

        if idx + 1 < N:
            # Update diagonal for odd index using recurrence and symmetry
            d[idx + 1] = coeff_C[idx + 2] + diag_scale - (coeff_B[idx + 1] ** 2) / d[idx - 1]
            # Update RHS for odd index
            z[idx + 1] = f[idx + 1] + (coeff_B[idx + 1] * z[idx - 1]) / d[idx - 1]

    # =========================================================================
    # STEP 5: Backward Substitution
    # =========================================================================
    # Solve system from last to first using recurrence and modified RHS

    # Initialize last two known values of solution vector
    w[N - 1] = z[N - 1] / d[N - 1]
    w[N - 2] = z[N - 2] / d[N - 2]

    # Work backwards in steps of two to exploit sparsity structure
    for j in range(half_N - 1, 0, -1):
        idx = 2 * (j - 1)  # Even index in reverse: ..., 4, 2, 0

        if idx + 2 < N:
            # Back-substitute to compute even-indexed value
            w[idx] = (z[idx] + coeff_B[idx + 2] * w[idx + 2]) / d[idx]

        if idx + 3 < N:
            # Back-substitute to compute odd-indexed value
            w[idx + 1] = (z[idx + 1] + coeff_B[idx + 3] * w[idx + 3]) / d[idx + 1]

    return w


# =========================================================================== #
#                            Quadrature Method Suite                          #
# =========================================================================== #

# ==========================================================
# Function: gauss_legendre_integral
# Purpose : Numerically integrate a function over [a, b] using Gauss–Legendre quadrature
# ==========================================================
def gauss_legendre_integral(f, a, b, n_gauss):
    """
    Compute the Gauss–Legendre quadrature of a function `f` over the interval [a, b].

    Gauss–Legendre quadrature provides an efficient and highly accurate method for
    numerical integration by evaluating the function at optimal nodes within the interval.

    Parameters:
        f       : callable
                  Function to integrate. Should ideally support NumPy vectorized input.
        a       : float
                  Lower bound of the integration interval.
        b       : float
                  Upper bound of the integration interval.
        n_gauss : int
                  Number of Gauss–Legendre nodes (degree of quadrature).

    Returns:
        float : Approximate integral of `f` over the interval [a, b].
    """

    # --------------------------------------------------
    # STEP 1: Generate Gauss–Legendre nodes and weights
    #         on the standard interval [-1, 1]
    # --------------------------------------------------
    nodes, weights = leggauss(n_gauss)  # Nodes: integration points; Weights: associated weights

    # --------------------------------------------------
    # STEP 2: Affine transformation from [-1, 1] to [a, b]
    # --------------------------------------------------
    mid = 0.5 * (a + b)                 # Midpoint of the interval [a, b]
    half_len = 0.5 * (b - a)            # Half the length of interval
    x_mapped = mid + half_len * nodes   # Transform nodes from [-1, 1] to [a, b]

    # --------------------------------------------------
    # STEP 3: Evaluate the function at the transformed nodes
    # --------------------------------------------------
    try:
        # Try vectorized evaluation for efficiency
        f_vals = np.asarray(f(x_mapped))

        # Validate shape to ensure correct mapping
        if f_vals.shape != x_mapped.shape:
            raise ValueError("Function output shape mismatch. Expected shape: {}".format(x_mapped.shape))

    except Exception:
        # Fallback for functions that do not support vectorized input
        f_vals = np.array([f(xi) for xi in x_mapped])

    # --------------------------------------------------
    # STEP 4: Compute the weighted sum and scale it
    #         to reflect the [a, b] interval
    # --------------------------------------------------
    integral = half_len * np.dot(weights, f_vals)  # Weighted sum scaled by interval length

    return integral

# ==========================================================
# Function: adaptive_gauss_legendre_integrator
# Purpose : Perform adaptive integration over [0, ell] using:
#           1. Increasing Gauss–Legendre node count,
#           2. Subinterval refinement when needed.
# ==========================================================
def adaptive_gauss_legendre_integrator(
    f: callable,
    ell: float,
    tol: float = 1e-6,
    min_dx: float = 1 / 128.0,
    n_gauss: int = 5,
    max_gauss: int = 50
    ) -> tuple[float, float, int, int]:
    """
    Approximate the integral of `f` over the interval [0, ell] using adaptive Gauss–Legendre quadrature.

    Strategy:
        - Start with low node count on full interval and try to converge.
        - If not successful, split the interval into smaller parts.
        - In each subinterval, adaptively increase node count until convergence or limits are reached.

    Parameters:
        f         : callable
                    Function to integrate. Must accept float input.
        ell       : float
                    Upper limit of integration interval [0, ell]. Must be ≥ 0.
        tol       : float, optional
                    Absolute convergence tolerance. Default is 1e-6.
        min_dx    : float, optional
                    Minimum subinterval width before halting refinement. Default is 1/128.
        n_gauss   : int, optional
                    Initial number of Gauss nodes to try. Default is 5.
        max_gauss : int, optional
                    Maximum allowed Gauss nodes per interval. Default is 50.

    Returns:
        tuple:
            - float : Estimated integral value
            - float : Estimated absolute error
            - int   : Number of interval halving iterations
            - int   : Maximum number of Gauss nodes used
    """

    # ------------------------------------------
    # Step 1: Validate input domain
    # ------------------------------------------
    if ell < 0:
        raise ValueError("Parameter 'ell' must be non-negative.")
    if ell == 0:
        return 0.0, 0.0, 0, 0  # Trivial integral

    # ------------------------------------------
    # Step 2: Attempt full interval integration with increasing node count
    # ------------------------------------------
    initial_n_gauss = n_gauss             # Preserve initial value for subinterval reuse
    max_nodes_used = n_gauss              # Track max Gauss nodes used overall
    converged = False                     # Flag for global convergence

    integral_prev = gauss_legendre_integral(f, 0.0, ell, n_gauss)

    while n_gauss + 5 <= max_gauss:
        n_gauss += 5
        integral_curr = gauss_legendre_integral(f, 0.0, ell, n_gauss)

        # Track maximum nodes used
        max_nodes_used = min(max(max_nodes_used, n_gauss), max_gauss)

        # Check convergence based on absolute difference
        if np.abs(integral_curr - integral_prev) < tol:
            estimated_error = np.abs(integral_curr - integral_prev)
            return integral_curr, estimated_error, 0, max_nodes_used

        integral_prev = integral_curr

    # ------------------------------------------
    # Step 3: Adaptive refinement by interval halving
    # ------------------------------------------
    counter = 0            # Number of halving iterations
    prev_total = None      # Store previous estimate for convergence check

    while not converged:
        counter += 1
        n_intervals = 2 ** counter
        dx = ell / n_intervals

        # Stop refinement if interval width is too small
        if dx < min_dx:
            break

        total_integral = 0.0
        converged = True  # Assume convergence unless proven otherwise

        for i in range(n_intervals):
            a = i * dx
            b = (i + 1) * dx

            n_gauss_local = initial_n_gauss
            integral_prev = gauss_legendre_integral(f, a, b, n_gauss_local)
            local_converged = False

            # Try to converge in this subinterval
            while n_gauss_local + 5 <= max_gauss:
                n_gauss_local += 5
                integral_curr = gauss_legendre_integral(f, a, b, n_gauss_local)

                if np.abs(integral_curr - integral_prev) < tol:
                    total_integral += integral_curr
                    local_converged = True
                    break

                integral_prev = integral_curr

            if not local_converged:
                # Accept last estimate even if not converged
                total_integral += integral_curr
                converged = False

            max_nodes_used = min(max(max_nodes_used, n_gauss_local), max_gauss)

        # ------------------------------------------
        # Step 4: Global convergence verification
        # ------------------------------------------
        if converged:
            if prev_total is not None and np.abs(total_integral - prev_total) < tol:
                estimated_error = np.abs(total_integral - prev_total)
                return total_integral, estimated_error, counter, max_nodes_used

            # If first converged estimate, store and continue
            prev_total = total_integral

    # ------------------------------------------
    # Step 5: Return best estimate if convergence not reached
    # ------------------------------------------
    estimated_error = np.abs(total_integral - prev_total) if prev_total is not None else float('inf')
    return total_integral, estimated_error, counter, max_nodes_used

# =============================================================================
# Module: compute_product_integral
# =============================================================================
# Purpose:
#   Compute a weighted integral over the interval [0, ell] of the form:
#
#       ∫₀^ell f(x, *args) × φₘ(x) dx      if multiplier = "galerkin_basis"
#       ∫₀^ell f(x, *args) × P̂ₘ(x) dx      if multiplier = "norm_leg_poly"
#
#   - φₘ(x): Galerkin-style shape function
#   - P̂ₘ(x): Normalized shifted Legendre polynomial
#
#   The integrand is constructed using the user-defined function `f` and a
#   selected basis function. Integration is carried out using adaptive 
#   Gauss–Legendre quadrature with tunable precision and node control.
# =============================================================================

def compute_product_integral(f, ell, m, *args, multiplier="galerkin_basis", **quad_kwargs):
    """
    -----------------------------------------------------------------------------
    Function: compute_product_integral
    -----------------------------------------------------------------------------
    Numerically computes a weighted integral of the form:

        ∫₀^ell f(x, *args) × φₘ(x) dx    or    ∫₀^ell f(x, *args) × P̂ₘ(x) dx

    depending on the basis function selected via the `multiplier` keyword.

    Parameters:
    -----------
    f : callable
        User-defined integrand. Must take `x` as first argument, followed by *args.

    ell : float
        Upper limit of the integration interval. Must be strictly positive.

    m : int
        Basis function index or order (used to generate φₘ or P̂ₘ).

    *args : tuple
        Additional positional arguments passed to `f`.

    multiplier : str, optional
        Determines which basis function is used in the product:
            - "galerkin_basis" → Galerkin-style shape function φₘ(x)
            - "norm_leg_poly"  → Normalized shifted Legendre polynomial P̂ₘ(x)

    **quad_kwargs : dict
        Optional quadrature control parameters:
            - tol       : Absolute error tolerance (default 1e-6)
            - min_dx    : Minimum subinterval width (default 1/128)
            - n_gauss   : Initial number of Gauss–Legendre nodes (default 5)
            - max_gauss : Maximum Gauss nodes allowed (default 50)

    Returns:
    --------
    tuple
        - integral_value   : float, result of the integral
        - convergence_info : float, estimated error from quadrature
    """

    # -------------------------------------------------------------------------
    # Step 1: Input validation for the integration bound
    # -------------------------------------------------------------------------
    if ell <= 0:
        raise ValueError("The integration upper bound 'ell' must be strictly positive.")

    # -------------------------------------------------------------------------
    # Step 2: Basis function selection via dispatch dictionary
    # -------------------------------------------------------------------------
    # Each basis function is a callable that accepts a scalar x.
    basis_dispatch = {
        "galerkin_basis": lambda x: phi_m(m, ell, x),              # Galerkin shape φₘ(x)
        "norm_leg_poly" : lambda x: normalized_shifted_legendre(m, ell, x)  # Normalized P̂ₘ(x)
    }

    # -------------------------------------------------------------------------
    # Step 3: Validate basis function key and retrieve corresponding function
    # -------------------------------------------------------------------------
    if multiplier not in basis_dispatch:
        raise ValueError(
            f"Invalid 'multiplier' value: {multiplier!r}. "
            f"Expected one of: {list(basis_dispatch.keys())}."
        )

    basis_function = basis_dispatch[multiplier]  # Extract basis function generator

    # -------------------------------------------------------------------------
    # Step 4: Define the integrand used in quadrature
    # -------------------------------------------------------------------------
    def integrand(x):
        """
        Combined integrand: product of user-defined function and basis function.
        """
        return f(x, *args) * basis_function(x)

    # -------------------------------------------------------------------------
    # Step 5: Set quadrature defaults and override using user-supplied values
    # -------------------------------------------------------------------------
    default_quadrature_options = {
        'tol': 1e-6,           # Absolute integration tolerance
        'min_dx': 1 / 128.0,   # Minimum step size for subdivision
        'n_gauss': 5,          # Initial number of Gauss nodes per interval
        'max_gauss': 50        # Max nodes allowed for adaptive refinement
    }

    # Merge user-supplied and default quadrature options
    filtered_kwargs = {
        key: quad_kwargs.get(key, default_quadrature_options[key])
        for key in default_quadrature_options
    }

    # -------------------------------------------------------------------------
    # Step 6: Call the adaptive Gauss–Legendre integrator
    # -------------------------------------------------------------------------
    value, error_estimate, _, _ = adaptive_gauss_legendre_integrator(
        integrand,
        ell,
        **filtered_kwargs
    )

    # -------------------------------------------------------------------------
    # Step 7: Return the result and quadrature error estimate
    # -------------------------------------------------------------------------
    return value, error_estimate

# =============================================================================
# Function: compute_time_dependent_integrals
# =============================================================================
# Purpose:
#   Compute a matrix of spatial integrals over time, where each entry is:
#
#       ∫₀^ell f(x, t[k+1]) × φₘ₊₁(x) dx   or   ∫₀^ell f(x, t[k+1]) × P̂ₘ₊₁(x) dx
#
#   for each time step k and basis index m. Commonly used in time-dependent
#   Galerkin or spectral discretizations to project source terms onto basis.
# =============================================================================

def compute_time_dependent_integrals(f, N, ell, t, multiplier="galerkin_basis", **quad_kwargs):
    """
    -----------------------------------------------------------------------------
    Function: compute_time_dependent_integrals
    -----------------------------------------------------------------------------
    Compute weighted spatial integrals across time steps for a function f(x, t),
    using either Galerkin basis functions or normalized shifted Legendre polynomials.

    Parameters
    ----------
    f : callable
        Function of two variables, f(x, t), evaluated at fixed time t[k+1]
        and integrated over spatial variable x ∈ [0, ell].

    N : int
        Number of spatial basis functions to use (1-based indexing from m=1 to m=N).

    ell : float
        Upper bound of the spatial interval [0, ell].

    t : array-like of shape (n,)
        Array of time nodes, where each time interval [t[k], t[k+1]] corresponds
        to a step in the output matrix. Must contain at least two values.

    multiplier : str, optional
        Basis function selection method:
            - "galerkin_basis" → φₘ(x)
            - "norm_leg_poly"  → P̂ₘ(x)

    **quad_kwargs : dict
        Additional arguments passed to `compute_product_integral`, such as:
            - tol, min_dx, n_gauss, max_gauss

    Returns
    -------
    integrals : np.ndarray of shape (n-1, N)
        Matrix of integral values:
            integrals[k, m] ≈ ∫₀^ell f(x, t[k+1]) × φₘ₊₁(x) dx
    """

    # -------------------------------------------------------------------------
    # Step 1: Validate time vector length
    # -------------------------------------------------------------------------
    n = len(t)
    if n < 2:
        raise ValueError("Time array 't' must contain at least two time points.")

    # -------------------------------------------------------------------------
    # Step 2: Allocate output matrix to store results
    # Shape: (n-1 time steps) × (N basis functions)
    # Use float64 for numerical precision
    # -------------------------------------------------------------------------
    integrals = np.zeros((n - 1, N), dtype=np.float64)

    # -------------------------------------------------------------------------
    # Step 3: Compute integrals for each time step and basis function
    # Loop over each time interval [t_k, t_{k+1}] and each spatial basis function
    # -------------------------------------------------------------------------
    for k in range(n - 1):
        t_next = t[k + 1]  # Use time at t_{k+1} for projection

        for m in range(N):
            # Compute integral at time t_{k+1} with φ_{m+1}(x) or P̂_{m+1}(x)
            value, _ = compute_product_integral(
                f,                      # Function to integrate
                ell,                    # Upper limit of spatial domain
                m + 1,                  # Use 1-based indexing for basis function
                t_next,                 # Pass t[k+1] as argument to f(x, t)
                multiplier=multiplier,  # Select basis function type
                **quad_kwargs           # Forward any optional quadrature controls
            )

            # Store result in the integral matrix (error estimate discarded)
            integrals[k, m] = value

    # -------------------------------------------------------------------------
    # Step 4: Return the computed matrix of integrals
    # -------------------------------------------------------------------------
    return integrals

# ===============================================================
# MODULE: Finite Difference Derivative Estimators
# ===============================================================
"""
Provides fourth-order accurate numerical schemes for estimating 
the first derivative of a function using either:
- `numdifftools` (adaptive, black-box style)
- manually coded fourth-order finite differences (forward, backward, central)
"""

# ===============================================================
# Function: _validate_and_prepare_input
# Purpose : Input preprocessing and step size adjustment
# ===============================================================

def _validate_and_prepare_input(x, ell, h_init):
    """
    Validates and preprocesses input values for derivative estimation.

    Parameters:
        x       : float or array-like
                  Point(s) at which the derivative is to be evaluated.
        ell     : float
                  Upper bound of the domain (used to constrain step size).
        h_init  : float
                  Initial finite difference step size.

    Returns:
        tuple:
            - x         : np.ndarray (broadcasted input)
            - is_scalar : bool (True if original input was scalar)
            - h_init    : float (adjusted step size to remain within domain)
    """
    if ell is None:
        raise ValueError("Parameter 'ell' must be specified.")

    is_scalar = np.isscalar(x)
    x = np.atleast_1d(x)

    # Ensure h_init is no larger than 1/4 of the domain span
    while h_init > ell / 4:
        h_init /= 2

    return x, is_scalar, h_init

# ===============================================================
# Function: first_order_derivative_nd
# Purpose : First derivative estimation using numdifftools (4th-order)
# ===============================================================

def first_order_derivative_nd(f, x, ell, h_init=1e-3):
    """
    Estimate the first derivative using numdifftools with a 4th-order scheme.

    Parameters:
        f       : callable
                  Function whose derivative is to be computed.
        x       : float or array-like
                  Evaluation point(s).
        ell     : float
                  Upper domain bound.
        h_init  : float
                  Initial step size.

    Returns:
        tuple:
            - derivative(s) at x : float or np.ndarray
            - h_init              : float (final step size used)
    """
    x, is_scalar, h_init = _validate_and_prepare_input(x, ell, h_init)
    derivs = []

    for xi in x:
        # Select direction of difference based on proximity to boundaries
        if xi - 2 * h_init < 0:
            method = 'forward'
        elif xi + 2 * h_init > ell:
            method = 'backward'
        else:
            method = 'central'

        try:
            df = nd.Derivative(f, n=1, step=h_init, order=4, method=method)
            deriv = df(xi)
        except Exception as e:
            warnings.warn(f"Derivative estimation failed at x={xi:.6f}: {e}")
            deriv = np.nan

        derivs.append(deriv)

    result = np.array(derivs)
    return (result[0] if is_scalar else result), h_init

# ===============================================================
# Function: first_order_derivative
# Purpose : Manual 4th-order finite difference (forward/backward/central)
# ===============================================================

def first_order_derivative(f, x, ell, h_init=1e-3):
    """
    Estimate the first derivative using manually coded 4th-order finite differences.

    Parameters:
        f       : callable
                  Function to differentiate.
        x       : float or array-like
                  Evaluation point(s).
        ell     : float
                  Upper domain bound.
        h_init  : float
                  Initial finite difference step size.

    Returns:
        tuple:
            - derivative(s) at x : float or np.ndarray
            - h_init              : float (step size used)
    """
    x, is_scalar, h_init = _validate_and_prepare_input(x, ell, h_init)
    derivs = []

    for xi in x:
        try:
            if xi - 2 * h_init < 0:
                # Near left boundary: use forward 4th-order difference
                deriv = (-25*f(xi) + 48*f(xi + h_init) - 36*f(xi + 2*h_init)
                         + 16*f(xi + 3*h_init) - 3*f(xi + 4*h_init)) / (12 * h_init)
            elif xi + 2 * h_init > ell:
                # Near right boundary: use backward 4th-order difference
                deriv = (25*f(xi) - 48*f(xi - h_init) + 36*f(xi - 2*h_init)
                         - 16*f(xi - 3*h_init) + 3*f(xi - 4*h_init)) / (12 * h_init)
            else:
                # In the interior: use central 4th-order difference
                deriv = (-f(xi + 2*h_init) + 8*f(xi + h_init)
                         - 8*f(xi - h_init) + f(xi - 2*h_init)) / (12 * h_init)
        except Exception as e:
            warnings.warn(f"Manual finite difference failed at x={xi:.6f}: {e}")
            deriv = np.nan

        derivs.append(deriv)

    result = np.array(derivs)
    return (result[0] if is_scalar else result), h_init

# ===============================================================
# Function: first_order_derivative_unified
# Purpose : Dispatch derivative computation via 'nd' or manual 'sfd'
# ===============================================================

def first_order_derivative_unified(f, x, ell, derivmeth='nd', h_init=1e-3):
    """
    Unified interface for estimating first-order derivatives using either:
        - 'nd'  : numdifftools-based differentiation (default)
        - 'sfd' : standard manually implemented 4th-order finite differences

    Parameters:
        f         : callable
                    Function whose derivative is to be estimated.
        x         : float or array-like
                    Evaluation point(s).
        ell       : float
                    Upper bound of the domain.
        derivmeth : str
                    Derivative method to use: 'nd' or 'sfd'.
        h_init    : float
                    Initial step size for finite differences.

    Returns:
        tuple:
            - derivative(s) at x : float or np.ndarray
            - h_init              : float (final step size used)
    """
    derivmeth = derivmeth.lower()

    if derivmeth == 'nd':
        return first_order_derivative_nd(f, x, ell, h_init=h_init)
    elif derivmeth == 'sfd':
        return first_order_derivative(f, x, ell, h_init=h_init)
    else:
        raise ValueError("Invalid method. Use 'nd' (numdifftools) or 'sfd' (standard finite difference).")

# ==============================================================================
# FUNCTION: integrate_derivative_form
# PURPOSE : Evaluate integrals involving the first derivative f′(x) over [0, ell], using:
#           - 'squared'       → ∫₀^ell [f′(x)]² dx  (energy norm)
#           - 'norm_leg_poly' → ∫₀^ell f′(x) · P̂ₘ(x) dx (Legendre projection)
# ==============================================================================

def integrate_derivative_form(
    f=None,
    df=None,
    ell=None,
    form='squared',
    m=None,
    h=1e-3,
    derivmeth='nd',
    **quad_kwargs
):
    """
    Numerically compute gradient-based integrals over [0, ell] using adaptive quadrature.

    Supported forms:
        - 'squared'       → ∫₀^ell [f′(x)]² dx
        - 'norm_leg_poly' → ∫₀^ell f′(x) · P̂ₘ(x) dx (projection onto Legendre polynomial)

    Parameters
    ----------
    f : callable, optional
        Function f(x). Required unless `df` is provided.

    df : callable, optional
        Analytical derivative function f′(x). Used if available; overrides numerical differentiation.

    ell : float
        Length of the interval domain [0, ell]. Must be a positive number.

    form : {'squared', 'norm_leg_poly'}, default='squared'
        Specifies which integral to compute.

    m : int, optional
        Degree of normalized shifted Legendre polynomial P̂ₘ(x). Required for 'norm_leg_poly'.

    h : float, default=1e-3
        Initial step size for numerical differentiation (only if df is not provided).

    derivmeth : {'nd', 'sfd'}, default='nd'
        Method for numerical differentiation:
            - 'nd'  → external library like `numdifftools`
            - 'sfd' → fourth-order symmetric finite difference

    **quad_kwargs : dict, optional
        Passed to `adaptive_gauss_legendre_integrator`. May include:
            - tol      : float, integration tolerance
            - n_gauss  : int, number of initial Gauss–Legendre points
            - min_dx   : float, minimum subinterval width
            - max_gauss: int, max number of Gauss nodes allowed

    Returns
    -------
    integral : float
        Result of the integral over [0, ell].

    error_estimate : float
        Estimated integration error from the quadrature routine.
    """

    # ==========================================================================
    # INPUT VALIDATION
    # ==========================================================================
    if ell is None or ell <= 0:
        raise ValueError("Parameter 'ell' must be a positive float.")

    if (f is None and df is None) or (f is not None and df is not None):
        raise ValueError("Specify exactly one of 'f' or 'df', not both or neither.")

    if form not in ('squared', 'norm_leg_poly'):
        raise ValueError("Parameter 'form' must be 'squared' or 'norm_leg_poly'.")

    if form == 'norm_leg_poly' and m is None:
        raise ValueError("Parameter 'm' must be specified for 'norm_leg_poly' integrals.")

    if form == 'squared' and m is not None:
        warnings.warn("Parameter 'm' is ignored when form='squared'.", stacklevel=2)

    # ==========================================================================
    # STEP SIZE ADJUSTMENT FOR NUMERICAL DIFFERENTIATION
    # ==========================================================================
    if f is not None:
        while h >= ell / 4:
            h /= 2  # Reduce h if too large relative to the domain (for stability)

    # ==============================================================================
    # INNER FUNCTION: integrand(x)
    # Purpose: Evaluates the integrand [f′(x)]² or f′(x)·P̂ₘ(x)
    # ==============================================================================
    def integrand(x):
        """
        Evaluate integrand at point(s) x for use in adaptive quadrature.

        Parameters
        ----------
        x : float or array-like
            Point(s) in the interval [0, ell] where the integrand is evaluated.

        Returns
        -------
        result : float or ndarray
            Value(s) of the integrand corresponding to each x.
        """
        x = np.atleast_1d(x)              # Convert scalar input to array, if necessary
        result = np.empty_like(x)         # Preallocate output array

        for i, xi in enumerate(x):
            # Compute derivative (analytical or numerical fallback)
            if df is not None:
                f_prime = df(xi)
            else:
                f_prime, _ = first_order_derivative_unified(
                    f, xi, ell=ell, h_init=h, derivmeth=derivmeth
                )

            # Evaluate integrand based on selected form
            if form == 'squared':
                result[i] = f_prime ** 2
            else:  # form == 'norm_leg_poly'
                Pm_val = normalized_shifted_legendre(m, ell, xi)  # Evaluate P̂ₘ(xi)
                result[i] = f_prime * Pm_val

        return result[0] if result.size == 1 else result  # Handle scalar output if needed

    # ==============================================================================
    # NUMERICAL INTEGRATION: Use adaptive Gauss–Legendre integration routine
    # ==============================================================================
    integral, error_estimate, *_ = adaptive_gauss_legendre_integrator(
        integrand, ell, **quad_kwargs
    )

    return integral, error_estimate

# =============================================================================
# Function: compute_initial_integrals
# -----------------------------------------------------------------------------
# Purpose:
#   Compute modal Galerkin coefficients for initial data functions u(x), v(x),
#   and their first and second derivatives over the spatial domain [0, ell],
#   using projections onto normalized shifted Legendre polynomials φₘ(x).
#
#   Supports both analytic and numerical differentiation.
# =============================================================================

def compute_initial_integrals(
    u, v, N, ell, *,
    du=None, dv=None,
    h=1e-3, derivmeth='nd', **quad_kwargs
):
    """
    --------------------------------------------------------------------------
    Method: compute_initial_integrals
    --------------------------------------------------------------------------
    Computes Legendre–Galerkin modal coefficients for u(x), v(x), and their
    first/second derivatives using weighted L² projections.

    Parameters
    ----------
    u, v : list of callable
        Initial condition functions:
            u = [u₀(x), u₁(x)], v = [v₀(x), v₁(x)]

    du, dv : list of callables or None
        Analytic first derivatives. If not provided or partially missing,
        numerical differentiation is used for missing entries.

    N : int
        Number of basis functions used (φ₁ to φ_N)

    ell : float
        Spatial domain length; integral domain is [0, ell]

    h : float, default=1e-3
        Step size for numerical differentiation (when used)

    derivmeth : {'nd', 'sfd'}, default='nd'
        Numerical differentiation method:
            - 'nd'  → Use `numdifftools`
            - 'sfd' → Use custom 4th-order finite difference

    **quad_kwargs : dict
        Passed directly to quadrature methods for integration control

    Returns
    -------
    dict
        Dictionary with arrays of modal coefficients:
            'u_proj', 'v_proj'   : L² projections
            'diff1_u1', 'diff1_v1': First derivative (by parts)
            'diff2_u', 'diff2_v' : Second derivative (numerical or analytic)
    """

    # -------------------------------------------------------------------------
    # Step 1: Helper functions for input validation
    # -------------------------------------------------------------------------
    def is_valid_func_list(lst):
        return isinstance(lst, list) and all(callable(f) for f in lst)

    def is_valid_deriv_list(lst):
        return isinstance(lst, list) and all(callable(f) or f is None for f in lst)

    # -------------------------------------------------------------------------
    # Step 2: Validate input lists and parameter types
    # -------------------------------------------------------------------------
    if not (is_valid_func_list(u) and is_valid_func_list(v)):
        raise ValueError("Inputs 'u' and 'v' must be lists of callable functions.")

    du = [None] * len(u) if du is None else du
    dv = [None] * len(v) if dv is None else dv

    if not (is_valid_deriv_list(du) and is_valid_deriv_list(dv)):
        raise ValueError("Inputs 'du' and 'dv' must be lists of callables or None.")

    if not (len(u) == len(v) == len(du) == len(dv)):
        raise ValueError("Lists 'u', 'v', 'du', and 'dv' must all be the same length.")

    if not isinstance(N, int) or N <= 0:
        raise ValueError("Parameter 'N' must be a positive integer.")

    if not isinstance(ell, (int, float)) or ell <= 0:
        raise ValueError("Parameter 'ell' must be a positive float.")

    # -------------------------------------------------------------------------
    # Step 3: Allocate arrays to hold modal coefficients
    # -------------------------------------------------------------------------
    num_components = len(u)  # Typically 2 (e.g., [u₀, u₁])
    u_proj = [np.zeros(N) for _ in range(num_components)]  # Modal ⟨uᵢ, φₘ⟩
    v_proj = [np.zeros(N) for _ in range(num_components)]  # Modal ⟨vᵢ, φₘ⟩
    diff1_u1 = np.zeros(N)  # ⟨u₁′, φₘ⟩ via integration by parts
    diff1_v1 = np.zeros(N)  # ⟨v₁′, φₘ⟩ via integration by parts
    diff2_u = np.zeros((num_components, N))  # ⟨uᵢ″, φₘ⟩
    diff2_v = np.zeros((num_components, N))  # ⟨vᵢ″, φₘ⟩

    # -------------------------------------------------------------------------
    # Step 4: Loop over basis indices m = 1 to N
    # -------------------------------------------------------------------------
    for m in range(N):
        m_idx = m + 1  # 1-based indexing for Legendre modes

        # ---------------------------------------------------------------------
        # Step 4.1: Compute L² projection coefficients ⟨uᵢ, φₘ⟩ and ⟨vᵢ, φₘ⟩
        # ---------------------------------------------------------------------
        for i in range(num_components):
            u_proj[i][m], _ = compute_product_integral(
                u[i], ell, m_idx,
                multiplier="galerkin_basis",  # Use Galerkin basis φₘ
                **quad_kwargs
            )
            v_proj[i][m], _ = compute_product_integral(
                v[i], ell, m_idx,
                multiplier="galerkin_basis",
                **quad_kwargs
            )

        # ---------------------------------------------------------------------
        # Step 4.2: First derivative projections ⟨u₁′, φₘ⟩ and ⟨v₁′, φₘ⟩
        # Done using integration by parts: ⟨f′, φₘ⟩ = -⟨f, φₘ′⟩
        # ---------------------------------------------------------------------
        diff1_u1[m] = compute_product_integral(
            lambda x: -u[1](x), ell, m_idx,
            multiplier="norm_leg_poly",  # φₘ′ is handled via norm_leg_poly
            **quad_kwargs
        )[0]

        diff1_v1[m] = compute_product_integral(
            lambda x: -v[1](x), ell, m_idx,
            multiplier="norm_leg_poly",
            **quad_kwargs
        )[0]

        # ---------------------------------------------------------------------
        # Step 4.3: Second derivative projections ⟨uᵢ″, φₘ⟩ and ⟨vᵢ″, φₘ⟩
        # Use either analytic first derivative (df), or original f for numeric
        # ---------------------------------------------------------------------
        for i in range(num_components):
            # Handle uᵢ″ projection
            f_u = None if du[i] else (lambda x, i=i: -u[i](x))         # For numerical diff
            df_u = (lambda x, i=i: -du[i](x)) if du[i] else None       # For analytic diff

            diff2_u[i][m], _ = integrate_derivative_form(
                f=f_u, df=df_u, ell=ell, form='norm_leg_poly', m=m_idx,
                h=h, derivmeth=derivmeth, **quad_kwargs
            )

            # Handle vᵢ″ projection
            f_v = None if dv[i] else (lambda x, i=i: -v[i](x))
            df_v = (lambda x, i=i: -dv[i](x)) if dv[i] else None

            diff2_v[i][m], _ = integrate_derivative_form(
                f=f_v, df=df_v, ell=ell, form='norm_leg_poly', m=m_idx,
                h=h, derivmeth=derivmeth, **quad_kwargs
            )

    # -------------------------------------------------------------------------
    # Step 5: Package and return all computed modal coefficients
    # -------------------------------------------------------------------------
    return {
        'u_proj': u_proj,       # ⟨uᵢ, φₘ⟩
        'v_proj': v_proj,       # ⟨vᵢ, φₘ⟩
        'diff1_u1': diff1_u1,   # ⟨u₁′, φₘ⟩
        'diff1_v1': diff1_v1,   # ⟨v₁′, φₘ⟩
        'diff2_u': diff2_u,     # ⟨uᵢ″, φₘ⟩
        'diff2_v': diff2_v      # ⟨vᵢ″, φₘ⟩
    }

# --------------------------------------------------------------------------- #
""" Construction of operator matrices (stencils)
    derived via the Legendre–Galerkin spectral method """
# --------------------------------------------------------------------------- #

# --- Identity Operator Assembly ---
def associated_identity_operator(N: int) -> csr_matrix:
    """
    Assemble a symmetric mass-like identity operator with a 3-point stencil.

    Parameters:
        N (int): Number of basis functions (matrix size).

    Returns:
        csr_matrix: Sparse matrix representing the identity operator.
    """
    # Main diagonal: C(m+1)
    main_diag = np.array([coeff_C[m + 1] for m in range(N)])

    # Off-diagonals: -B(m+2), symmetric about ±2 diagonals
    off_diag = np.array([-coeff_B[m + 2] for m in range(N - 2)])

    # Create sparse matrix with diagonals at positions 0, ±2
    H = diags(
        diagonals=[main_diag, off_diag, off_diag],
        offsets=[0, -2, 2],
        shape=(N, N),
        format="csr"
    )
    return H

# --- First-Order Operator Assembly ---
def associated_first_order_operator(N: int) -> csr_matrix:
    """
    Assemble a skew-symmetric first-order derivative operator using a ±1 stencil.

    Parameters:
        N (int): Number of basis functions (matrix size).

    Returns:
        csr_matrix: Sparse matrix representing the first-order operator.
    """
    # Upper diagonal: A(m+1)*A(m+2), size (N-1)
    upper_diag = np.array([
        coeff_A[m + 1] * coeff_A[m + 2] for m in range(N - 1)
    ])

    # Lower diagonal: Negative of upper for skew-symmetry
    lower_diag = -upper_diag

    # Assemble sparse skew-symmetric matrix
    B = diags(
        diagonals=[lower_diag, upper_diag],
        offsets=[-1, 1],
        shape=(N, N),
        format="csr"
    )
    return B

# --- Operator Dispatcher ---
def associated_operators(N: int, operator: str) -> csr_matrix:
    """
    Dispatcher for assembling different Galerkin operator matrices.

    Parameters:
        N (int): Matrix size (number of basis functions).
        operator (str): Operator type: "identity" or "first-order".

    Returns:
        csr_matrix: Assembled sparse operator matrix.
    
    Raises:
        ValueError: If the operator type is invalid.
    """
    if operator == "identity":
        return associated_identity_operator(N)
    elif operator == "first-order":
        return associated_first_order_operator(N)
    else:
        raise ValueError(f"Unknown operator type '{operator}'. Use 'identity' or 'first-order'.")

# --- Galerkin Operator Application ---
def galerkin_stencils(N: int, v: np.ndarray, operator: str = "identity") -> np.ndarray:
    """
    Apply a Galerkin operator to a vector using sparse matrix multiplication.

    Parameters:
        N (int): Vector size (must match operator matrix size).
        v (np.ndarray): Input vector (shape: (N,)).
        operator (str): Operator type to apply.

    Returns:
        np.ndarray: Output vector A * v.
    
    Raises:
        ValueError: If vector shape doesn't match N or invalid operator.
    """
    if v.shape[0] != N:
        raise ValueError(f"Input vector length mismatch: expected {N}, got {v.shape[0]}.")

    A = associated_operators(N, operator)
    return A.dot(v)

# --------------------------------------------------------------------------- #
""" Computation of the condition number associated with the Galerkin 
    system matrix """
# --------------------------------------------------------------------------- #

def condition_number_associated_matrix(N: int, ell: float, a: float, b: float) -> float:
    """
    Compute the condition number κ₂(A) of a modified Galerkin matrix:
        A = H + (4b / (a * ell²)) * I

    Parameters:
        N (int): Matrix size.
        ell (float): Scaling parameter.
        a (float): Physical coefficient.
        b (float): Additive coefficient.

    Returns:
        float: 2-norm condition number of the matrix.
    """
    # Galerkin identity operator H
    H = associated_operators(N, operator="identity")

    # Identity matrix scaled by constant
    scalar = (4 * b) / (a * ell ** 2)
    I_scaled = scalar * identity(N, format="csr")

    # Final matrix A
    A = H + I_scaled

    # Convert to dense for condition number computation
    return cond(A.toarray(), p=2)

# ----------------------------------------------------------------------
# Function: galerkin_approx
# Purpose : Computes the Galerkin approximation of u(x) using a set of 
#           basis functions φₘ over a spatial domain [0, ell]. The 
#           approximation is formed as a linear combination of φₘ 
#           weighted by provided coefficients for each temporal layer.
# Inputs  : 
#    - ell   : Length of spatial domain (float)
#    - coeff : Coefficient matrix of shape (n, N), where each row 
#              corresponds to a temporal layer
#    - x     : Spatial locations (float or np.ndarray)
# Output  : 
#    - Approximation result at x (scalar or np.ndarray of shape (n, len(x)))
# ----------------------------------------------------------------------

def galerkin_approx(ell: float, coeff: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute the Galerkin approximation:
        u(x) ≈ sum_{m=1}^{N} coeff[m-1] * phi_m(m, ell, x)

    Parameters:
    - ell (float): Length of spatial interval [0, ell] for each basis function phi_m(m, ell, x).
    - coeff (np.ndarray): Coefficient matrix of shape (n, N), typically from solving a Galerkin system.
                          Each row corresponds to a temporal layer.
    - x (float or np.ndarray): Locations at which the approximation is evaluated.

    Returns:
    - np.ndarray or float: Evaluated approximation. Returns a scalar if input x is scalar,
                           otherwise returns an array of shape (n, len(x)).
    """
    
    # Ensure 'ell' is explicitly a float (e.g., if passed as an int)
    ell = float(ell)

    # Convert input arrays to float-type NumPy arrays to avoid unexpected behavior
    coeff = np.asarray(coeff, dtype=float)
    x = np.asarray(x, dtype=float)

    # Determine the number of basis functions (assumes coeff shape is (n, N))
    N = coeff.shape[1]

    # Check if input 'x' is a scalar (to format output accordingly)
    is_scalar = np.isscalar(x)

    # Flatten 'x' to ensure it's always a 1D array for consistent processing
    x = np.atleast_1d(x)

    # Evaluate basis functions φₘ(x) for m = 1 to N
    # Resulting shape: (N, len(x)), each row is phi_m for m in 1..N
    phi_vals = np.array([phi_m(m + 1, ell, x) for m in range(N)])

    # Compute linear combination: matrix multiplication of coeff (n x N) with phi_vals (N x len(x))
    result = coeff @ phi_vals

    # If x was scalar, return a vector (n,) with each row's result at that scalar x
    return result[:, 0] if is_scalar else result

# =============================================================================
# Analytical Solution Evaluation Utilities for Timoshenko Beam Model
# -----------------------------------------------------------------------------
# These two functions allow evaluation and inspection of exact analytical
# solutions (u or v) of the Timoshenko beam model, either as precomputed arrays
# over time-space grids or as callables for on-demand evaluation.
# =============================================================================

# -----------------------------------------------------------------------------
# Function: exact_solution_on_grid
# Description: Evaluate the analytical solution u(x, t) or v(x, t)
#              at a spatial point or over a spatial grid for all/specific times.
# -----------------------------------------------------------------------------
def exact_solution_on_grid(
    func: callable,
    config,
    unif_prt_spc: int = None,
    x_val: float = None,
    k: int = None
) -> np.ndarray | float:
    """
    Evaluate the analytical solution of the Timoshenko beam model on a spatial grid
    or specific point at given time(s).

    Parameters
    ----------
    func : callable
        The analytical solution of the Timoshenko beam model.
        - Must have signature: func(x, t)
        - Should return displacement u(x, t) or rotation v(x, t).
        - Must support vectorized spatial input `x` and scalar temporal input `t`.

    config : object
        Configuration object with the following required attributes:
        - config.ell : float
            Length of the spatial domain.
        - config.t : np.ndarray
            Array of discrete time values.
        - config.n : int
            Maximum index (or total number of steps - 1) for time grid.

    unif_prt_spc : int, optional
        Number of uniform spatial partitions in the interval [0, config.ell].
        Used to generate a grid of spatial points.

    x_val : float, optional
        A specific spatial coordinate in [0, config.ell] at which to evaluate the solution.
        Used for single-point evaluation instead of a grid.

    k : int, optional
        Specific time step index (from config.t) at which to extract the result.
        If None, the full time evolution is returned.

    Returns
    -------
    np.ndarray or float
        - If `k` is None: returns a 2D np.ndarray of shape (len(config.t), len(x)),
          containing values at all time steps for the specified spatial location(s).
        - If `k` is provided: returns a 1D np.ndarray or float with the values
          at the specified time step.
    """

    # -------------------------------------------------------------------------
    # Input Validation
    # -------------------------------------------------------------------------
    # Require at least one spatial argument: either a single point or a full grid
    if x_val is None and unif_prt_spc is None:
        raise ValueError("Specify either `x_val` or `unif_prt_spc`.")

    # -------------------------------------------------------------------------
    # Spatial Grid or Point Setup
    # -------------------------------------------------------------------------
    if x_val is not None:
        # Ensure x_val is within the defined spatial domain
        if not (0 <= x_val <= config.ell):
            raise ValueError(f"x_val = {x_val} is outside domain [0, {config.ell}].")
        # Wrap single point in array for vector compatibility
        x = np.array([x_val])
    else:
        # Generate uniform spatial grid with (unif_prt_spc + 1) points
        x = np.linspace(0, config.ell, unif_prt_spc + 1)

    # -------------------------------------------------------------------------
    # Evaluate Analytical Function Over Time
    # -------------------------------------------------------------------------
    # Compute the solution func(x, t) for each time t_i in the time grid
    values = np.array([func(x, t_i) for t_i in config.t])
    # Resulting shape: (len(config.t), len(x)) for grid,
    # or (len(config.t), 1) for a single point

    # -------------------------------------------------------------------------
    # Extract Specific Time Step if Requested
    # -------------------------------------------------------------------------
    if k is not None:
        # Check that the time index is within valid bounds
        if not (0 <= k <= config.n):
            raise ValueError(f"Time index k = {k} out of range [0, {config.n}].")
        return values[k]  # Return values at a specific time step

    # -------------------------------------------------------------------------
    # Return Full Time Evolution
    # -------------------------------------------------------------------------
    return values  # Return values across all time steps

# -----------------------------------------------------------------------------
# Function: callable_exact_solution
# Description: Return callable(s) or evaluated result(s) for u(x, t) or v(x, t)
#              depending on whether x_vals and/or time index k is provided.
# -----------------------------------------------------------------------------
def callable_exact_solution(
    func: callable,
    config,
    k: int = None,
    x_vals: float | int | list | np.ndarray = None
):
    """
    Return callable(s) or evaluated values of the analytical solution
    of the Timoshenko beam model at specified spatial locations and/or time steps.

    Parameters
    ----------
    func : callable
        Analytical solution function with the signature func(x, t), representing
        displacement u(x, t) or rotation v(x, t).
        - Must support vectorized spatial input `x` and scalar time input `t`.

    config : object
        Configuration object with the following required attributes:
        - config.t : np.ndarray
            1D array of time values used for evaluation.
        - config.n : int
            Maximum index for time steps, usually len(config.t) - 1.

    k : int, optional
        Specific time step index to evaluate at. If None, operates across all time steps.

    x_vals : float | int | list | np.ndarray, optional
        Spatial coordinate(s) where the solution is to be evaluated.
        If None, callables are returned instead of evaluated values.

    Returns
    -------
    callable | list[callable] | float | np.ndarray
        - If only `k` is provided: returns a callable func(x) at fixed t_k.
        - If both `k` and `x_vals` are given: returns the evaluated result at (x_vals, t_k).
        - If only `x_vals` is provided: returns an array of values for each t in config.t.
        - If both `k` and `x_vals` are None: returns list of callables for all time steps.
    """

    # -------------------------------------------------------------------------
    # Helper Function: Standardize and Validate x_vals Input
    # -------------------------------------------------------------------------
    def validate_and_convert_x_vals(x_input):
        """
        Converts x_vals input to a consistent format (float or ndarray).

        Parameters
        ----------
        x_input : float | int | list | np.ndarray | None
            Input value(s) for spatial coordinates.

        Returns
        -------
        float | np.ndarray | None
            Standardized format suitable for vectorized computation.
        """
        if isinstance(x_input, (float, int)):
            return float(x_input)  # Single scalar value
        elif isinstance(x_input, list):
            return np.array(x_input, dtype=float)  # Convert list to NumPy array
        elif isinstance(x_input, np.ndarray):
            return x_input.astype(float)  # Ensure float dtype
        elif x_input is None:
            return None  # No evaluation requested
        else:
            raise TypeError("x_vals must be float, int, list, or np.ndarray.")

    # Convert spatial input to a consistent and safe format
    x_vals = validate_and_convert_x_vals(x_vals)

    # -------------------------------------------------------------------------
    # Helper Function: Construct Callable func(x) for Fixed Time Index
    # -------------------------------------------------------------------------
    def construct_exact_function_at_k(k_idx: int):
        """
        Constructs a callable function that maps x ↦ func(x, t_k) for a fixed t_k.

        Parameters
        ----------
        k_idx : int
            Index into the time grid (config.t).

        Returns
        -------
        callable
            Function taking x and returning func(x, config.t[k_idx]).
        """
        if not (0 <= k_idx <= config.n):
            raise ValueError(f"Time index k = {k_idx} must be in [0, {config.n}].")
        return lambda x: func(x, config.t[k_idx])

    # -------------------------------------------------------------------------
    # Case 1: Fixed Time Step k is Specified
    # -------------------------------------------------------------------------
    if k is not None:
        fn = construct_exact_function_at_k(k)  # Get callable at specific time
        return fn if x_vals is None else fn(x_vals)  # Return function or its evaluated result

    # -------------------------------------------------------------------------
    # Case 2: Operate Over All Time Steps
    # -------------------------------------------------------------------------
    # Create list of callables, one for each time step
    all_functions = [construct_exact_function_at_k(k_idx) for k_idx in range(config.n + 1)]

    # If x_vals is None, return list of callables; else evaluate each at x_vals
    return all_functions if x_vals is None else np.array([fn(x_vals) for fn in all_functions])

# =============================================================
# Kahan–Babuška–Neumaier Summation Algorithm Implementation
# =============================================================
# This module provides a highly accurate floating-point summation
# using the Kahan–Babuška–Neumaier method.
# It supports both Python lists and NumPy ndarrays and helps mitigate
# rounding errors common in naive summation, especially when adding
# numbers with large differences in magnitude.
# =============================================================

# =============================================================
# Function: kahan_babuska_neumaier_sum
# Purpose:  Perform accurate floating-point summation using the
#           Kahan–Babuška–Neumaier algorithm.
# =============================================================
def kahan_babuska_neumaier_sum(numbers):
    """
    ------------------------------------------------------------
    Accurate Summation Using Kahan–Babuška–Neumaier Algorithm
    ------------------------------------------------------------
    Computes a numerically stable sum of floating-point numbers
    by compensating for rounding errors during accumulation.

    Parameters
    ----------
    numbers : list or np.ndarray
        A sequence of floats (1D or multi-D) to be summed.

    Returns
    -------
    float
        The accurately summed result.
    
    Raises
    ------
    TypeError
        If an individual element in the sequence is not a scalar.
    """
    
    # Flatten multidimensional arrays into a 1D view for iteration
    if isinstance(numbers, np.ndarray):
        numbers = numbers.ravel()

    total = 0.0         # Main accumulator for the sum
    compensation = 0.0  # Correction term for lost low-order bits

    # Iterate over each number in the sequence
    for x in numbers:
        # Prevent nested arrays or invalid types from entering the summation
        if isinstance(x, np.ndarray):
            raise TypeError("Each element passed to summation must be a scalar (float or int).")

        # Perform the summation with compensation
        temp = total + x

        # Error compensation logic based on which term is larger
        if abs(total) >= abs(x):
            compensation += (total - temp) + x
        else:
            compensation += (x - temp) + total

        total = temp  # Update total with the temporary result

    # Return the final corrected sum
    return total + compensation

# =============================================================================
# FUNCTION: compute_L2_norm_galerkin_approx
# Purpose : Compute the L² norm of a Galerkin approximation ũₖ(x)
#           over the spatial domain [0, ell] using adaptive quadrature.
# Method  : Uses adaptive Gauss–Legendre quadrature
# Dependencies:
#   - config.ell: Domain length from configuration object
#   - adaptive_gauss_legendre_integrator: Custom quadrature routine
# =============================================================================

def compute_L2_norm_galerkin_approx(
    func,
    config,
    k=None,
    **quad_kwargs
):
    """
    Computes the L² norm of Galerkin-approximated solution(s) over the interval [0, ell]:

        L2_norm = sqrt( ∫₀^ell [uₖ(x)]² dx )

    Supports both single function input or a sequence of functions over time.

    Parameters
    ----------
    func : callable or list of callables
        The Galerkin-approximated function(s), typically u(x, tₖ) or v(x, tₖ).
        - Single callable: compute norm for that function.
        - List of callables: compute for all or selected time index `k`.

    config : object
        Configuration object containing:
        - config.ell : float
            Upper bound of the spatial domain interval [0, ell].

    k : int, optional
        Time index to evaluate. If None, computes norms for all functions in `func`.

    **quad_kwargs : dict, optional
        Keyword arguments passed to `adaptive_gauss_legendre_integrator`.

        Recognized keys (with default values):
        - tol       : float = 1e-6     → Absolute tolerance for convergence
        - min_dx    : float = 1/128    → Minimum interval width for subdivision
        - n_gauss   : int = 5          → Initial Gauss–Legendre nodes
        - max_gauss : int = 50         → Max Gauss–Legendre nodes during refinement

    Returns
    -------
    float or list of float
        - Single float if `k` is provided (norm at time index `k`).
        - List of floats if `k` is None (norms over all time steps).

    Raises
    ------
    ValueError
        If `func` is not a callable or list of callables, or if `k` is invalid.
    """

    # -------------------------------------------------------------------------
    # STEP 1: Retrieve domain length from configuration
    # -------------------------------------------------------------------------
    ell = config.ell  # Upper bound of the spatial domain

    # -------------------------------------------------------------------------
    # STEP 2: Normalize `func` into a list of callables
    # -------------------------------------------------------------------------
    if callable(func):
        func_list = [func]  # Wrap single function in list for uniform handling
    elif isinstance(func, list) and all(callable(f) for f in func):
        func_list = func  # Use list as-is
    else:
        raise ValueError("`func` must be a callable or a list of callables.")

    # -------------------------------------------------------------------------
    # STEP 3: Define helper to compute L² norm for one function
    # -------------------------------------------------------------------------
    def compute_single_l2_norm(i):
        """
        Compute the L² norm of the i-th Galerkin-approximated function.

        Parameters
        ----------
        i : int
            Index of the function in func_list.

        Returns
        -------
        float
            The computed L² norm: sqrt( ∫₀^ell [uₖ(x)]² dx )
        """
        approx_fn = func_list[i]

        # Define the integrand function: square of the Galerkin approximation
        def integrand(x):
            return approx_fn(x) ** 2

        # Call adaptive Gauss–Legendre integrator over [0, ell]
        integral, _, *_ = adaptive_gauss_legendre_integrator(
            integrand,
            ell,
            **quad_kwargs
        )

        return np.sqrt(integral)

    # -------------------------------------------------------------------------
    # STEP 4: Evaluate for a specific time step if `k` is provided
    # -------------------------------------------------------------------------
    if k is not None:
        if not isinstance(k, int):
            raise ValueError("Parameter `k` must be an integer.")
        if not (0 <= k < len(func_list)):
            raise ValueError(f"Invalid time index `k={k}`. Valid range: 0 to {len(func_list) - 1}.")
        return compute_single_l2_norm(k)

    # -------------------------------------------------------------------------
    # STEP 5: Evaluate for all time steps (if `k` is None)
    # -------------------------------------------------------------------------
    return [compute_single_l2_norm(i) for i in range(len(func_list))]

# =============================================================================
# FUNCTION: compute_L2_difference_norms
# -----------------------------------------------------------------------------
# Purpose :
#     Compute L² norms of differences between two Galerkin approximations
#     over the spatial domain [0, ell] at each time step.
#
# Use Case :
#     Typically used in convergence studies by comparing solutions obtained
#     with different numbers of basis functions (e.g., N and N+1).
#
# Method :
#     - Extract modal solutions from each solver instance using
#       `callable_compute_ansatz()`.
#     - Construct the difference functions f_diff(x) = f_next(x) - f_init(x).
#     - Compute the L² norm of each difference function using
#       `compute_L2_norm_galerkin_approx`.
#
# Dependencies:
#     - compute_L2_norm_galerkin_approx
#     - callable_compute_ansatz() from each solver
# =============================================================================

def compute_L2_difference_norms(
    solver_init,
    solver_next,
    solution_type: str,
    **quad_kwargs
):
    """
    Compute the L² norm of the difference between two Galerkin-approximated
    solutions at each time step over the spatial domain [0, ell].

    Parameters
    ----------
    solver_init : TimoshenkoModelSolver
        Solver instance representing the initial (coarser) approximation.
        Must implement `callable_compute_ansatz()` and contain `.ell`.

    solver_next : TimoshenkoModelSolver
        Solver instance representing the refined (finer) approximation.
        Must implement `callable_compute_ansatz()`.

    solution_type : str
        Field to compare. Must be one of:
            - 'u' : displacement
            - 'v' : rotation

    **quad_kwargs : dict, optional
        Additional keyword arguments passed to `compute_L2_norm_galerkin_approx`.
        Examples:
            tol       : float = 1e-6   → Integration tolerance
            min_dx    : float = 1/128  → Minimum subinterval width
            n_gauss   : int = 5        → Initial Gauss–Legendre points
            max_gauss : int = 50       → Max refinement points

    Returns
    -------
    list[float]
        L² norms of the difference between the two solutions at each time step:
        ‖u_next(x) − u_init(x)‖_L2 or ‖v_next(x) − v_init(x)‖_L2

    Raises
    ------
    ValueError
        If `solution_type` is not in {'u', 'v'}, or if solvers differ in time step count.
    """

    # -------------------------------------------------------------------------
    # STEP 1: Validate input solution type
    # -------------------------------------------------------------------------
    if solution_type not in {"u", "v"}:
        raise ValueError(
            f"Invalid solution_type '{solution_type}'. Must be 'u' or 'v'."
        )

    # -------------------------------------------------------------------------
    # STEP 2: Extract Galerkin-mode functions from both solvers
    #         Each returns a list of functions: [f₀(x), f₁(x), ..., fₙ(x)]
    # -------------------------------------------------------------------------
    funcs_init = solver_init.callable_compute_ansatz(solution_type=solution_type)
    funcs_next = solver_next.callable_compute_ansatz(solution_type=solution_type)

    # -------------------------------------------------------------------------
    # STEP 3: Ensure matching number of time steps between solvers
    # -------------------------------------------------------------------------
    if len(funcs_init) != len(funcs_next):
        raise ValueError(
            "Mismatch in number of time steps: "
            f"{len(funcs_init)} (init) vs {len(funcs_next)} (next)."
        )

    # -------------------------------------------------------------------------
    # STEP 4: Build a list of difference functions at each time step
    #         Using closure-safe lambda to ensure proper binding
    # -------------------------------------------------------------------------
    diff_funcs = [
        (lambda f1=f1, f2=f2: lambda x: f2(x) - f1(x))()
        for f1, f2 in zip(funcs_init, funcs_next)
    ]

    # -------------------------------------------------------------------------
    # STEP 5: Compute the L² norm of each difference using adaptive quadrature
    # -------------------------------------------------------------------------
    return compute_L2_norm_galerkin_approx(
        func=diff_funcs,
        config=solver_init,  # Assumes solver_init has `.ell` used in integration
        **quad_kwargs         # Pass through integration settings
    )

# =============================================================================
# FUNCTION: compute_L2_norm_from_galerkin_coeffs
# -----------------------------------------------------------------------------
# Purpose:
#     Computes the L² norm(s) of a Galerkin-approximated solution from its
#     coefficient matrix using the mass matrix from the Legendre polynomial basis:
#
#         L2 = (ell / 2) * sqrt(cᵀ * H * c)
#
# Assumptions:
#     - Time grid: t = np.linspace(0, T, n + 1)
#     - Coefficients exclude initial conditions (u₀, u₁): shape is (n−1, N)
#     - Time layers in coeff begin from k = 2 (i.e., coeff[0] ↔ k = 2)
#
# Dependencies:
#     - config.ell : float → spatial domain length
#     - galerkin_stencils(N, v, operator="identity") : applies mass matrix H
# =============================================================================

def compute_L2_norm_from_galerkin_coeffs(
    coeff: np.ndarray,
    config,
    time_layer: int = None
) -> float | list[float]:
    """
    Compute the L² norm(s) of a Galerkin-approximated solution using a matrix formulation.

    The L² norm at time step k is given by:
        L2 = (ell / 2) * sqrt(c_kᵀ * H * c_k)

    Parameters
    ----------
    coeff : np.ndarray
        2D array of Galerkin coefficients with shape (n - 1, N),
        where each row corresponds to time layer k in [2, n].

    config : object
        Configuration object with attribute:
        - config.ell : float
            Length of the spatial domain.

    time_layer : int, optional
        Time step `k` (must satisfy k ≥ 2). If None, norms for all time layers are returned.

    Returns
    -------
    float or list[float]
        - A single float if `time_layer` is specified
        - A list of floats if norms are computed for all time layers

    Raises
    ------
    ValueError
        If `coeff` is not a 2D NumPy array.

    IndexError
        If `time_layer` is invalid or out of bounds.
    """

    # -------------------------------------------------------------------------
    # STEP 1: Extract domain length (ell) from config object
    # -------------------------------------------------------------------------
    ell = config.ell  # Spatial domain [0, ell]

    # -------------------------------------------------------------------------
    # STEP 2: Validate coefficient matrix shape (must be 2D)
    # -------------------------------------------------------------------------
    if coeff.ndim != 2:
        raise ValueError("Input `coeff` must be a 2D NumPy array of shape (n-1, N).")

    num_time_layers, N = coeff.shape  # n-1 time layers, N basis coefficients

    # =========================================================================
    # INTERNAL FUNCTION: compute_l2_at
    # -------------------------------------------------------------------------
    # Computes the L² norm at a specific time index using:
    #     L2_k = (ell / 2) * sqrt(c_kᵀ * H * c_k)
    #
    # Parameters
    # ----------
    # k_idx : int
    #     Zero-based index into coeff (coeff[k_idx] ↔ time layer k = k_idx + 2)
    #
    # Returns
    # -------
    # float : L² norm for that time layer
    # =========================================================================
    def compute_l2_at(k_idx: int) -> float:
        c_k = coeff[k_idx, :]  # Extract coefficient vector for time layer k
        H_c = galerkin_stencils(N=N, v=c_k, operator="identity")  # Apply mass matrix
        inner_product = np.dot(c_k, H_c)  # Compute cᵀ * H * c
        return (ell / 2.0) * np.sqrt(inner_product)  # Scale result and return

    # -------------------------------------------------------------------------
    # STEP 3: If a specific time_layer is requested, validate and compute only that
    # -------------------------------------------------------------------------
    if time_layer is not None:
        if not isinstance(time_layer, int) or time_layer < 2:
            raise IndexError(
                f"Invalid `time_layer = {time_layer}`. Must be an integer ≥ 2 "
                "(initial conditions are at k = 0 and k = 1)."
            )

        k_idx = time_layer - 2  # Convert time layer k to zero-based index

        if k_idx >= num_time_layers:
            raise IndexError(
                f"`time_layer = {time_layer}` exceeds available data. "
                f"Valid range: 2 ≤ k ≤ {num_time_layers + 1} (matrix shape = {coeff.shape})."
            )

        return compute_l2_at(k_idx)

    # -------------------------------------------------------------------------
    # STEP 4: Otherwise compute norms for all time layers (k = 2 to n)
    # -------------------------------------------------------------------------
    return [compute_l2_at(k_idx) for k_idx in range(num_time_layers)]


# =============================================================================
# FUNCTION: compute_L2_difference_norms_from_coeffs
# -----------------------------------------------------------------------------
# Purpose:
#     Computes the L² norm of the difference between two Galerkin-approximated
#     solutions at each time step using their coefficient matrices.
#
#     Formula:
#         L2_diff_k = (ell / 2) * sqrt((c_next - c_init)ᵀ H (c_next - c_init))
#
# Notes:
#     - Returns 0.0 for time layers k = 0 or 1 (analytic initial conditions).
#     - Automatically zero-pads matrices to handle spatial resolution mismatch.
#
# Requirements:
#     - config.ell : float → spatial domain length
#     - compute_L2_norm_from_galerkin_coeffs() must be available
# =============================================================================

def compute_L2_difference_norms_from_coeffs(
    coeff_init: np.ndarray,
    coeff_next: np.ndarray,
    config,
    time_layer: int = None
) -> float | list[float]:
    """
    Compute L² norm(s) of the difference between two Galerkin
    approximations at specified time layers.

    Handles:
    - Zero-padding for mismatched spatial resolution (N₁ ≠ N₂)
    - Analytic returns for k = 0, 1 (initial conditions)
    - Delegates numerical computation to L2 norm engine for k ≥ 2

    Parameters
    ----------
    coeff_init : np.ndarray
        Coefficient matrix (n−1, N₁) for initial approximation (prior timestep).

    coeff_next : np.ndarray
        Coefficient matrix (n−1, N₂) for next approximation (current timestep).

    config : object
        Configuration object with attribute:
        - config.ell : float
            Length of the spatial domain.

    time_layer : int, optional
        Time step index `k` to evaluate.
        - If None: computes norms for all layers (k = 0 to n)
        - If 0 or 1: returns np.float64(0.0)
        - If ≥ 2: returns scalar norm for specified k

    Returns
    -------
    float or list[float]
        - Single float if `time_layer` is provided.
        - List of floats from k = 0 to n if `time_layer` is None.

    Raises
    ------
    ValueError
        If `coeff_init` and `coeff_next` differ in number of time layers.
    """

    # =========================================================================
    # STEP 1: Return analytic zero for initial conditions at k = 0 or 1
    # =========================================================================
    if time_layer in {0, 1}:
        return np.float64(0.0)  # These layers are known analytically, not computed

    # =========================================================================
    # STEP 2: Sanity check — ensure consistent number of time layers
    # =========================================================================
    if coeff_init.shape[0] != coeff_next.shape[0]:
        raise ValueError(
            f"Incompatible number of time layers: "
            f"{coeff_init.shape[0]} (init) vs {coeff_next.shape[0]} (next)."
        )

    # =========================================================================
    # INTERNAL FUNCTION: pad_matrix
    # -----------------------------------------------------------------------------
    # Pads a coefficient matrix to match a target number of basis functions (columns).
    # =========================================================================
    def pad_matrix(matrix: np.ndarray, target_cols: int) -> np.ndarray:
        """
        Pads a coefficient matrix with zeros to the specified column size.

        Parameters
        ----------
        matrix : np.ndarray
            Coefficient matrix of shape (n-1, N).

        target_cols : int
            Desired number of columns.

        Returns
        -------
        np.ndarray
            Zero-padded matrix of shape (n-1, target_cols).
        """
        pad_width = target_cols - matrix.shape[1]
        if pad_width == 0:
            return matrix  # Already correct size
        return np.pad(
            matrix,
            pad_width=((0, 0), (0, pad_width)),  # Only pad along spatial dimension
            mode="constant",
            constant_values=np.float64(0.0)
        )

    # =========================================================================
    # STEP 3: Equalize spatial resolution (basis size) by padding both matrices
    # =========================================================================
    N_init = coeff_init.shape[1]
    N_next = coeff_next.shape[1]
    max_N = max(N_init, N_next)  # Target number of basis functions

    coeff_init_padded = pad_matrix(coeff_init, max_N)
    coeff_next_padded = pad_matrix(coeff_next, max_N)

    # =========================================================================
    # STEP 4: Compute coefficient difference ΔC = C_next − C_init
    # =========================================================================
    coeff_diff = coeff_next_padded - coeff_init_padded

    # =========================================================================
    # STEP 5: Compute L² norm at a specific time layer (if requested)
    # =========================================================================
    if time_layer is not None:
        return compute_L2_norm_from_galerkin_coeffs(
            coeff=coeff_diff,
            config=config,
            time_layer=time_layer
        )

    # =========================================================================
    # STEP 6: Compute L² norms for all time layers
    # - k = 0, 1 → return 0.0 (analytic)
    # - k ≥ 2   → compute numerically from coefficient difference
    # =========================================================================
    norms_k2_to_n = compute_L2_norm_from_galerkin_coeffs(
        coeff=coeff_diff,
        config=config
    )
    return [np.float64(0.0), np.float64(0.0)] + norms_k2_to_n


# ===============================================================
# Function: compute_L2_error
# Purpose : Compute the L² error between exact and Galerkin-approximated
#           solutions over a spatial domain [0, ell].
# Method  : Uses adaptive Gauss–Legendre quadrature
# ===============================================================

def compute_L2_error(
    exact_solution_generator,
    approx_solution_generator,
    ell,
    k=None,
    **quad_kwargs
):
    """
    Computes the L² error between exact and Galerkin-approximated solutions
    over the spatial domain [0, ell] at a specific time step or across all steps.

    L² error is defined as:
        L2_error = sqrt( ∫₀^ell [u_exact(x) - u_approx(x)]² dx )

    Parameters
    ----------
    exact_solution_generator : callable or list of callables
        - The exact solution(s) u(x, tₖ).
        - Can be a single callable or a list of time-dependent callables.

    approx_solution_generator : callable or list of callables
        - The Galerkin approximation(s) ũₖ(x).
        - Must correspond in size and order to the exact solutions.

    ell : float
        The upper bound of the spatial domain [0, ell].

    k : int, optional
        Time index for which to compute the error.
        If None, the function computes L² error across all time steps.

    **quad_kwargs : dict, optional
        Additional parameters passed to `adaptive_gauss_legendre_integrator`.

        Supported keyword arguments:
        - tol       : float (default = 1e-6)
              → Integration convergence tolerance.
        - min_dx    : float (default = 1 / 128)
              → Minimum subinterval width in the adaptive scheme.
        - n_gauss   : int (default = 5)
              → Initial number of Gauss–Legendre nodes per subinterval.
        - max_gauss : int (default = 50)
              → Maximum allowable Gauss–Legendre nodes for refinement.

    Returns
    -------
    float or list of float
        - If `k` is specified: returns the L² error at that time step.
        - If `k` is None: returns a list of L² errors for all time steps.

    Raises
    ------
    ValueError
        If the input types are not callable or lists of callables,
        or if the lengths of exact and approximated lists differ,
        or if the time index `k` is out of range.
    """

    # ----------------------------------------------------------
    # Step 1: Normalize input to lists of callables
    # ----------------------------------------------------------
    if callable(exact_solution_generator):
        exact_solution_generator = [exact_solution_generator]

    if callable(approx_solution_generator):
        approx_solution_generator = [approx_solution_generator]

    if not (isinstance(exact_solution_generator, list) and
            isinstance(approx_solution_generator, list) and
            all(callable(f) for f in exact_solution_generator) and
            all(callable(f) for f in approx_solution_generator)):
        raise ValueError("Inputs must be callable functions or lists of callables.")

    if len(exact_solution_generator) != len(approx_solution_generator):
        raise ValueError("Exact and approximate function lists must be the same length.")

    # ----------------------------------------------------------
    # Step 2: Define internal function for a specific time index
    # ----------------------------------------------------------
    def compute_error_at_k(k_idx):
        """
        Compute L² error for time step `k_idx` by integrating the
        pointwise squared error over [0, ell].

        Parameters
        ----------
        k_idx : int
            Index of the solution pair to evaluate.

        Returns
        -------
        float
            L² error at time step `k_idx`.
        """
        exact_fn = exact_solution_generator[k_idx]
        approx_fn = approx_solution_generator[k_idx]

        # Define squared error function
        def squared_error(x):
            return (exact_fn(x) - approx_fn(x)) ** 2

        # Perform adaptive Gauss–Legendre quadrature
        integral, _, *_ = adaptive_gauss_legendre_integrator(
            squared_error, ell, **quad_kwargs
        )

        return np.sqrt(integral)

    # ----------------------------------------------------------
    # Step 3: Compute and return error(s)
    # ----------------------------------------------------------
    if k is not None:
        if not isinstance(k, int):
            raise ValueError("`k` must be an integer index.")
        if not (0 <= k < len(exact_solution_generator)):
            raise ValueError(f"Invalid index k={k}. Valid range: 0 to {len(exact_solution_generator) - 1}.")
        return compute_error_at_k(k)

    # Compute for all time steps
    return [compute_error_at_k(i) for i in range(len(exact_solution_generator))]

# =============================================================================
# FUNCTION: named
# -----------------------------------------------------------------------------
# Purpose:
#   Dynamically assign a `.name` attribute to any Python object, enabling
#   identification and traceability (e.g., for logging, plotting, or filenames).
#
# Usage Example:
#   model = named("test1", TimoshenkoModelSolver(...))
#   print(model.name)  # Output: "test1"
#
# Inputs:
#   - name : str     — the name to assign to the object's `.name` attribute
#   - obj  : object  — any class instance to which the name will be attached
#
# Output:
#   - The same object with an added `.name` attribute
# =============================================================================

def named(name: str, obj: object) -> object:
    """
    Assigns a `.name` attribute to the given object for labeling purposes.

    Parameters:
        name (str): The string label to assign to the object.
        obj (object): The instance to be labeled.

    Returns:
        object: The same instance, now with a `.name` attribute set to `name`.
    """
    # -----------------------------------------------------------
    # Attach the name attribute dynamically to the object
    # This allows any object to be later identified or tracked
    # without modifying its original class definition.
    # -----------------------------------------------------------
    setattr(obj, "name", name)

    # -----------------------------------------------------------
    # Return the same object, now with the .name attribute added.
    # Useful in method chaining or inline object instantiation.
    # -----------------------------------------------------------
    return obj

# =============================================================================
# Function: plot_L2_errors_over_time
# -----------------------------------------------------------------------------
# Purpose:
#   Generate and save two LaTeX-styled plots of L2 errors over time for the
#   displacement (u) and rotation (v) fields in the Timoshenko beam model.
#   Also saves formatted CSV logs for errors and condition numbers.
# -----------------------------------------------------------------------------
# Inputs:
#   - time_array : 1D array of time points
#   - error_u    : L2 error array for displacement u
#   - error_v    : L2 error array for rotation v
#   - config     : object with attributes:
#                   config.n       : number of time steps
#                   config.N       : number of Galerkin modes
#                   config.cond_u  : list of condition numbers for system u
#                   config.cond_v  : list of condition numbers for system v
#   - output_dir : path to directory for PDF/CSV outputs (default="plots")
# -----------------------------------------------------------------------------
# Outputs:
#   - (pdf_u_path, pdf_v_path) : paths to the saved PDF plots for u and v
# =============================================================================

def plot_L2_errors_over_time(
    time_array,
    error_u,
    error_v,
    config,
    output_dir: str = "plots"
) -> tuple[str, str]:
    """
    Title: Plot and Log L2 Error Evolution for Timoshenko Beam Model

    Description:
        Generates two LaTeX-rendered plots showing the time evolution of L2 
        errors for displacement (u) and rotation (v). Also logs these errors 
        and corresponding condition numbers to CSV files.

    Parameters:
        time_array (array-like): Time samples for each timestep.
        error_u (array-like): L2 error at each timestep for displacement u.
        error_v (array-like): L2 error at each timestep for rotation v.
        config (object): Contains solver parameters (n, N, cond_u, cond_v).
        output_dir (str): Path to the output directory (default: 'plots').

    Returns:
        tuple[str, str]: Full paths to the generated PDF plots (u, v).
    """

    # =========================================================================
    # MODULE IMPORTS (local to avoid unnecessary global dependencies)
    # =========================================================================
    from pathlib import Path             # File and directory path management
    from datetime import datetime        # Used for timestamping output files
    import matplotlib.pyplot as plt      # Visualization and plotting
    from matplotlib import rcParams      # Fine-grained plot styling
    
    # =========================================================================
    # CONFIGURE MATPLOTLIB FOR LaTeX-STYLED RENDERING
    # =========================================================================
    rcParams["text.usetex"] = True
    rcParams["font.family"] = "lmodern"  # Use Latin Modern math-friendly font
    rcParams["text.latex.preamble"] = r"""
    \usepackage[utf8]{inputenc}
    \usepackage[T1]{fontenc}
    \usepackage{lmodern}
    \usepackage{slantsc}
    \usepackage{dsfont}
    \usepackage{upgreek}
    \usepackage{amsmath,amssymb,amsthm,amstext,amsfonts}
    \usepackage{mathtools}
    \usepackage{nicefrac}
    \usepackage{xcolor}
    """

    # =========================================================================
    # STYLING CONSTANTS FOR PLOTTING
    # =========================================================================
    LINE_WIDTH = 2.0
    color_u = "#0072B2"  # Blue: for u displacement error (Okabe–Ito palette)
    color_v = "#E69F00"  # Orange: for v rotation error (Okabe–Ito palette)

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================
    if not (len(time_array) and len(error_u) and len(error_v)):
        raise ValueError("Inputs 'time_array', 'error_u', and 'error_v' must be non-empty.")
    if not (len(time_array) == len(error_u) == len(error_v)):
        raise ValueError("Input arrays must be of equal length.")

    # =========================================================================
    # OUTPUT DIRECTORY PREPARATION
    # =========================================================================
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)  # Create output directory

    # Determine range of time and generate unique timestamp for filenames
    t_min, t_max = float(time_array[0]), float(time_array[-1])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = getattr(config, "name", "config")  # Default name if not defined

    # =========================================================================
    # PLOT: L2 ERROR FOR DISPLACEMENT u(x, t)
    # =========================================================================
    plt.figure(figsize=(8, 4))
    plt.plot(
        time_array, error_u,
        marker='o', linestyle='-', linewidth=LINE_WIDTH,
        color=color_u,
        label=r"$E_{1,k} = \left\| u(\cdot, t_k) - \tilde{u}_{k,N}(\cdot) \right\|$"
    )
    plt.xlabel(rf"Time $t \in \left[ {t_min:g}, {t_max:g} \right]$")
    plt.ylabel(r"$E_{1, k}$")
    plt.title(r"$L^2$ Error Evolution for $u(x, t)$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    pdf_u = output_path / f"{config_name}_L2_error_u_n{config.n}_N{config.N}_{timestamp}.pdf"
    plt.savefig(pdf_u)
    plt.close()

    # =========================================================================
    # PLOT: L2 ERROR FOR ROTATION v(x, t)
    # =========================================================================
    plt.figure(figsize=(8, 4))
    plt.plot(
        time_array, error_v,
        marker='s', linestyle='--', linewidth=LINE_WIDTH,
        color=color_v,
        label=r"$E_{2,k} = \left\| v(\cdot, t_k) - \tilde{v}_{k,N}(\cdot) \right\|$"
    )
    plt.xlabel(rf"Time $t \in \left[ {t_min:g}, {t_max:g} \right]$")
    plt.ylabel(r"$E_{2, k}$")
    plt.title(r"$L^2$ Error Evolution for $v(x, t)$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    pdf_v = output_path / f"{config_name}_L2_error_v_n{config.n}_N{config.N}_{timestamp}.pdf"
    plt.savefig(pdf_v)
    plt.close()

    # =========================================================================
    # CSV EXPORT: LOGGING ERROR VALUES FOR u AND v
    # =========================================================================
    csv_u = output_path / f"{config_name}_L2_error_u_n{config.n}_N{config.N}_{timestamp}.csv"
    with csv_u.open("w") as f_u:
        for k, err in enumerate(error_u):
            f_u.write(f"Time step {k:3d}: L2 error for solution 'u' = {err:.6e}\n")

    csv_v = output_path / f"{config_name}_L2_error_v_n{config.n}_N{config.N}_{timestamp}.csv"
    with csv_v.open("w") as f_v:
        for k, err in enumerate(error_v):
            f_v.write(f"Time step {k:3d}: L2 error for solution 'v' = {err:.6e}\n")

    # =========================================================================
    # CSV EXPORT: LOGGING CONDITION NUMBERS FOR u AND v (OPTIONAL DIAGNOSTICS)
    # =========================================================================
    cond_csv_u = output_path / f"{config_name}_cond_numb_u_n{config.n}_N{config.N}_{timestamp}.csv"
    with cond_csv_u.open("w") as f_cu:
        for k, val in enumerate(config.cond_u):
            f_cu.write(f"Time step {k:3d}: condition number for 'u' = {val:.6e}\n")

    cond_csv_v = output_path / f"{config_name}_cond_numb_v_n{config.n}_N{config.N}_{timestamp}.csv"
    with cond_csv_v.open("w") as f_cv:
        for k, val in enumerate(config.cond_v):
            f_cv.write(f"Time step {k:3d}: condition number for 'v' = {val:.6e}\n")

    # =========================================================================
    # RETURN: PATHS TO THE GENERATED PDF FILES
    # =========================================================================
    return str(pdf_u), str(pdf_v)

# =============================================================================
# FUNCTION: plot_exact_vs_approx_solution_at_time_k
# -----------------------------------------------------------------------------
# Purpose:
#   Generate a side-by-side comparison plot between the analytical (exact) and
#   Galerkin (approximate) solution for either displacement (u) or rotation (v)
#   at a fixed time index k. The output is a LaTeX-styled, high-resolution PDF
#   using a colorblind-friendly palette (Okabe-Ito).
#
# Inputs:
#   - exact_soln    : Callable function that returns exact solution, takes (x, t)
#   - approx_solver : Object that exposes a method callable_compute_ansatz()
#   - solution_type : 'u' for displacement or 'v' for rotation
#   - time_layer    : Time index k (must be 0 ≤ k ≤ config.n)
#   - config        : Configuration object with attributes: ell, t, N, n
#   - output_dir    : Output directory for saving the PDF plot (default: 'plots')
#
# Output:
#   - str : Full file path to the saved PDF plot
# =============================================================================

from pathlib import Path                # Used for portable path handling
from datetime import datetime           # To timestamp plot filenames
import numpy as np                      # For spatial grid generation and array ops
import matplotlib.pyplot as plt         # Plotting interface
from matplotlib import rcParams         # Configuration for LaTeX-rendered text

def plot_exact_vs_approx_solution_at_time_k(
    exact_soln: callable,
    approx_solver: object,
    solution_type: str,
    time_layer: int,
    config,
    output_dir: str = "plots"
) -> str:
    """
    Title: Plot Exact vs Approximate Solution at a Specific Time Layer

    Description:
        This function compares the exact analytical solution to a Galerkin-based
        approximation of a field variable (u or v) at a specific time layer `k`.
        It generates a side-by-side plot of both solutions and saves it as a 
        LaTeX-styled, publication-quality PDF in the specified output directory.

    Parameters:
        exact_soln (callable): Function that evaluates the exact solution at (x, t).
        approx_solver (object): Must provide callable_compute_ansatz(solution_type, k, x_vals).
        solution_type (str): Field to plot; 'u' (displacement) or 'v' (rotation).
        time_layer (int): Index of the time layer to visualize (0 ≤ time_layer ≤ config.n).
        config (object): Object containing attributes: ell (float), t (List[float]), N (int), n (int).
        output_dir (str): Directory to save the output plot (default is "plots").

    Returns:
        str: Full path to the saved PDF plot.
    """

    # =========================================================================
    # VALIDATE TIME LAYER INDEX
    # =========================================================================
    if not (0 <= time_layer <= config.n):
        raise ValueError(f"time_layer must be in the range [0, {config.n}]")

    # =========================================================================
    # DEFINE PLOT CONSTANTS
    # =========================================================================
    LINE_WIDTH = 3.0  # Line thickness for both curves
    num_points = 201  # Number of spatial sample points for plot resolution

    # =========================================================================
    # GENERATE SPATIAL GRID AND EXTRACT CURRENT TIME
    # =========================================================================
    x_vals = np.linspace(0, config.ell, num_points)  # Spatial domain [0, ell]
    t_k = config.t[time_layer]                       # Time at index k

    # =========================================================================
    # EVALUATE EXACT AND APPROXIMATE SOLUTIONS AT t_k
    # =========================================================================
    exact_values = exact_soln(x_vals, t_k)           # Exact u(x, t_k) or v(x, t_k)
    approx_values = approx_solver.callable_compute_ansatz(
        solution_type=solution_type,                 # Field: 'u' or 'v'
        k=time_layer,                                # Time step index
        x_vals=x_vals                                # Evaluation grid
    )

    # =========================================================================
    # DEFINE COLORBLIND-FRIENDLY COLORS (Okabe-Ito Palette)
    # =========================================================================
    color_exact = "#009E73"   # Green: Exact solution
    color_approx = "#D55E00"  # Orange: Approximate (Galerkin) solution

    # =========================================================================
    # CONFIGURE MATPLOTLIB TO USE LaTeX FOR RENDERING TEXT
    # =========================================================================
    rcParams["text.usetex"] = True
    rcParams["font.family"] = "lmodern"  # Use a LaTeX-compatible font
    rcParams["text.latex.preamble"] = r"""
    \usepackage[utf8]{inputenc}
    \usepackage[T1]{fontenc}
    \usepackage{lmodern}
    \usepackage{slantsc}
    \usepackage{dsfont}
    \usepackage{upgreek}
    \usepackage{amsmath,amssymb,amsthm,amstext,amsfonts}
    \usepackage{mathtools}
    \usepackage{nicefrac}
    \usepackage{xcolor}
    """

    # =========================================================================
    # ENSURE OUTPUT DIRECTORY EXISTS
    # =========================================================================
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)  # Create directory if not present

    # =========================================================================
    # CREATE PLOT: EXACT VS APPROXIMATE SOLUTION
    # =========================================================================
    plt.figure(figsize=(8, 4))  # Define figure size in inches (W x H)

    # --- Plot exact solution ---
    plt.plot(
        x_vals,
        exact_values,
        label=rf"Exact: ${solution_type}(x, {t_k:g})$",
        color=color_exact,
        linestyle='-',
        linewidth=LINE_WIDTH
    )

    # --- Plot approximate solution ---
    plt.plot(
        x_vals,
        approx_values,
        label=rf"Approximate: $\tilde{{{solution_type}}}_{{k,N}}(x)$",
        color=color_approx,
        linestyle='--',
        linewidth=LINE_WIDTH
    )

    # =========================================================================
    # CUSTOMIZE AXES, TITLE, LEGEND, AND GRID
    # =========================================================================
    plt.xlabel(rf"Spatial coordinate $x \in \left[0, {config.ell:g} \right]$")
    plt.ylabel("Solution value")
    plt.title(rf"Exact vs Approximate Solution: ${solution_type}(x, t_{{{time_layer}}})$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()  # Auto-adjust layout to avoid text clipping

    # =========================================================================
    # GENERATE TIMESTAMPED FILE NAME AND SAVE FIGURE
    # =========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. "20250720_132100"
    config_name = getattr(config, "name", "config")  # Use config.name if available
    filename = output_path / f"{config_name}_solution_{solution_type}_t{time_layer}_N{config.N}_{timestamp}.pdf"

    plt.savefig(filename)  # Save to PDF format for high-quality output
    plt.close()            # Close figure to release memory

    # =========================================================================
    # RETURN FULL FILE PATH AS STRING
    # =========================================================================
    return str(filename)