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
# LEGENDRE POLYNOMIAL UTILITIES
# --------------------------------------------------------------------------- #
"""
Functions related to Legendre polynomials, including their
differences as employed in spectral basis constructions.
"""

# --------------------------------------------------------------------------- #
# Function: shifted_legendre
# Purpose: Evaluate the shifted Legendre polynomial P_m(x) over [0, ell]
# --------------------------------------------------------------------------- #
def shifted_legendre(m: int, ell: float, x: float | np.ndarray) -> np.float64 | np.ndarray:
    """
    Compute the shifted Legendre polynomial P_m(x) over the interval [0, ell].

    The function maps input values from [0, ell] to the standard Legendre domain [-1, 1],
    evaluates the polynomial of degree m using NumPy's Legendre implementation,
    and returns the result with float64 precision.

    Parameters
    ----------
    m : int
        Degree of the Legendre polynomial (non-negative integer).
    
    ell : float
        Upper bound of the evaluation interval [0, ell]; must be strictly positive.
    
    x : float or np.ndarray
        Scalar or array of input values. Values outside [0, ell] are clipped to this range.

    Returns
    -------
    np.float64 or np.ndarray
        Evaluated polynomial values:
        - Returns a scalar np.float64 if input x is a scalar.
        - Returns a NumPy array of dtype float64 if x is an array.
    """

    # ---------------------------- Input Validation ---------------------------- #
    # Convert x to a NumPy array for uniform handling and clip to [0, ell]
    x = np.clip(np.asarray(x), 0.0, ell)

    # -------------------------- Domain Transformation -------------------------- #
    # Map x from [0, ell] → [-1, 1] to match the standard Legendre domain
    x_mapped = 2.0 * x / ell - 1.0

    # ------------------------ Polynomial Computation --------------------------- #
    # Generate and evaluate the m-th Legendre polynomial at the mapped point(s)
    result = legendre(m)(x_mapped)

    # ----------------------------- Type Handling ------------------------------- #
    # Ensure consistent return type: np.float64 for scalar input, array otherwise
    result = np.array(result, dtype=np.float64)
    return result.item() if np.isscalar(x) else result

# --------------------------------------------------------------------------- #
# Function: normalized_shifted_legendre
# Purpose: Evaluate the normalized shifted Legendre polynomial P_m^*(x) over [0, ell]
# --------------------------------------------------------------------------- #
def normalized_shifted_legendre(m: int, ell: float, x: float | np.ndarray) -> np.float64 | np.ndarray:
    """
    Compute the normalized shifted Legendre polynomial P_m^*(x) over the interval [0, ell].

    The normalization is defined by:
        P_m^*(x) = shifted_legendre(m, ell, x) / (A_m * sqrt(ell))

    where:
        - A_m is a precomputed normalization constant (depends on m),
        - ell is the interval width,
        - shifted_legendre is the unnormalized shifted Legendre polynomial over [0, ell].

    Parameters
    ----------
    m : int
        Degree of the polynomial (must be a non-negative integer).

    ell : float
        Upper bound of the interval [0, ell]; must be strictly positive.

    x : float or np.ndarray
        Scalar or array of input values at which to evaluate the polynomial.

    Returns
    -------
    np.float64 or np.ndarray
        Normalized polynomial values:
        - Returns a scalar `np.float64` if input x is a scalar,
        - Returns an `np.ndarray` of dtype float64 if input x is array-like.

    Raises
    ------
    ValueError
        If the normalization coefficient A_m is not defined for degree m.
    """

    # ----------------------- Retrieve Normalization Coefficient ------------------------ #
    # The normalization coefficients A_m must be precomputed and accessible via coeff_A
    try:
        A_m = coeff_A[m]  # Get A_m for the given degree m
    except (IndexError, KeyError):
        # Raise an informative error if A_m is not defined
        raise ValueError(f"Normalization coefficient A_m is not defined for m = {m}")

    # ---------------------- Compute Shifted Legendre Polynomial ------------------------ #
    # Evaluate the base (unnormalized) shifted Legendre polynomial over the interval [0, ell]
    P_m_x = shifted_legendre(m, ell, x)

    # --------------------------- Apply Normalization ----------------------------------- #
    # Compute the normalization factor: A_m * sqrt(ell)
    norm_factor = A_m * np.sqrt(ell)

    # Divide the unnormalized polynomial by the normalization factor
    normalized = P_m_x / norm_factor

    # --------------------------- Return Type Handling ---------------------------------- #
    # Ensure output type matches the input type:
    # - If x is scalar, return a scalar np.float64
    # - If x is array-like, return a float64 NumPy array
    normalized = np.array(normalized, dtype=np.float64)
    return normalized.item() if np.isscalar(x) else normalized

# --------------------------------------------------------------------------- #
# Function: phi_m
# Purpose : Evaluate the m-th Galerkin basis function φ_m(x) over the interval [0, ell]
# --------------------------------------------------------------------------- #
def phi_m(m: int, ell: float, x: float | np.ndarray) -> np.float64 | np.ndarray:
    """
    Compute the m-th Galerkin basis function φ_m(x) on the interval [0, ell].

    The Galerkin basis is defined as:
        φ_m(x) = (sqrt(ell) / 2) * A_m * [P_{m+1}(x) - P_{m-1}(x)]

    where:
        - P_k(x) is the k-th shifted Legendre polynomial on [0, ell]
        - A_m is the normalization constant specific to index m

    Parameters
    ----------
    m : int
        Index of the Galerkin basis function (must be ≥ 1).
    
    ell : float
        Length of the domain interval [0, ell]; must be strictly positive.
    
    x : float or np.ndarray
        Input value(s) at which to evaluate the basis function.

    Returns
    -------
    np.float64 or np.ndarray
        Evaluated basis function φ_m(x):
        - Returns a scalar `np.float64` if x is scalar
        - Returns a `np.ndarray` of dtype float64 if x is array-like

    Raises
    ------
    ValueError
        If m < 1, or if normalization coefficient A_m is undefined.
    """

    # ----------------------------------------------------------------------- #
    # Step 1: Validate basis index m                                          #
    # ----------------------------------------------------------------------- #
    # Galerkin basis functions are only defined for m ≥ 1 due to the use of P_{m-1}
    if m < 1:
        raise ValueError("m must be ≥ 1 for Galerkin basis functions φ_m(x).")

    # ----------------------------------------------------------------------- #
    # Step 2: Retrieve normalization coefficient A_m                         #
    # ----------------------------------------------------------------------- #
    # These must be defined externally in a dictionary or list (e.g., coeff_A[m])
    try:
        A_m = coeff_A[m]
    except (IndexError, KeyError):
        raise ValueError(f"Normalization coefficient A_m is not defined for m = {m}.")

    # ----------------------------------------------------------------------- #
    # Step 3: Convert input to NumPy array (supports scalar and vector input) #
    # ----------------------------------------------------------------------- #
    x_array = np.asarray(x)

    # ----------------------------------------------------------------------- #
    # Step 4: Evaluate shifted Legendre polynomials P_{m+1}(x) and P_{m-1}(x) #
    # ----------------------------------------------------------------------- #
    P_plus  = shifted_legendre(m + 1, ell, x_array)   # P_{m+1}(x)
    P_minus = shifted_legendre(m - 1, ell, x_array)   # P_{m-1}(x)

    # ----------------------------------------------------------------------- #
    # Step 5: Compute φ_m(x) using the Galerkin basis formula                 #
    # ----------------------------------------------------------------------- #
    # Formula: φ_m(x) = (sqrt(ell) / 2) * A_m * (P_{m+1}(x) - P_{m-1}(x))
    sqrt_ell = np.sqrt(ell)
    phi_vals = (sqrt_ell / 2.0) * A_m * (P_plus - P_minus)

    # ----------------------------------------------------------------------- #
    # Step 6: Format output as np.float64 or np.ndarray                      #
    # ----------------------------------------------------------------------- #
    # Ensure output matches the type of input: scalar or array
    phi_vals = np.array(phi_vals, dtype=np.float64)
    return phi_vals.item() if np.isscalar(x) else phi_vals

# =============================================================================
# Global Variables (Must be defined externally in the environment)
# =============================================================================

# coeff_B: np.ndarray
#   Recurrence coefficients B_k for the second-derivative operator
#   Length must be ≥ N + 3
#
# coeff_C: np.ndarray
#   Recurrence coefficients C_k for the diagonal entries
#   Length must be ≥ N + 3


# =============================================================================
# Function: sys_soln
# Title   : Spectral-Galerkin Linear System Solver
# Purpose : Solve symmetric banded linear systems from Galerkin discretizations
# =============================================================================
def sys_soln(f: np.ndarray, N: int, a: float, b: float, ell: float) -> np.ndarray:
    """
    Solves a symmetric banded linear system arising from spectral-Galerkin
    discretization of a second-order PDE on the domain [0, ell].

    The solver uses a custom algorithm that:
        - Exploits matrix symmetry and 2-banded structure
        - Performs forward elimination with recurrence
        - Uses backward substitution to compute the solution

    Parameters
    ----------
    f : np.ndarray
        Right-hand side vector of shape (N,)
    N : int
        Number of unknowns; must be ≥ 2
    a : float
        Coefficient of the identity term in the PDE
    b : float
        Coefficient of the second-derivative term in the PDE
    ell : float
        Length of the spatial domain [0, ell]

    Returns
    -------
    np.ndarray
        Solution vector w of shape (N,), representing Galerkin coefficients

    Raises
    ------
    ValueError
        If N is less than 2
    """

    # =========================================================================
    # STEP 1: Input Validation
    # =========================================================================
    if N < 2:
        raise ValueError("N must be at least 2 for the system to be solvable.")

    # =========================================================================
    # STEP 2: Preallocate Arrays
    # =========================================================================
    # d: modified diagonal entries
    # z: transformed right-hand side vector
    # w: solution vector to be returned
    d = np.empty(N, dtype=np.float64)
    z = np.empty(N, dtype=np.float64)
    w = np.empty(N, dtype=np.float64)

    # =========================================================================
    # STEP 3: Initialize Diagonals and RHS for First Two Indices
    # =========================================================================
    diag_scale = (4 * b) / (a * ell ** 2)  # Constant appearing in weak form stiffness matrix

    # Initialize first diagonal entries and RHS
    d[0] = coeff_C[1] + diag_scale
    d[1] = coeff_C[2] + diag_scale
    z[0] = f[0]
    z[1] = f[1]

    # =========================================================================
    # STEP 4: Forward Elimination (Recurrence Sweep)
    # =========================================================================
    # Recurrence relations eliminate lower bandwidth terms and transform RHS

    half_N = (N + 1) // 2  # Ensures correct range for both even and odd N

    for j in range(2, half_N + 1):
        idx = 2 * (j - 1)  # Current even index: 2, 4, 6, ...

        if idx < N:
            # Update diagonal for even index using recursive formula
            d[idx] = (
                coeff_C[idx + 1] + diag_scale
                - (coeff_B[idx] ** 2) / d[idx - 2]
            )
            # Update transformed RHS vector
            z[idx] = f[idx] + (coeff_B[idx] * z[idx - 2]) / d[idx - 2]

        if idx + 1 < N:
            # Update diagonal for odd index (paired with even)
            d[idx + 1] = (
                coeff_C[idx + 2] + diag_scale
                - (coeff_B[idx + 1] ** 2) / d[idx - 1]
            )
            # Update RHS for odd index
            z[idx + 1] = f[idx + 1] + (coeff_B[idx + 1] * z[idx - 1]) / d[idx - 1]

    # =========================================================================
    # STEP 5: Backward Substitution (Solve Upper Triangular System)
    # =========================================================================
    # Starting from the last two values, solve recursively in reverse

    w[N - 1] = z[N - 1] / d[N - 1]
    w[N - 2] = z[N - 2] / d[N - 2]

    for j in range(half_N - 1, 0, -1):
        idx = 2 * (j - 1)  # Reverse index: ..., 4, 2, 0

        if idx + 2 < N:
            # Solve even index coefficient
            w[idx] = (z[idx] + coeff_B[idx + 2] * w[idx + 2]) / d[idx]

        if idx + 3 < N:
            # Solve odd index coefficient
            w[idx + 1] = (z[idx + 1] + coeff_B[idx + 3] * w[idx + 3]) / d[idx + 1]

    return w

# =========================================================================== #
#                            Quadrature Method Suite                          #
# =========================================================================== #

# =========================================================================== #
# Function: gauss_legendre_integral
# Purpose : Numerically integrate a function over [a, b] using Gauss–Legendre quadrature
# =========================================================================== #

def gauss_legendre_integral(f, a: float, b: float, n_gauss: int) -> np.float64:
    """
    Approximate the definite integral of a function f over the interval [a, b]
    using Gauss–Legendre quadrature with a specified number of nodes.

    This method is ideal for smooth integrands, as it minimizes integration
    error using optimally chosen sample points (nodes) and associated weights.

    Parameters
    ----------
    f : function
        The function to integrate. Should ideally accept NumPy arrays as input
        for vectorized evaluation, but scalar fallback is supported.
    
    a : float
        Lower bound of the integration interval.
    
    b : float
        Upper bound of the integration interval.
    
    n_gauss : int
        Number of Gauss–Legendre nodes to use (higher = better accuracy).

    Returns
    -------
    np.float64
        Approximate integral value over [a, b].

    Raises
    ------
    ValueError
        If the shape of the function's output does not match the shape of the evaluation nodes.
    """

    # -----------------------------------------------------------------------
    # STEP 1: Generate Gauss–Legendre nodes and weights on the interval [-1, 1]
    # -----------------------------------------------------------------------
    # Nodes are the x-values at which to evaluate f; weights are used for integration
    nodes, weights = leggauss(n_gauss)

    # -----------------------------------------------------------------------
    # STEP 2: Transform nodes from [-1, 1] to the interval [a, b]
    # -----------------------------------------------------------------------
    mid = 0.5 * (a + b)                 # Midpoint of the interval
    half_len = 0.5 * (b - a)            # Half of the interval length
    x_mapped = mid + half_len * nodes   # Affine transformation to [a, b]

    # -----------------------------------------------------------------------
    # STEP 3: Evaluate function f at the transformed nodes
    # -----------------------------------------------------------------------
    try:
        # Prefer vectorized evaluation for performance
        f_vals = np.asarray(f(x_mapped))

        # Confirm that the output shape matches the expected shape
        if f_vals.shape != x_mapped.shape:
            raise ValueError(
                f"Function output shape mismatch: expected {x_mapped.shape}, got {f_vals.shape}"
            )

    except Exception:
        # If vectorized evaluation fails, fall back to scalar loop
        f_vals = np.array([f(xi) for xi in x_mapped])

    # -----------------------------------------------------------------------
    # STEP 4: Compute the weighted sum and scale it to the interval length
    # -----------------------------------------------------------------------
    # The integral is approximated by: sum(w_i * f(x_i)) * (b - a)/2
    integral = half_len * np.dot(weights, f_vals)

    # -----------------------------------------------------------------------
    # STEP 5: Return result as high-precision NumPy float
    # -----------------------------------------------------------------------
    return np.float64(integral)

# =========================================================================== #
# Function: adaptive_gauss_legendre_integrator
# Purpose : Adaptive numerical integration over [0, ell] using:
#           1. Increasing Gauss–Legendre node count for accuracy,
#           2. Recursive subinterval refinement if convergence fails.
# Dependencies: Requires gauss_legendre_integral(f, a, b, n_gauss) to be defined.
# =========================================================================== #

def adaptive_gauss_legendre_integrator(
    f,                       # Integrand: any function accepting float input
    ell: float,              # Upper limit of integration interval [0, ell]
    tol: float = 1e-6,       # Tolerance for convergence
    min_dx: float = 1/128.0, # Minimum width of subinterval before halting refinement
    n_gauss: int = 5,        # Initial number of Gauss–Legendre nodes
    max_gauss: int = 50      # Maximum number of Gauss–Legendre nodes per interval
) -> tuple[np.float64, np.float64, int, int]:
    """
    Adaptively approximate the definite integral of `f` over [0, ell]
    using Gauss–Legendre quadrature and subinterval refinement.

    Parameters
    ----------
    f : function
        Function to integrate; must accept float inputs and return float outputs.
    ell : float
        Upper bound of the integration domain [0, ell]; must be ≥ 0.
    tol : float, optional
        Absolute error tolerance for convergence. Default is 1e-6.
    min_dx : float, optional
        Minimum subinterval width. Prevents infinite subdivision. Default is 1/128.
    n_gauss : int, optional
        Starting number of Gauss nodes per interval. Default is 5.
    max_gauss : int, optional
        Maximum allowable Gauss nodes in any interval. Default is 50.

    Returns
    -------
    tuple[np.float64, np.float64, int, int]
        - Estimated integral value
        - Estimated absolute error
        - Number of interval refinements performed
        - Maximum number of Gauss nodes used in any interval
    """

    # ---------------------------------------------------------------------------
    # STEP 1: Handle trivial case where ell = 0
    # ---------------------------------------------------------------------------
    if ell < 0:
        raise ValueError("Parameter 'ell' must be non-negative.")
    if ell == 0:
        return np.float64(0.0), np.float64(0.0), 0, 0  # Zero-length interval

    # ---------------------------------------------------------------------------
    # STEP 2: Try full interval integration with increasing Gauss node count
    # ---------------------------------------------------------------------------
    initial_n_gauss = n_gauss          # Save starting node count
    max_nodes_used = n_gauss           # Track highest node count used
    converged = False                  # Convergence flag

    # First approximation using initial node count
    integral_prev = gauss_legendre_integral(f, 0.0, ell, n_gauss)

    while n_gauss + 5 <= max_gauss:
        n_gauss += 5
        integral_curr = gauss_legendre_integral(f, 0.0, ell, n_gauss)

        max_nodes_used = min(max(max_nodes_used, n_gauss), max_gauss)

        if np.abs(integral_curr - integral_prev) < tol:
            # Converged within tolerance; return result
            estimated_error = np.abs(integral_curr - integral_prev)
            return (
                np.float64(integral_curr),
                np.float64(estimated_error),
                0,  # No interval splitting occurred
                max_nodes_used
            )
        integral_prev = integral_curr  # Continue refining

    # ---------------------------------------------------------------------------
    # STEP 3: Begin adaptive refinement by subinterval splitting
    # ---------------------------------------------------------------------------
    counter = 0                # Count of subdivision steps
    prev_total = None          # Store previous total for convergence check

    while not converged:
        counter += 1
        n_intervals = 2 ** counter      # Number of subintervals
        dx = ell / n_intervals          # Width of each subinterval

        if dx < min_dx:
            break  # Stop if intervals become too small

        total_integral = 0.0
        converged = True  # Will set False if any subinterval fails

        for i in range(n_intervals):
            a = i * dx
            b = (i + 1) * dx

            n_gauss_local = initial_n_gauss
            integral_prev = gauss_legendre_integral(f, a, b, n_gauss_local)
            local_converged = False

            while n_gauss_local + 5 <= max_gauss:
                n_gauss_local += 5
                integral_curr = gauss_legendre_integral(f, a, b, n_gauss_local)

                if np.abs(integral_curr - integral_prev) < tol:
                    # This subinterval converged
                    total_integral += integral_curr
                    local_converged = True
                    break

                integral_prev = integral_curr

            if not local_converged:
                # Accept best estimate, but mark global failure
                total_integral += integral_curr
                converged = False

            max_nodes_used = min(max(max_nodes_used, n_gauss_local), max_gauss)

        # -----------------------------------------------------------------------
        # STEP 4: Check for global convergence over full domain
        # -----------------------------------------------------------------------
        if converged:
            if prev_total is not None:
                if np.abs(total_integral - prev_total) < tol:
                    estimated_error = np.abs(total_integral - prev_total)
                    return (
                        np.float64(total_integral),
                        np.float64(estimated_error),
                        counter,
                        max_nodes_used
                    )

            prev_total = total_integral  # Store for next check

    # ---------------------------------------------------------------------------
    # STEP 5: Fallback – Return best estimate even if not converged
    # ---------------------------------------------------------------------------
    estimated_error = (
        np.abs(total_integral - prev_total)
        if prev_total is not None else np.float64(np.inf)
    )

    return (
        np.float64(total_integral),
        np.float64(estimated_error),
        counter,
        max_nodes_used
    )

# =============================================================================
# Module: compute_product_integral
# Purpose:
#   Compute a weighted integral over [0, ell] of the form:
#
#       ∫₀^ell f(x, *args) × φₘ(x) dx      if multiplier = "galerkin_basis"
#       ∫₀^ell f(x, *args) × P̂ₘ(x) dx      if multiplier = "norm_leg_poly"
#
#   - φₘ(x): Galerkin-style basis function
#   - P̂ₘ(x): Normalized shifted Legendre polynomial
#
#   Integration is performed using adaptive Gauss–Legendre quadrature.
# Dependencies:
#   - Requires: `phi_m`, `normalized_shifted_legendre`, and 
#     `adaptive_gauss_legendre_integrator` to be defined externally.
# =============================================================================

# =============================================================================
# Function: compute_product_integral
# Title   : Weighted Integral with Basis Multiplier
# =============================================================================
def compute_product_integral(
    f,                    # Function f(x, *args): user-defined integrand
    ell: float,           # Upper limit of integration interval [0, ell]
    m: int,               # Index of basis function (φₘ or P̂ₘ)
    *args,                # Additional positional arguments passed to f
    multiplier: str = "galerkin_basis",  # Type of basis multiplier
    **quad_kwargs         # Optional keyword arguments for quadrature tuning
) -> tuple[np.float64, np.float64]:
    """
    Compute the integral of f(x, *args) × φₘ(x) or f(x, *args) × P̂ₘ(x) over [0, ell].

    Parameters
    ----------
    f : callable
        Integrand function. Must accept float `x` as first argument, followed by optional *args.
    
    ell : float
        Upper limit of the integration interval. Must be strictly positive.
    
    m : int
        Index/order of the basis function.
    
    *args : tuple
        Additional positional arguments passed directly to `f(x, *args)`.

    multiplier : str, optional
        Which multiplier to apply to `f(x)` in the product:
        - "galerkin_basis" : Galerkin basis φₘ(x)
        - "norm_leg_poly"  : Normalized shifted Legendre polynomial P̂ₘ(x)

    **quad_kwargs : dict, optional
        Quadrature control parameters:
        - tol       : float, absolute tolerance (default = 1e-6)
        - min_dx    : float, minimum subinterval width (default = 1/128)
        - n_gauss   : int, initial Gauss nodes per interval (default = 5)
        - max_gauss : int, maximum Gauss nodes (default = 50)

    Returns
    -------
    tuple[np.float64, np.float64]
        - integral_val     : Result of the numerical integration
        - error_estimate   : Estimated absolute quadrature error
    """

    # -------------------------------------------------------------------------
    # Step 1: Validate integration domain
    # -------------------------------------------------------------------------
    if ell <= 0:
        raise ValueError("The integration upper bound 'ell' must be strictly positive.")

    # -------------------------------------------------------------------------
    # Step 2: Select basis function based on multiplier flag
    # -------------------------------------------------------------------------
    basis_dispatch = {
        "galerkin_basis": lambda x: phi_m(m, ell, x),  # Galerkin basis φₘ(x)
        "norm_leg_poly" : lambda x: normalized_shifted_legendre(m, ell, x)  # Normalized P̂ₘ(x)
    }

    if multiplier not in basis_dispatch:
        raise ValueError(
            f"Invalid 'multiplier' argument: {multiplier!r}. "
            f"Expected one of {list(basis_dispatch.keys())}."
        )

    basis_function = basis_dispatch[multiplier]  # Select corresponding basis function

    # -------------------------------------------------------------------------
    # Step 3: Define the product integrand f(x) × φₘ(x) or f(x) × P̂ₘ(x)
    # -------------------------------------------------------------------------
    def integrand(x):
        return f(x, *args) * basis_function(x)

    # -------------------------------------------------------------------------
    # Step 4: Set default quadrature settings and override via kwargs
    # -------------------------------------------------------------------------
    default_quadrature_options = {
        'tol': 1e-6,
        'min_dx': 1 / 128.0,
        'n_gauss': 5,
        'max_gauss': 50
    }

    # Merge user-supplied overrides with defaults
    quadrature_options = {
        key: quad_kwargs.get(key, default_quadrature_options[key])
        for key in default_quadrature_options
    }

    # -------------------------------------------------------------------------
    # Step 5: Compute integral using adaptive Gauss–Legendre quadrature
    # -------------------------------------------------------------------------
    integral_val, error_estimate, _, _ = adaptive_gauss_legendre_integrator(
        integrand,
        ell,
        **quadrature_options
    )

    # -------------------------------------------------------------------------
    # Step 6: Return final result and error estimate
    # -------------------------------------------------------------------------
    return np.float64(integral_val), np.float64(error_estimate)

# =============================================================================
# Function: compute_time_dependent_integrals
# Title   : Time-Dependent Spatial Projection Matrix
# Purpose : Compute integrals of the form ∫₀^ell f(x, t[k+1]) × φₘ₊₁(x) dx or
#           ∫₀^ell f(x, t[k+1]) × P̂ₘ₊₁(x) dx for use in Galerkin/spectral methods.
# =============================================================================

def compute_time_dependent_integrals(
    f,                        # Function to integrate: f(x, t)
    N: int,                   # Number of spatial basis functions (m = 1 to N)
    ell: float,               # Length of spatial domain [0, ell]
    t,                        # 1D array of time values (length ≥ 2)
    multiplier: str = "galerkin_basis",  # Basis type: φₘ or P̂ₘ
    **quad_kwargs             # Optional kwargs for quadrature control
) -> np.ndarray:
    """
    Compute a matrix of time-dependent spatial integrals of the form:

        integrals[k, m] ≈ ∫₀^ell f(x, t[k+1]) × φₘ₊₁(x) dx

    or using normalized shifted Legendre polynomials:

        ≈ ∫₀^ell f(x, t[k+1]) × P̂ₘ₊₁(x) dx

    These integrals are typically used to project time-dependent source terms
    onto a spatial basis for time-stepping PDE solvers.

    Parameters
    ----------
    f : callable
        Function of two variables: f(x, t). Called with a fixed t at each time step.

    N : int
        Number of spatial basis functions to compute (m = 1 to N).

    ell : float
        Length of spatial domain [0, ell]. Must be strictly positive.

    t : array-like of shape (n,)
        Array of time values. Must contain at least two entries.

    multiplier : str, optional
        Basis function to use in the integral:
        - "galerkin_basis" → Galerkin function φₘ(x)
        - "norm_leg_poly"  → Normalized shifted Legendre polynomial P̂ₘ(x)

    **quad_kwargs : dict, optional
        Additional arguments passed to `compute_product_integral`, such as:
        - tol: float, error tolerance
        - min_dx: float, smallest allowed interval
        - n_gauss: int, initial Gauss–Legendre node count
        - max_gauss: int, upper limit on Gauss nodes

    Returns
    -------
    np.ndarray of shape (n-1, N)
        Matrix of computed spatial integrals for each time step and basis index.
    """

    # -------------------------------------------------------------------------
    # STEP 1: Validate input time array
    # -------------------------------------------------------------------------
    t = np.asarray(t, dtype=np.float64)
    n = len(t)

    if n < 2:
        raise ValueError("Time array 't' must contain at least two time values.")
    if ell <= 0:
        raise ValueError("Parameter 'ell' must be strictly positive.")

    # -------------------------------------------------------------------------
    # STEP 2: Allocate output matrix for integral results
    # Shape: (n-1 time intervals) × (N basis functions)
    # -------------------------------------------------------------------------
    integrals = np.zeros((n - 1, N), dtype=np.float64)

    # -------------------------------------------------------------------------
    # STEP 3: Loop over time steps and basis function indices
    # -------------------------------------------------------------------------
    for k in range(n - 1):
        t_next = t[k + 1]  # Project onto basis at time t_{k+1}

        for m in range(N):
            # Compute spatial integral with basis index m+1
            result, _ = compute_product_integral(
                f,              # Integrand: f(x, t[k+1])
                ell,            # Domain upper bound
                m + 1,          # Basis function index (1-based)
                t_next,         # Fixed time passed to f(x, t)
                multiplier=multiplier,
                **quad_kwargs   # Forwarded quadrature options
            )

            # Store only the integral result (discard error estimate)
            integrals[k, m] = result

    # -------------------------------------------------------------------------
    # STEP 4: Return the final integral matrix
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

# =============================================================================
# Function: integrate_derivative_form
# Title   : Integrals Involving the First Derivative of f(x) over [0, ell]
# Purpose : Evaluate either:
#           - ∫₀^ell [f′(x)]² dx        (energy norm)         if form='squared'
#           - ∫₀^ell f′(x) × P̂ₘ(x) dx  (Legendre projection) if form='norm_leg_poly'
# =============================================================================

def integrate_derivative_form(
    f=None,                   # Callable function f(x) for numerical differentiation
    df=None,                  # Callable analytical derivative f′(x), if available
    ell=None,                 # Upper limit of integration domain [0, ell]
    form='squared',           # Integration type: 'squared' or 'norm_leg_poly'
    m=None,                   # Degree of Legendre polynomial (used if form = 'norm_leg_poly')
    h=1e-3,                   # Step size for numerical differentiation
    derivmeth='nd',           # 'nd' = external lib (e.g., numdifftools), 'sfd' = standard finite difference
    **quad_kwargs             # Additional quadrature options (tol, n_gauss, min_dx, max_gauss)
) -> tuple[float, float]:
    """
    Numerically compute one of the following integrals over [0, ell]:

        - Energy norm     : ∫₀^ell [f′(x)]² dx
        - Legendre projection : ∫₀^ell f′(x) × P̂ₘ(x) dx

    Parameters
    ----------
    f : callable, optional
        Function f(x) used for numerical differentiation if `df` is not provided.

    df : callable, optional
        Analytical derivative f′(x), overrides numerical differentiation if given.

    ell : float
        Length of the integration interval. Must be strictly positive.

    form : {'squared', 'norm_leg_poly'}, default='squared'
        Specifies the integral form to compute.

    m : int, optional
        Degree of the normalized shifted Legendre polynomial P̂ₘ(x).
        Required if form='norm_leg_poly'.

    h : float, default=1e-3
        Initial step size for numerical differentiation.

    derivmeth : {'nd', 'sfd'}, default='nd'
        Method for numerical differentiation:
            - 'nd'  : External library like `numdifftools`
            - 'sfd' : Fourth-order standard finite difference

    **quad_kwargs : dict
        Additional keyword arguments passed to `adaptive_gauss_legendre_integrator`.

    Returns
    -------
    integral : float
        Result of the integral over [0, ell].

    error_estimate : float
        Estimated integration error.
    """

    # =========================================================================
    # STEP 1: Input Validation
    # =========================================================================
    if ell is None or ell <= 0:
        raise ValueError("Parameter 'ell' must be a positive float.")

    if (f is None and df is None) or (f is not None and df is not None):
        raise ValueError("Specify exactly one of 'f' or 'df', not both or neither.")

    if form not in ('squared', 'norm_leg_poly'):
        raise ValueError("Parameter 'form' must be 'squared' or 'norm_leg_poly'.")

    if form == 'norm_leg_poly' and m is None:
        raise ValueError("Parameter 'm' is required when form='norm_leg_poly'.")

    if form == 'squared' and m is not None:
        warnings.warn("Parameter 'm' is ignored when form='squared'.", stacklevel=2)

    # =========================================================================
    # STEP 2: Adjust step size for differentiation stability
    # =========================================================================
    if f is not None:
        while h >= ell / 4:
            h /= 2  # Ensure step size is small enough for finite differences

    # =========================================================================
    # STEP 3: Construct the integrand function to be integrated
    # =========================================================================
    def integrand(x):
        """
        Evaluates the integrand at each x-point:
            - [f′(x)]²       if form = 'squared'
            - f′(x) × P̂ₘ(x) if form = 'norm_leg_poly'

        Parameters
        ----------
        x : float or np.ndarray
            Evaluation point(s) in [0, ell]

        Returns
        -------
        float or np.ndarray
            Value(s) of the integrand
        """
        x = np.atleast_1d(x)              # Ensure input is an array
        result = np.empty_like(x)         # Preallocate output

        for i, xi in enumerate(x):
            # Compute f′(x) from df or numerically
            if df is not None:
                f_prime = df(xi)
            else:
                f_prime, _ = first_order_derivative_unified(
                    f, xi, ell=ell, h_init=h, derivmeth=derivmeth
                )

            # Select integrand type
            if form == 'squared':
                result[i] = f_prime ** 2
            else:  # form == 'norm_leg_poly'
                Pm_val = normalized_shifted_legendre(m, ell, xi)
                result[i] = f_prime * Pm_val

        return result[0] if result.size == 1 else result

    # =========================================================================
    # STEP 4: Apply adaptive Gauss–Legendre quadrature to compute integral
    # =========================================================================
    integral, error_estimate, *_ = adaptive_gauss_legendre_integrator(
        integrand, ell, **quad_kwargs
    )

    return integral, error_estimate

# =============================================================================
# Function: compute_initial_integrals
# Title   : Compute Modal Galerkin Coefficients for Initial Conditions
# Purpose : Projects initial condition functions u(x), v(x) and their first and
#           second derivatives onto a normalized Legendre–Galerkin basis over [0, ell].
#           Handles both analytic and numerical differentiation.
# =============================================================================

def compute_initial_integrals(
    u, v, N, ell,
    *,
    du=None, dv=None,
    h: float = 1e-3,
    derivmeth: str = 'nd',
    **quad_kwargs
) -> dict:
    """
    Compute modal projection coefficients for u(x), v(x), u′(x), v′(x), u″(x), v″(x)
    onto normalized shifted Legendre basis functions φₘ(x) over [0, ell].

    Parameters
    ----------
    u, v : list of callable
        Initial functions u = [u₀(x), u₁(x)], v = [v₀(x), v₁(x)].

    du, dv : list of callable or None
        First derivative functions (or None to trigger numerical differentiation).

    N : int
        Number of basis functions (from m = 1 to N).

    ell : float
        Length of spatial domain [0, ell].

    h : float, optional
        Initial step size for numerical differentiation. Default is 1e-3.

    derivmeth : {'nd', 'sfd'}, optional
        Numerical differentiation method:
            - 'nd'  : external package (e.g., numdifftools)
            - 'sfd' : fourth-order standard finite difference

    **quad_kwargs : dict
        Passed to internal quadrature routines (e.g., tol, n_gauss, max_gauss).

    Returns
    -------
    dict
        Dictionary containing arrays of modal coefficients for each projection:
            - 'u_proj', 'v_proj'   : L² projections of u, v
            - 'diff1_u1', 'diff1_v1': Projections of first derivatives (via parts)
            - 'diff2_u', 'diff2_v' : Projections of second derivatives
    """

    # =========================================================================
    # STEP 1: Validate Input Arguments
    # =========================================================================

    def is_valid_func_list(lst):
        return isinstance(lst, list) and all(callable(f) for f in lst)

    def is_valid_deriv_list(lst):
        return isinstance(lst, list) and all(callable(f) or f is None for f in lst)

    if not (is_valid_func_list(u) and is_valid_func_list(v)):
        raise ValueError("Inputs 'u' and 'v' must be lists of callable functions.")

    du = [None] * len(u) if du is None else du
    dv = [None] * len(v) if dv is None else dv

    if not (is_valid_deriv_list(du) and is_valid_deriv_list(dv)):
        raise ValueError("Inputs 'du' and 'dv' must be lists of callables or None.")

    if not (len(u) == len(v) == len(du) == len(dv)):
        raise ValueError("All input lists ('u', 'v', 'du', 'dv') must be the same length.")

    if not isinstance(N, int) or N <= 0:
        raise ValueError("'N' must be a positive integer.")

    if not isinstance(ell, (int, float)) or ell <= 0:
        raise ValueError("'ell' must be a positive float.")

    # =========================================================================
    # STEP 2: Initialize Modal Coefficient Arrays
    # =========================================================================
    num_components = len(u)

    u_proj = [np.zeros(N) for _ in range(num_components)]       # ⟨uᵢ, φₘ⟩
    v_proj = [np.zeros(N) for _ in range(num_components)]       # ⟨vᵢ, φₘ⟩
    diff1_u1 = np.zeros(N)                                      # ⟨u₁′, φₘ⟩
    diff1_v1 = np.zeros(N)                                      # ⟨v₁′, φₘ⟩
    diff2_u = np.zeros((num_components, N))                     # ⟨uᵢ″, φₘ⟩
    diff2_v = np.zeros((num_components, N))                     # ⟨vᵢ″, φₘ⟩

    # =========================================================================
    # STEP 3: Loop Over Basis Indices (m = 1 to N)
    # =========================================================================
    for m in range(N):
        m_idx = m + 1  # Shift to 1-based indexing for basis φₘ

        # ---------------------------------------------------------------------
        # STEP 3.1: Project uᵢ, vᵢ onto Galerkin basis φₘ(x)
        # ---------------------------------------------------------------------
        for i in range(num_components):
            u_proj[i][m], _ = compute_product_integral(
                u[i], ell, m_idx,
                multiplier="galerkin_basis",
                **quad_kwargs
            )
            v_proj[i][m], _ = compute_product_integral(
                v[i], ell, m_idx,
                multiplier="galerkin_basis",
                **quad_kwargs
            )

        # ---------------------------------------------------------------------
        # STEP 3.2: Project first derivative ⟨u₁′, φₘ⟩ and ⟨v₁′, φₘ⟩
        # Done via integration by parts: ⟨f′, φₘ⟩ = -⟨f, φₘ′⟩
        # ---------------------------------------------------------------------
        diff1_u1[m] = compute_product_integral(
            lambda x: -u[1](x), ell, m_idx,
            multiplier="norm_leg_poly",
            **quad_kwargs
        )[0]
        diff1_v1[m] = compute_product_integral(
            lambda x: -v[1](x), ell, m_idx,
            multiplier="norm_leg_poly",
            **quad_kwargs
        )[0]

        # ---------------------------------------------------------------------
        # STEP 3.3: Project second derivatives ⟨uᵢ″, φₘ⟩ and ⟨vᵢ″, φₘ⟩
        # Use analytic or numerical differentiation
        # Done via integration by parts: ⟨f′′, φₘ⟩ = -⟨f′, φₘ′⟩
        # ---------------------------------------------------------------------
        for i in range(num_components):
            f_u = None if du[i] else (lambda x, i=i: -u[i](x))
            df_u = (lambda x, i=i: -du[i](x)) if du[i] else None

            diff2_u[i][m], _ = integrate_derivative_form(
                f=f_u, df=df_u, ell=ell,
                form='norm_leg_poly', m=m_idx,
                h=h, derivmeth=derivmeth,
                **quad_kwargs
            )

            f_v = None if dv[i] else (lambda x, i=i: -v[i](x))
            df_v = (lambda x, i=i: -dv[i](x)) if dv[i] else None

            diff2_v[i][m], _ = integrate_derivative_form(
                f=f_v, df=df_v, ell=ell,
                form='norm_leg_poly', m=m_idx,
                h=h, derivmeth=derivmeth,
                **quad_kwargs
            )

    # =========================================================================
    # STEP 4: Return Computed Coefficients in Dictionary Format
    # =========================================================================
    return {
        'u_proj'   : u_proj,        # Projections of uᵢ
        'v_proj'   : v_proj,        # Projections of vᵢ
        'diff1_u1' : diff1_u1,      # First derivative projection ⟨u₁′, φₘ⟩
        'diff1_v1' : diff1_v1,      # First derivative projection ⟨v₁′, φₘ⟩
        'diff2_u'  : diff2_u,       # Second derivative ⟨uᵢ″, φₘ⟩
        'diff2_v'  : diff2_v        # Second derivative ⟨vᵢ″, φₘ⟩
    }


# --------------------------------------------------------------------------- #
""" Construction of operator matrices (stencils)
    derived via the Legendre–Galerkin spectral method """
# --------------------------------------------------------------------------- #

# ==============================================================================
# Function: associated_identity_operator
# Title   : Assemble Symmetric Mass-Like Identity Operator Matrix (Sparse)
# Purpose : Constructs a symmetric sparse matrix using a 3-point stencil pattern,
#           often used in spectral Galerkin discretizations with Legendre polynomials.
#           The matrix is assembled using pre-defined coefficients:
#           - Main diagonal  →  C(m+1)
#           - ±2 diagonals   → -B(m+2)
# ==============================================================================

def associated_identity_operator(N: int) -> csr_matrix:
    """
    Assemble a symmetric sparse mass-like identity operator for Galerkin methods.

    Parameters
    ----------
    N : int
        Number of basis functions (i.e., the size of the operator matrix).

    Returns
    -------
    csr_matrix
        Symmetric sparse matrix (CSR format) with diagonals populated as:
            - Main diagonal:    C(m+1)
            - Off-diagonals:   -B(m+2) at positions ±2
    """
    
    # ==========================================================================
    # STEP 1: Compute Main Diagonal
    # Contains C(m+1) entries for m = 0 to N-1
    # ==========================================================================
    main_diag = np.array([coeff_C[m + 1] for m in range(N)], dtype=np.float64)

    # ==========================================================================
    # STEP 2: Compute Off-Diagonals (±2)
    # Contains -B(m+2) entries for m = 0 to N-3 (since it shifts ±2)
    # ==========================================================================
    off_diag = np.array([-coeff_B[m + 2] for m in range(N - 2)], dtype=np.float64)

    # ==========================================================================
    # STEP 3: Assemble Sparse Matrix Using SciPy's diags()
    # Place the main and off-diagonals at offsets [0, -2, 2]
    # CSR format ensures efficient row-based storage/access
    # ==========================================================================
    H = diags(
        diagonals=[main_diag, off_diag, off_diag],  # Three diagonals
        offsets=[0, -2, 2],                         # Positions: center and ±2
        shape=(N, N),                               # Matrix shape
        format="csr"                                # Compressed Sparse Row format
    )

    return H

# ==============================================================================
# Function: associated_first_order_operator
# Title   : Assemble First-Order Skew-Symmetric Operator Matrix (Sparse)
# Purpose : Constructs a sparse matrix for the first derivative operator
#           using a ±1 stencil. This is typically used in Galerkin-type
#           formulations involving spectral methods or orthogonal polynomials.
#           The operator is skew-symmetric by design.
# ==============================================================================

def associated_first_order_operator(N: int) -> csr_matrix:
    """
    Assemble a skew-symmetric sparse matrix representing the first derivative
    operator using a ±1 diagonal stencil.

    Parameters
    ----------
    N : int
        Number of basis functions (i.e., matrix size). Must be ≥ 2.

    Returns
    -------
    csr_matrix
        Sparse skew-symmetric matrix (CSR format) where:
            - Upper diagonal (offset +1):  A(m+1) * A(m+2)
            - Lower diagonal (offset -1): -A(m+1) * A(m+2)
    """

    # ==========================================================================
    # STEP 1: Input Validation
    # Ensure sufficient size to support ±1 off-diagonal structure
    # ==========================================================================
    if not isinstance(N, int) or N < 2:
        raise ValueError("N must be an integer ≥ 2 for a valid ±1 stencil.")

    # ==========================================================================
    # STEP 2: Compute Upper Diagonal Elements
    # For m = 0 to N-2, construct A(m+1) * A(m+2)
    # This defines the upper diagonal (i, i+1) of the matrix
    # ==========================================================================
    upper_diag = np.array([
        coeff_A[m + 1] * coeff_A[m + 2] for m in range(N - 1)
    ], dtype=np.float64)

    # ==========================================================================
    # STEP 3: Compute Lower Diagonal Elements
    # Skew-symmetry requires lower diagonal = -upper_diag
    # ==========================================================================
    lower_diag = -upper_diag

    # ==========================================================================
    # STEP 4: Assemble the Sparse Matrix
    # - Upper diagonal placed at offset +1
    # - Lower diagonal placed at offset -1
    # ==========================================================================
    B = diags(
        diagonals=[lower_diag, upper_diag],  # Diagonals: below and above the main
        offsets=[-1, 1],                     # Positions relative to the main diagonal
        shape=(N, N),                        # Matrix dimensions
        format="csr"                         # Use Compressed Sparse Row for efficiency
    )

    return B


# ==============================================================================
# Function: associated_operators
# Title   : Galerkin Operator Dispatcher
# Purpose : Return the appropriate Galerkin operator matrix based on user input.
#           Supports identity (mass-like) and first-order derivative operators.
# ==============================================================================

def associated_operators(N: int, operator: str) -> csr_matrix:
    """
    Dispatches to the correct operator assembly routine based on `operator` type.

    Parameters
    ----------
    N : int
        Number of basis functions (matrix dimension).

    operator : str
        Type of operator to assemble:
            - "identity"      : Returns symmetric mass-like identity matrix.
            - "first-order"   : Returns skew-symmetric derivative matrix.

    Returns
    -------
    csr_matrix
        Sparse matrix (in CSR format) representing the selected operator.

    Raises
    ------
    ValueError
        If the operator type is unrecognized.
    """
    if operator == "identity":
        return associated_identity_operator(N)
    elif operator == "first-order":
        return associated_first_order_operator(N)
    else:
        raise ValueError(
            f"Unknown operator type '{operator}'. "
            "Use 'identity' or 'first-order'."
        )


# ==============================================================================
# Function: galerkin_stencils
# Title   : Apply Galerkin Operator to Vector
# Purpose : Applies a selected sparse Galerkin operator matrix to an input vector.
#           Useful in spectral-Galerkin schemes involving modal transformations.
# ==============================================================================

def galerkin_stencils(N: int, v: np.ndarray, operator: str = "identity") -> np.ndarray:
    """
    Apply a Galerkin operator matrix to a vector via sparse matrix multiplication.

    Parameters
    ----------
    N : int
        Size of the Galerkin operator matrix (and expected vector length).

    v : np.ndarray
        Input vector of shape (N,). Represents modal coefficients or function samples.

    operator : str, default="identity"
        Operator type to apply ("identity" or "first-order").

    Returns
    -------
    np.ndarray
        Output vector resulting from matrix-vector product: A @ v

    Raises
    ------
    ValueError
        If input vector `v` has incompatible shape.
    """
    if v.shape[0] != N:
        raise ValueError(
            f"Input vector length mismatch: expected {N}, got {v.shape[0]}."
        )

    A = associated_operators(N, operator)  # Retrieve sparse operator matrix
    return A.dot(v)                        # Perform matrix-vector multiplication

# ==============================================================================
# Function: condition_number_associated_matrix
# Title   : Condition Number of Modified Galerkin System Matrix
# Purpose : Compute the 2-norm condition number κ₂(A) for a matrix of the form:
#               A = H + (4b / (a * ell²)) * I
#           where H is the Galerkin identity (mass-like) operator.
# ==============================================================================

def condition_number_associated_matrix(
    N: int,
    ell: float,
    a: float,
    b: float
) -> np.float64:
    """
    Compute the 2-norm condition number κ₂(A) of a Galerkin system matrix:
        A = H + (4b / (a * ell²)) * I

    Parameters
    ----------
    N : int
        Number of basis functions (size of the Galerkin matrix).
    
    ell : float
        Length of the spatial domain [0, ell]; must be strictly positive.

    a : float
        Physical coefficient for the Galerkin operator term.

    b : float
        Coefficient applied to the identity term in A.

    Returns
    -------
    np.float64
        The 2-norm condition number of matrix A, as a float64.
    """

    # ----------------------------------------------------------------------
    # STEP 1: Input Validation
    # ----------------------------------------------------------------------
    if not isinstance(N, int) or N <= 0:
        raise ValueError("Parameter 'N' must be a positive integer.")

    if not isinstance(ell, (int, float)) or ell <= 0:
        raise ValueError("Parameter 'ell' must be a strictly positive float.")

    if not isinstance(a, (int, float)) or a == 0:
        raise ValueError("Parameter 'a' must be a non-zero number.")

    if not isinstance(b, (int, float)):
        raise ValueError("Parameter 'b' must be a numeric value.")

    # ----------------------------------------------------------------------
    # STEP 2: Assemble the System Matrix A
    # ----------------------------------------------------------------------

    # Retrieve the symmetric mass-like operator H using a Galerkin stencil
    H = associated_operators(N, operator="identity")  # H is in sparse CSR format

    # Compute scaling factor for identity matrix component
    scalar = (4 * b) / (a * ell**2)  # Derived from the physical PDE formulation

    # Assemble the scaled identity operator in CSR sparse format
    I_scaled = scalar * identity(N, format="csr")

    # Combine Galerkin and identity contributions to form A
    A = H + I_scaled  # Still sparse, efficient for large N

    # ----------------------------------------------------------------------
    # STEP 3: Condition Number Computation
    # ----------------------------------------------------------------------

    # Convert sparse matrix to dense array for use with SciPy's cond()
    A_dense = A.toarray()

    # Compute the condition number using 2-norm (spectral norm)
    kappa_2 = cond(A_dense, p=2)

    # Cast result to consistent NumPy float64 type for precision
    return np.float64(kappa_2)

# ==============================================================================
# Function: galerkin_approx
# Title   : Galerkin Series Evaluation Over Space
# Purpose : Computes the Galerkin approximation u(x) as a linear combination
#           of orthonormal basis functions φₘ(x), using precomputed coefficients.
#           Handles both scalar and array-valued spatial inputs.
# ==============================================================================

def galerkin_approx(
    ell: float,
    coeff: np.ndarray,
    x: float | np.ndarray
) -> np.ndarray | np.float64:
    """
    --------------------------------------------------------------------------
    Method: galerkin_approx
    --------------------------------------------------------------------------
    Evaluate the Galerkin approximation:
        u(x) ≈ Σₘ coeff[i, m-1] · φₘ(x)
    using precomputed modal coefficients for each time layer.

    Parameters
    ----------
    ell : float
        Length of the spatial domain [0, ell].

    coeff : np.ndarray
        Coefficient matrix of shape (n, N), where:
            - n  = number of time layers (or trials)
            - N  = number of basis functions (φ₁ to φ_N)

    x : float or np.ndarray
        Spatial coordinate(s) at which to evaluate u(x). Can be scalar or array-like.

    Returns
    -------
    np.ndarray or np.float64
        Approximation u(x). Return shape:
            - (n,)              → if x is scalar
            - (n, len(x))       → if x is an array of points
    """

    # ==========================================================================
    # STEP 1: Input Normalization
    # ==========================================================================

    ell = np.float64(ell)                     # Ensure ell is float64 for precision
    coeff = np.asarray(coeff, dtype=np.float64)  # Convert coeff to NumPy array if needed

    N = coeff.shape[1]                        # Number of basis functions
    is_scalar = np.isscalar(x)                # Remember if x was scalar for output formatting

    x = np.atleast_1d(x).astype(np.float64)   # Ensure x is 1D float64 array for consistency

    # ==========================================================================
    # STEP 2: Basis Function Evaluation
    # ==========================================================================
    # Compute φₘ(x) for m = 1 to N
    # Output: shape (N, len(x)), where each row is φₘ(x) for a specific m
    phi_vals = np.array([
        phi_m(m + 1, ell, x)  # m + 1 since φ₁ corresponds to index m=0
        for m in range(N)
    ], dtype=np.float64)

    # ==========================================================================
    # STEP 3: Modal Expansion Computation
    # ==========================================================================
    # Multiply coeff (shape: n x N) with phi_vals (shape: N x len(x))
    # Result shape: (n, len(x)) — evaluates the Galerkin sum for each time layer
    result = coeff @ phi_vals

    # ==========================================================================
    # STEP 4: Return Formatted Output
    # ==========================================================================
    # If x was scalar, return shape (n,)
    return result[:, 0] if is_scalar else result

# ==============================================================================
# Function: exact_solution_on_grid
# Title   : Evaluate Analytical Solution of Timoshenko Beam Model on Grid
# Purpose : Computes the analytical solution u(x, t) or v(x, t) across either:
#           - a uniform spatial grid (over full or specific time steps), or
#           - a specific spatial location over time.
#           Useful for benchmarking or comparison with numerical results.
# ==============================================================================

def exact_solution_on_grid(
    func: callable,
    config,
    unif_prt_spc: int = None,
    x_val: float = None,
    k: int = None
) -> np.ndarray | float:
    """
    --------------------------------------------------------------------------
    Method: exact_solution_on_grid
    --------------------------------------------------------------------------
    Evaluate the exact (analytical) solution u(x, t) or v(x, t) for the
    Timoshenko beam model either:
        - across a spatial grid for all time steps,
        - or at a specific spatial location across time,
        - or at a specific time and location(s).

    Parameters
    ----------
    func : callable
        The analytical solution function of the form func(x, t).
        Must accept vector-valued `x` and scalar `t`.

    config : object
        Configuration object with attributes:
            - ell (float): Length of the spatial domain.
            - t (np.ndarray): Discrete time values.
            - n (int): Number of time steps minus one (len(t) - 1).

    unif_prt_spc : int, optional
        Number of uniform spatial subintervals. Defines the grid.
        If given, generates a grid of `unif_prt_spc + 1` points.

    x_val : float, optional
        Specific spatial location in [0, ell] for single-point evaluation.

    k : int, optional
        Time index. If provided, returns data only at that time step.

    Returns
    -------
    np.ndarray or float
        Solution values:
            - If `k` is None: array of shape (len(t), len(x)).
            - If `k` is given: array of shape (len(x),) or scalar.
    """

    # ==========================================================================
    # STEP 1: Validate Inputs
    # ==========================================================================

    # Ensure at least one spatial evaluation mode is requested
    if x_val is None and unif_prt_spc is None:
        raise ValueError("You must provide either `x_val` or `unif_prt_spc`.")

    # ==========================================================================
    # STEP 2: Define Spatial Evaluation Points
    # ==========================================================================

    if x_val is not None:
        # Single-point evaluation mode
        if not (0 <= x_val <= config.ell):
            raise ValueError(f"x_val = {x_val} is outside the domain [0, {config.ell}].")

        x = np.array([x_val])  # Convert to 1-element array for consistency
    else:
        # Generate uniformly spaced grid: includes endpoints
        x = np.linspace(0, config.ell, unif_prt_spc + 1)

    # ==========================================================================
    # STEP 3: Evaluate func(x, t_i) at All Time Steps
    # ==========================================================================

    # Vectorized loop over time: apply func(x, t_i) at each t_i
    values = np.array([func(x, t_i) for t_i in config.t])

    # Shape of `values`:
    #   - (len(t), len(x)) if x is grid
    #   - (len(t), 1) if x is a single point

    # ==========================================================================
    # STEP 4: Return Time Slice or Full Time Evolution
    # ==========================================================================

    if k is not None:
        # Extract solution at specific time step
        if not (0 <= k <= config.n):
            raise ValueError(f"Invalid time index k={k}. Must be in range [0, {config.n}].")
        return values[k]  # 1D array or scalar depending on x

    # Return all time steps (e.g., for animation or full solution plot)
    return values

# ==============================================================================
# Function: callable_exact_solution
# Title   : Flexible Evaluation of Exact Timoshenko Beam Solution
# Purpose : Supports evaluation or callable generation for the analytical
#           solution u(x, t) or v(x, t) of the Timoshenko beam model.
#           Can evaluate:
#             - single time-step function at many x,
#             - all time steps at fixed x,
#             - single (x, t) pair,
#             - return callable objects.
# ==============================================================================

def callable_exact_solution(
    func: callable,
    config,
    k: int = None,
    x_vals: float | int | list | np.ndarray = None
):
    """
    Flexibly evaluate the exact analytical solution u(x, t) or v(x, t)
    from the Timoshenko beam model.

    Depending on input, the output can be:
    - a single callable u_k(x) = u(x, t_k),
    - a value at fixed (x, t_k),
    - an array of u(x_val, t_k) over t_k,
    - or a list of all callables u_k(x).

    Parameters
    ----------
    func : callable
        The exact solution function with signature func(x, t), supporting:
            - vectorized x (scalar or array)
            - scalar t
        Should return scalar or array outputs.

    config : object
        Configuration object with:
            - config.t : np.ndarray
                Discrete time values (shape = (n+1,))
            - config.n : int
                Number of time steps (typically len(config.t) - 1)

    k : int, optional
        Specific time index to use for fixed-time evaluation.
        If None, all time steps are processed.

    x_vals : float | int | list | np.ndarray, optional
        Spatial value(s) to evaluate. If None, returns callable(s) instead of evaluating.

    Returns
    -------
    callable | list[callable] | float | np.ndarray
        Output depends on input combination:
        - (k only):          Returns callable func_k(x) = func(x, t_k)
        - (k and x_vals):    Returns evaluated result func(x_vals, t_k)
        - (x_vals only):     Returns np.ndarray of func(x_vals, t_i) for all i
        - (neither given):   Returns list of all callables func_i(x)
    """

    # ==========================================================================
    # STEP 1: Validate and Normalize Spatial Input
    # ==========================================================================
    def validate_and_convert_x_vals(x_input):
        """Standardizes x_vals into float or float array."""
        if isinstance(x_input, (float, int)):
            return float(x_input)
        elif isinstance(x_input, list):
            return np.array(x_input, dtype=float)
        elif isinstance(x_input, np.ndarray):
            return x_input.astype(float)
        elif x_input is None:
            return None
        else:
            raise TypeError("x_vals must be float, int, list, np.ndarray, or None.")

    x_vals = validate_and_convert_x_vals(x_vals)

    # ==========================================================================
    # STEP 2: Time-Fixed Callable Generator
    # ==========================================================================
    def construct_exact_function_at_k(k_idx: int) -> callable:
        """
        Create a callable x ↦ func(x, t_k_idx) for a fixed time index.
        """
        if not (0 <= k_idx <= config.n):
            raise ValueError(f"Time index k = {k_idx} must be within [0, {config.n}].")
        return lambda x: func(x, config.t[k_idx])

    # ==========================================================================
    # CASE 1: Time index k is provided
    # ==========================================================================
    if k is not None:
        fn = construct_exact_function_at_k(k)
        return fn if x_vals is None else fn(x_vals)

    # ==========================================================================
    # CASE 2: Time index not provided → process over all time steps
    # ==========================================================================
    all_functions = [construct_exact_function_at_k(k_idx) for k_idx in range(config.n + 1)]

    if x_vals is None:
        return all_functions  # Return list of callables for all time steps
    else:
        # Evaluate each time-specific function at the same spatial input
        return np.array([fn(x_vals) for fn in all_functions])

# ==============================================================================
# Function: kahan_babuska_neumaier_sum
# Title   : Accurate Floating-Point Summation Using Kahan–Babuška–Neumaier Method
# Purpose : Improve numerical stability when summing floats, especially when
#           large and small magnitudes are mixed (to reduce round-off errors).
# ==============================================================================

def kahan_babuska_neumaier_sum(numbers) -> np.float64:
    """
    ------------------------------------------------------------
    Accurate Summation Using Kahan–Babuška–Neumaier Algorithm
    ------------------------------------------------------------
    Performs a numerically stable summation of floating-point numbers
    by tracking and correcting rounding errors at each step.

    Parameters
    ----------
    numbers : list or np.ndarray
        Sequence of floats (1D or multi-D) to sum.

    Returns
    -------
    np.float64
        Accurately summed result using compensated summation.

    Raises
    ------
    TypeError
        If any element is not a scalar numeric type (float or int).
    """

    # -------------------------------------------------------------------------
    # STEP 1: Flatten input to 1D array for consistent iteration
    # -------------------------------------------------------------------------
    flat = np.asarray(numbers).ravel()  # Ensure NumPy array, then flatten to 1D

    total = np.float64(0.0)             # Accumulator for the running total
    compensation = np.float64(0.0)      # Tracks small errors lost in floating-point math

    # -------------------------------------------------------------------------
    # STEP 2: Iterate through numbers, applying compensated summation
    # -------------------------------------------------------------------------
    for x in flat:
        if isinstance(x, np.ndarray):
            raise TypeError("Nested arrays are not supported; elements must be scalar.")

        x = np.float64(x)  # Promote to consistent float64 type

        temp = total + x   # Tentative sum

        # Apply compensation depending on which operand has greater magnitude
        if abs(total) >= abs(x):
            compensation += (total - temp) + x
        else:
            compensation += (x - temp) + total

        total = temp  # Update main accumulator

    # -------------------------------------------------------------------------
    # STEP 3: Return final corrected total (sum + error compensation)
    # -------------------------------------------------------------------------
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
    # MODULE IMPORTS (local to reduce global dependency footprint)
    # =========================================================================
    from pathlib import Path             # For file path creation and management
    from datetime import datetime        # For timestamping output filenames
    import matplotlib.pyplot as plt      # For generating plots
    from matplotlib import rcParams      # For detailed plot styling

    # =========================================================================
    # CONFIGURE MATPLOTLIB FOR LaTeX-STYLED RENDERING
    # =========================================================================
    rcParams["text.usetex"] = True
    rcParams["font.family"] = "lmodern"  # Consistent math-friendly font
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
    color_u = "#0072B2"  # Blue: displacement u (Okabe–Ito palette)
    color_v = "#E69F00"  # Orange: rotation v (Okabe–Ito palette)

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================
    if not (len(time_array) and len(error_u) and len(error_v)):
        raise ValueError("Inputs 'time_array', 'error_u', and 'error_v' must be non-empty.")
    
    if not (len(time_array) == len(error_u) == len(error_v)):
        raise ValueError("Input arrays must be of equal length.")

    # =========================================================================
    # PREPARE OUTPUT DIRECTORY STRUCTURE
    # =========================================================================
    config_name = getattr(config, "name", "config")  # Default if no name attribute
    output_path = Path(output_dir) / config_name     # Save under 'output_dir/config_name'
    output_path.mkdir(parents=True, exist_ok=True)   # Create directory if missing

    # =========================================================================
    # GENERATE TIMESTAMP AND TIME RANGE LABELS
    # =========================================================================
    t_min, t_max = float(time_array[0]), float(time_array[-1])  # Time range for x-axis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")        # Unique suffix

    # =========================================================================
    # PLOT 1: L2 ERROR FOR DISPLACEMENT u(x, t)
    # =========================================================================
    plt.figure(figsize=(8, 4))
    plt.plot(
        time_array, error_u,
        marker='o', linestyle='-', linewidth=LINE_WIDTH,
        color=color_u,
        label=r"$E_{1,k} = \left\| u\left( \cdot, t_k \right) - \tilde{u}_{k,N}\left( \cdot \right) \right\|$"
    )
    plt.xlabel(rf"Time $t \in \left[ {t_min:g}, {t_max:g} \right]$")
    plt.ylabel(r"$E_{1, k}$")
    plt.title(r"$L^2$ Error Evolution for $u(x, t)$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save figure to output path
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
        label=r"$E_{2,k} = \left\| v\left( \cdot, t_k \right) - \tilde{v}_{k,N}\left( \cdot \right) \right\|$"
    )
    plt.xlabel(rf"Time $t \in \left[ {t_min:g}, {t_max:g} \right]$")
    plt.ylabel(r"$E_{2, k}$")
    plt.title(r"$L^2$ Error Evolution for $v(x, t)$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save figure to output path
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
#   at a fixed time index `k`. The plot is LaTeX-styled and saved as a high-
#   quality, timestamped PDF using a colorblind-friendly (Okabe–Ito) palette.
#
# Inputs:
#   - exact_soln    : Callable returning exact solution values, accepting (x, t)
#   - approx_solver : Object that provides a method `callable_compute_ansatz()`
#   - solution_type : 'u' for displacement or 'v' for rotation
#   - time_layer    : Time index k (must satisfy 0 ≤ k ≤ config.n)
#   - config        : Configuration object with attributes: ell, t, N, n, (opt) name
#   - output_dir    : Output directory to store the PDF file (default: "plots")
#
# Output:
#   - str : Full file path to the saved PDF plot
# =============================================================================

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
        Compares the exact analytical solution to a Galerkin-based approximation 
        at a specific time layer. Produces a LaTeX-styled, publication-quality PDF 
        plot that overlays both curves on the spatial domain.

    Parameters:
        exact_soln (callable): Exact solution function accepting (x, t).
        approx_solver (object): Must implement callable_compute_ansatz().
        solution_type (str): 'u' (displacement) or 'v' (rotation).
        time_layer (int): Time index to evaluate (0 ≤ time_layer ≤ config.n).
        config (object): Simulation config with `ell`, `t`, `N`, `n`, optionally `name`.
        output_dir (str): Directory to store output (default: "plots").

    Returns:
        str: Path to the saved PDF plot.
    """
    
    # =========================================================================
    # MODULE IMPORTS (local to reduce global dependency footprint)
    # =========================================================================
    from pathlib import Path                # Handles filesystem paths across OSes
    from datetime import datetime           # For unique timestamped file naming
    import numpy as np                      # Numerical operations and array handling
    import matplotlib.pyplot as plt         # Main plotting API
    from matplotlib import rcParams         # For advanced plot style configurations

    # =========================================================================
    # VALIDATE INPUTS
    # =========================================================================
    if not (0 <= time_layer <= config.n):
        raise ValueError(f"time_layer must be in the range [0, {config.n}]")

    # =========================================================================
    # PLOT SETTINGS
    # =========================================================================
    LINE_WIDTH = 3.0      # Line thickness for plotting
    num_points = 201      # Number of x-samples for evaluating the solution

    # =========================================================================
    # EVALUATE SOLUTIONS ON SPATIAL GRID
    # =========================================================================
    x_vals = np.linspace(0, config.ell, num_points)   # Uniform grid over [0, ell]
    t_k = config.t[time_layer]                        # Get the time value at layer k

    # Compute the exact solution at (x, t_k)
    exact_values = exact_soln(x_vals, t_k)

    # Compute the Galerkin approximation at (x, t_k)
    approx_values = approx_solver.callable_compute_ansatz(
        solution_type=solution_type,   # Specify field: 'u' or 'v'
        k=time_layer,                  # Time index
        x_vals=x_vals                  # Spatial evaluation points
    )

    # =========================================================================
    # COLOR SETTINGS (Colorblind-safe Okabe–Ito palette)
    # =========================================================================
    color_exact = "#009E73"   # Green for exact
    color_approx = "#D55E00"  # Orange for approximation

    # =========================================================================
    # CONFIGURE LaTeX-STYLED RENDERING
    # =========================================================================
    rcParams["text.usetex"] = True
    rcParams["font.family"] = "lmodern"
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
    # CREATE OUTPUT DIRECTORY
    # =========================================================================
    config_name = getattr(config, "name", "config")          # Use config.name if defined
    output_path = Path(output_dir) / config_name             # e.g., plots/config1/
    output_path.mkdir(parents=True, exist_ok=True)           # Create full path if missing

    # =========================================================================
    # CREATE COMPARISON PLOT
    # =========================================================================
    plt.figure(figsize=(8, 4))

    # Plot: Exact solution curve
    plt.plot(
        x_vals,
        exact_values,
        label=rf"Exact: ${solution_type}\left( x, {t_k:g} \right)$",
        color=color_exact,
        linestyle='-',
        linewidth=LINE_WIDTH
    )

    # Plot: Approximate (Galerkin) solution curve
    plt.plot(
        x_vals,
        approx_values,
        label=rf"Approximate: $\tilde{{{solution_type}}}_{{k,N}}\left( x \right)$",
        color=color_approx,
        linestyle='--',
        linewidth=LINE_WIDTH
    )

    # Axes and legend formatting
    plt.xlabel(rf"Spatial coordinate $x \in \left[0, {config.ell:g} \right]$")
    plt.ylabel("Solution value")
    plt.title(rf"Exact vs Approximate Solution: ${solution_type}\left( x, t_{{{time_layer}}} \right)$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # =========================================================================
    # SAVE PLOT TO FILE
    # =========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Unique identifier
    filename = output_path / (
        f"{config_name}_solution_{solution_type}_t{time_layer}_N{config.N}_{timestamp}.pdf"
    )
    plt.savefig(filename)  # Save as high-quality vector PDF
    plt.close()            # Release memory and avoid overlap on next plot

    # =========================================================================
    # RETURN FULL FILE PATH TO THE SAVED PLOT
    # =========================================================================
    return str(filename)

# =============================================================================
# METHOD: plot_approx_solution_at_time_k
# =============================================================================
# Title:
#     Plot Galerkin Approximate Solution at Discrete Time Layers
#
# Description:
#     Generates LaTeX-styled, publication-quality plots of the Galerkin-based 
#     approximate solution at five equally spaced time layers in the simulation 
#     domain (t = 0, T/4, T/2, 3T/4, T). Each snapshot is saved as a 
#     timestamped PDF for traceability.
#
# Parameters:
#     approx_solver (object): Solver instance with callable_compute_ansatz() method.
#     solution_type (str): Solution field to evaluate: 'u' (displacement) or 'v' (rotation).
#     config (object): Configuration object with fields like `ell`, `n`, `N`, `t`, and optional `name`.
#     output_dir (str): Directory to save the plots (default is "plots").
#
# Returns:
#     list[str]: List of absolute paths to the saved PDF plots.
# =============================================================================
def plot_approx_solution_at_time_k(
    approx_solver: object,
    solution_type: str,
    config,
    output_dir: str = "plots"
) -> list[str]:
    """
    Title: Plot Galerkin Approximate Solution at Discrete Time Layers

    Description:
        Generates publication-quality plots of the Galerkin-based approximate
        solution at five time layers: t = 0, T/4, T/2, 3T/4, T (clamped to grid).
        Each snapshot is saved as a timestamped PDF (falls back to PNG if LaTeX is unavailable).

    Parameters:
        approx_solver (object): Must implement callable_compute_ansatz(solution_type) -> callable.
        solution_type (str): 'u' (displacement) or 'v' (rotation).
        config (object): Simulation configuration/solver with .ell, .n, .N, .t (optional), .name (optional).
        output_dir (str): Directory to save the plots (default: "plots").

    Returns:
        list[str]: List of paths to the saved files.
    """

    # =========================================================================
    # MODULE IMPORTS (Scoped locally to minimize global side effects)
    # =========================================================================
    from pathlib import Path
    from datetime import datetime
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    # =========================================================================
    # PLOT AND SAMPLE CONFIGURATION
    # =========================================================================
    LINE_WIDTH = 3.0
    NUM_POINTS = 1001

    # Spatial grid over [0, ℓ] (float64 for numerical stability)
    x_vals = np.linspace(0.0, float(config.ell), NUM_POINTS, dtype=float)

    # Compute time indices near [0, n]; make them safe even if n < 4
    n = int(getattr(config, "n", 0))
    step = max(1, n // 4) if n > 0 else 1
    raw_layers = [0, step, 2 * step, 3 * step, 4 * step]
    time_layers = [min(k, n) for k in raw_layers]
    time_layers = sorted(set(time_layers))

    # =========================================================================
    # COLOR SETTINGS (Colorblind-safe: Okabe–Ito palette)
    # =========================================================================
    COLOR_APPROX = "#D55E00"  # Orange for approximation curve

    # =========================================================================
    # ENABLE LATEX-STYLED RENDERING (with safe fallback)
    # =========================================================================
    use_tex = True
    try:
        rcParams["text.usetex"] = True
        rcParams["font.family"] = "lmodern"
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
    except Exception:
        use_tex = False
        rcParams["text.usetex"] = False

    # =========================================================================
    # PREPARE OUTPUT DIRECTORY
    # =========================================================================
    config_name = getattr(config, "name", "config")
    output_path = Path(output_dir) / config_name
    output_path.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # OBTAIN THE ANSATZ CALLABLE (robustly)
    # =========================================================================
    if hasattr(approx_solver, "callable_compute_ansatz"):
        ansatz = approx_solver.callable_compute_ansatz(solution_type)
    else:
        raise AttributeError(
            "approx_solver must define `callable_compute_ansatz(solution_type)`."
        )

    # =========================================================================
    # LOOP OVER TIME LAYERS AND GENERATE PLOTS
    # =========================================================================
    saved_files: list[str] = []

    for k in time_layers:
        # ---------------- Evaluate Galerkin approximation safely ----------------
        approx_values = None

        # 1) ansatz(x, k)
        try:
            approx_values = ansatz(x_vals, k)
        except TypeError:
            pass

        # 2) ansatz(x, t=…)
        if approx_values is None:
            try:
                t_k = float(getattr(config, "t", [k])[k]) if hasattr(config, "t") else float(k)
                approx_values = ansatz(x_vals, t=t_k)
            except Exception:
                pass

        # 3) fallback API
        if approx_values is None and hasattr(approx_solver, "callable_compute_ansatz"):
            try:
                approx_values = approx_solver.callable_compute_ansatz(
                    solution_type=solution_type, k=k, x_vals=x_vals
                )
            except Exception:
                pass

        if approx_values is None:
            raise TypeError(
                "Unable to evaluate ansatz. Expected a callable from "
                "`callable_compute_ansatz(solution_type)` accepting (x, k) "
                "or (x, t=...). Please check the solver API."
            )

        # Ensure 1D shape matches x_vals
        y = np.asarray(approx_values)
        if y.ndim > 1:
            y = y.reshape(-1)
        if y.size == 1:
            y = np.full_like(x_vals, float(y))
        if y.shape[0] != x_vals.shape[0]:
            raise ValueError(
                f"x and y must align: x has shape {x_vals.shape}, "
                f"but y has shape {y.shape}. Check ansatz output."
            )

        # Safely fetch time value `t_k` for labeling (use index if missing)
        try:
            t_k = float(config.t[k])
        except Exception:
            t_k = float(k)

        # ------------------------------- Plot -----------------------------------
        plt.figure(figsize=(8, 4))

        # Build curve label (legends are removed below; label kept for completeness)
        if use_tex:
            title = rf"Approximate Solution: ${solution_type}(x, {t_k:g})$"
            xlab = rf"Spatial coordinate $x \in [0, {float(config.ell):g}]$"
        else:
            title = f"Approximate Solution: {solution_type}(x, t={t_k:g})"
            xlab = f"Spatial coordinate x ∈ [0, {float(config.ell):g}]"

        plt.plot(x_vals, y, color=COLOR_APPROX, linestyle='-', linewidth=LINE_WIDTH)
        plt.xlabel(xlab)
        plt.ylabel("Solution value")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()

        # ------------------------- Save figure (robust) --------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_pdf = output_path / f"{config_name}_solution_{solution_type}_t{k}_N{config.N}_{timestamp}.pdf"
        try:
            plt.savefig(filename_pdf)
            out_path = filename_pdf
        except Exception:
            rcParams["text.usetex"] = False
            filename_png = output_path / f"{config_name}_solution_{solution_type}_t{k}_N{config.N}_{timestamp}.png"
            plt.savefig(filename_png, dpi=600)
            out_path = filename_png
        finally:
            plt.close()

        saved_files.append(str(out_path))

    return saved_files