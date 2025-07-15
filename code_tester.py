# # ==========================================================
# # Module Imports
# # ==========================================================

# # NumPy: Used for efficient numerical operations and vectorized computations
# import numpy as np

# # leggauss: Generates Gauss–Legendre quadrature nodes and weights for interval [-1, 1]
# from numpy.polynomial.legendre import leggauss

# # Project-specific utilities and configuration (assumed to support this numerical framework)
# import utils.auxiliary as aux
# import utils.config as cfg


# # ==========================================================
# # Function: gauss_legendre_integral
# # Purpose : Numerically integrate a function over [a, b] using Gauss–Legendre quadrature
# # ==========================================================
# def gauss_legendre_integral(f, a, b, n_gauss):
#     """
#     Compute the Gauss–Legendre quadrature of function `f` over the interval [a, b].

#     Parameters:
#         f       : callable
#                   Function to integrate. Should ideally support NumPy vectorized input.
#         a       : float
#                   Lower limit of integration.
#         b       : float
#                   Upper limit of integration.
#         n_gauss : int
#                   Number of Gauss–Legendre nodes to use.

#     Returns:
#         float : Approximate integral of f from a to b.
#     """

#     # --------------------------------------------------
#     # Generate nodes and weights on the canonical interval [-1, 1]
#     # --------------------------------------------------
#     nodes, weights = leggauss(n_gauss)

#     # --------------------------------------------------
#     # Transform nodes to the interval [a, b] using affine map
#     # --------------------------------------------------
#     mid = 0.5 * (a + b)              # Midpoint of integration interval
#     half_len = 0.5 * (b - a)         # Half the interval length
#     x_mapped = mid + half_len * nodes  # Transformed nodes for [a, b]

#     # --------------------------------------------------
#     # Evaluate function at transformed nodes
#     # --------------------------------------------------
#     try:
#         # Prefer vectorized evaluation
#         f_vals = np.asarray(f(x_mapped))

#         # Sanity check on output shape
#         if f_vals.shape != x_mapped.shape:
#             raise ValueError("Function output shape mismatch.")

#     except Exception:
#         # Fallback to scalar evaluation (non-vectorized function)
#         f_vals = np.array([f(xi) for xi in x_mapped])

#     # --------------------------------------------------
#     # Return the weighted sum scaled by interval length
#     # --------------------------------------------------
#     return half_len * np.dot(weights, f_vals)


# # ==========================================================
# # Function: adaptive_gauss_legendre_integrator
# # Purpose : Perform adaptive integration over [0, ell] using:
# #           1. Increasing Gauss–Legendre node count,
# #           2. Subinterval refinement when needed.
# # ==========================================================
# def adaptive_gauss_legendre_integrator(
#     f: callable,
#     ell: float,
#     tol: float = 1e-6,
#     min_dx: float = 1 / 128.0,
#     n_gauss: int = 5,
#     max_gauss: int = 50
# ) -> tuple[float, float, int, int]:
#     """
#     Approximate the integral of `f` over the interval [0, ell] using adaptive Gauss–Legendre quadrature.

#     Strategy:
#         - Start with low node count on full interval and try to converge.
#         - If not successful, split the interval into smaller parts.
#         - In each subinterval, adaptively increase node count until convergence or limits are reached.

#     Parameters:
#         f         : callable
#                     Function to integrate. Must accept float input.
#         ell       : float
#                     Upper limit of integration interval [0, ell]. Must be ≥ 0.
#         tol       : float, optional
#                     Absolute convergence tolerance. Default is 1e-6.
#         min_dx    : float, optional
#                     Minimum subinterval width before halting refinement. Default is 1/128.
#         n_gauss   : int, optional
#                     Initial number of Gauss nodes to try. Default is 5.
#         max_gauss : int, optional
#                     Maximum allowed Gauss nodes per interval. Default is 50.

#     Returns:
#         tuple:
#             - float : Estimated integral value
#             - float : Estimated absolute error
#             - int   : Number of interval halving iterations
#             - int   : Maximum number of Gauss nodes used
#     """

#     # ------------------------------------------
#     # Step 1: Validate input domain
#     # ------------------------------------------
#     if ell < 0:
#         raise ValueError("Parameter 'ell' must be non-negative.")
#     if ell == 0:
#         return 0.0, 0.0, 0, 0  # Trivial integral

#     # ------------------------------------------
#     # Step 2: Attempt full interval integration with increasing node count
#     # ------------------------------------------
#     initial_n_gauss = n_gauss             # Preserve initial value for subinterval reuse
#     max_nodes_used = n_gauss              # Track max Gauss nodes used overall
#     converged = False                     # Flag for global convergence

#     integral_prev = gauss_legendre_integral(f, 0.0, ell, n_gauss)

#     while n_gauss + 5 <= max_gauss:
#         n_gauss += 5
#         integral_curr = gauss_legendre_integral(f, 0.0, ell, n_gauss)

#         # Track maximum nodes used
#         max_nodes_used = min(max(max_nodes_used, n_gauss), max_gauss)

#         # Check convergence based on absolute difference
#         if np.abs(integral_curr - integral_prev) < tol:
#             estimated_error = np.abs(integral_curr - integral_prev)
#             return integral_curr, estimated_error, 0, max_nodes_used

#         integral_prev = integral_curr

#     # ------------------------------------------
#     # Step 3: Adaptive refinement by interval halving
#     # ------------------------------------------
#     counter = 0            # Number of halving iterations
#     prev_total = None      # Store previous estimate for convergence check

#     while not converged:
#         counter += 1
#         n_intervals = 2 ** counter
#         dx = ell / n_intervals

#         # Stop refinement if interval width is too small
#         if dx < min_dx:
#             break

#         total_integral = 0.0
#         converged = True  # Assume convergence unless proven otherwise

#         for i in range(n_intervals):
#             a = i * dx
#             b = (i + 1) * dx

#             n_gauss_local = initial_n_gauss
#             integral_prev = gauss_legendre_integral(f, a, b, n_gauss_local)
#             local_converged = False

#             # Try to converge in this subinterval
#             while n_gauss_local + 5 <= max_gauss:
#                 n_gauss_local += 5
#                 integral_curr = gauss_legendre_integral(f, a, b, n_gauss_local)

#                 if np.abs(integral_curr - integral_prev) < tol:
#                     total_integral += integral_curr
#                     local_converged = True
#                     break

#                 integral_prev = integral_curr

#             if not local_converged:
#                 # Accept last estimate even if not converged
#                 total_integral += integral_curr
#                 converged = False

#             max_nodes_used = min(max(max_nodes_used, n_gauss_local), max_gauss)

#         # ------------------------------------------
#         # Step 4: Global convergence verification
#         # ------------------------------------------
#         if converged:
#             if prev_total is not None and np.abs(total_integral - prev_total) < tol:
#                 estimated_error = np.abs(total_integral - prev_total)
#                 return total_integral, estimated_error, counter, max_nodes_used

#             # If first converged estimate, store and continue
#             prev_total = total_integral

#     # ------------------------------------------
#     # Step 5: Return best estimate if convergence not reached
#     # ------------------------------------------
#     estimated_error = np.abs(total_integral - prev_total) if prev_total is not None else float('inf')
#     return total_integral, estimated_error, counter, max_nodes_used


# m = 41
# f = lambda x: (m * np.pi / cfg.ell) * np.sin(m * np.pi * x / cfg.ell)

# integral_global = gauss_legendre_integral(f=f, a=0.0, b=cfg.ell, n_gauss=50)

# result, error, steps, max_nodes = adaptive_gauss_legendre_integrator(f=f, ell=cfg.ell)
# print(f"Result = {result}, Error = {error}, Halving steps = {steps}, Max nodes = {max_nodes}")


# Standard Summation (without Kahan's correction)
def standard_sum(arr):
    total = 0.0
    for number in arr:
        total += number
    return total

# Kahan Summation (with error compensation)
def kahan_sum(arr):
    total = 0.0  # Running total
    c = 0.0  # Compensation for lost low-order bits
    
    for number in arr:
        y = number - c  # Subtract the compensation from the current number
        t = total + y   # Add the corrected number to the total
        c = (t - total) - y  # Compute the new compensation
        total = t  # Update the total
    
    return total

# Kahan-Babuška-Neumaier Summation (with improved error compensation)
def kahan_babushka_neumaier_sum(arr):
    total = 0.0  # Running total
    correction = 0.0  # Error correction term
    
    for number in arr:
        # Calculate the temporary sum
        temp = total + number
        # Calculate the error term
        if abs(total) >= abs(number):
            correction += (total - temp) + number
        else:
            correction += (number - temp) + total
        # Update the total sum with the corrected value
        total = temp
    
    return total + correction

# Example list with large and small numbers
numbers = [1.0e16, 1, 1.0, 1.0e-10, -1.0e16]

# Calculate the sums using all three methods
standard_result = standard_sum(numbers)
kahan_result = kahan_sum(numbers)
kahan_babushka_result = kahan_babushka_neumaier_sum(numbers)

# Print the results for comparison
print("Standard Sum:", standard_result)
print("Kahan Sum:", kahan_result)
print("Kahan-Babuška-Neumaier Sum:", kahan_babushka_result)

# Check the differences between the methods
difference_standard_kahan = abs(standard_result - kahan_result)
difference_standard_kbn = abs(standard_result - kahan_babushka_result)
difference_kahan_kbn = abs(kahan_result - kahan_babushka_result)

print("Difference between Standard and Kahan Sum:", difference_standard_kahan)
print("Difference between Standard and Kahan-Babuška-Neumaier Sum:", difference_standard_kbn)
print("Difference between Kahan and Kahan-Babuška-Neumaier Sum:", difference_kahan_kbn)