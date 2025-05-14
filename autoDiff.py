# Import core JAX library for automatic differentiation and JIT compilation
import jax

# Enable 64-bit floating-point precision for more accurate numerical computations
jax.config.update("jax_enable_x64", True)

# -----------------------------------------------------------------------------
# Automatic Differentiation: First-Order Derivative with respect to x or t
# -----------------------------------------------------------------------------

class AutoDiff:
    """
    A utility class to compute first-order and second-order partial derivatives of scalar functions
    with respect to x or t using JAX's automatic differentiation.
    """

    def __init__(self, func):
        """
        Initialize the AutoDiff object with a function to differentiate.

        Args:
            func (callable): A function of (x, t), e.g., u(x, t) or v(x, t)
        """
        self.func = func

    def first_deriv(self, x: float, t: float, var: str = 'x') -> float:
        """
        Compute the first-order partial derivative of the function with respect to x or t.

        Args:
            x (float): Spatial variable
            t (float): Time variable
            var (str): Variable to differentiate with respect to ('x' or 't')

        Returns:
            float: The derivative of func with respect to the chosen variable (∂/∂x or ∂/∂t)
        """
        if var == 'x':
            # Compute the first-order derivative w.r.t. x
            return jax.grad(lambda x_val: self.func(x_val, t))(x)
        elif var == 't':
            # Compute the first-order derivative w.r.t. t
            return jax.grad(lambda t_val: self.func(x, t_val))(t)
        else:
            raise ValueError("Argument 'var' must be either 'x' or 't'.")

    def second_deriv(self, x: float, t: float, var: str = 'x') -> float:
        """
        Compute the second-order partial derivative of the function with respect to x or t.

        Args:
            x (float): Spatial variable
            t (float): Time variable
            var (str): Variable to differentiate with respect to ('x' or 't')

        Returns:
            float: The second-order partial derivative of func with respect to the chosen variable (∂²/∂x² or ∂²/∂t²)
        """
        if var == 'x':
            # Compute the second-order derivative w.r.t. x
            return jax.grad(jax.grad(lambda x_val: self.func(x_val, t)))(x)
        elif var == 't':
            # Compute the second-order derivative w.r.t. t
            return jax.grad(jax.grad(lambda t_val: self.func(x, t_val)))(t)
        else:
            raise ValueError("Argument 'var' must be either 'x' or 't'.")