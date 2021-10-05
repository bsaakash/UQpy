import logging
import numpy as np
from UQpy.surrogates.polynomial_chaos.polynomials.baseclass import Polynomials
from UQpy.surrogates.polynomial_chaos.regressions.baseclass.Regression import Regression


class RidgeRegression(Regression):
    """
     Class to calculate the polynomial_chaos coefficients with the Ridge regression method.

     **Inputs:**

     * **poly_object** ('class'):
        Object from the 'Polynomial' class

     **Methods:**
     """

    def __init__(self,
                 polynomials: Polynomials,
                 learning_rate: float = 0.01,
                 iterations: int = 1000,
                 penalty: float = 1):
        self.polynomials = polynomials
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.penalty = penalty
        self.logger = logging.getLogger(__name__)

    def run(self, x, y):
        """
        Implements the LASSO method to compute the polynomial_chaos coefficients.

        **Inputs:**

        * **poly_object** (`object`):
            Polynomial object.

        * **learning_rate** (`float`):
            Size of steps for the gradient descent.

        * **iterations** (`int`):
            Number of iterations of the optimization algorithm.

        * **penalty** (`float`):
            Penalty parameter controls the strength of regularization. When it
            is close to zero, then the ridge regression converges to the linear
            regression, while when it goes to infinity, polynomial_chaos coefficients
            converge to zero.

        **Outputs:**

        * **w** (`ndarray`):
            Returns the weights (polynomial_chaos coefficients) of the regressor.

        * **b** (`float`):
            Returns the bias of the regressor.

        """

        xx = self.polynomials.evaluate(x)
        m, n = xx.shape

        if y.ndim == 1 or y.shape[1] == 1:
            y = y.reshape(-1, 1)
            w = np.zeros(n).reshape(-1, 1)
            b = 0

            for _ in range(self.iterations):
                y_pred = (xx.dot(w) + b).reshape(-1, 1)

                dw = (-(2 * xx.T.dot(y - y_pred)) + (2 * self.penalty * w)) / m
                db = - 2 * np.sum(y - y_pred) / m

                w = w - self.learning_rate * dw
                b = b - self.learning_rate * db

        else:
            n_out_dim = y.shape[1]
            w = np.zeros((n, n_out_dim))
            b = np.zeros(n_out_dim).reshape(1, -1)

            for _ in range(self.iterations):
                y_pred = (xx.dot(w) + b)

                dw = (-(2 * xx.T.dot(y - y_pred)) + (2 * self.penalty * w)) / m
                db = - 2 * np.sum((y - y_pred), axis=0).reshape(1, -1) / m

                w = w - self.learning_rate * dw
                b = b - self.learning_rate * db

        return w, b