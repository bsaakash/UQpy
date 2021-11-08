import logging
import numpy as np
from UQpy.surrogates.polynomial_chaos_new.polynomials.baseclass import Polynomials
from UQpy.surrogates.polynomial_chaos_new.regressions.baseclass.Regression import Regression


class RidgeRegression(Regression):

    def __init__(self, polynomial_basis: Polynomials, learning_rate: float = 0.01, iterations: int = 1000,
                 penalty: float = 1):
        """
        Class to calculate the polynomial_chaos coefficients with the Ridge regression method.

        :param polynomial_basis: Object from the 'Polynomial' class
        :param learning_rate: Size of steps for the gradient descent.
        :param iterations: Number of iterations of the optimization algorithm.
        :param penalty: Penalty parameter controls the strength of regularization. When it
         is close to zero, then the ridge regression converges to the linear
         regression, while when it goes to infinity, polynomial_chaos coefficients
         converge to zero.
        """
        super().__init__(polynomial_basis)
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.penalty = penalty
        self.logger = logging.getLogger(__name__)

    def run(self, x, y):
        """
        Implements the LASSO method to compute the polynomial_chaos coefficients.

        :param x: `ndarray` containing the training points (samples).
        :param y: `ndarray` containing the model evaluations (labels) at the training points.
        :return: Weights (polynomial_chaos coefficients)  and Bias of the regressor
        """
        xx = self.polynomial_basis.evaluate_basis(x)
        m, n = xx.shape

        if y.ndim == 1 or y.shape[1] == 1:
            y = y.reshape(-1, 1)
            w = np.zeros(n).reshape(-1, 1)
            b = 0

            for _ in range(self.iterations):
                y_pred = (xx.dot(w) + b).reshape(-1, 1)

                dw = (-(2 * xx.T.dot(y - y_pred)) + (2 * self.penalty * w)) / m
                db = -2 * np.sum(y - y_pred) / m

                w = w - self.learning_rate * dw
                b = b - self.learning_rate * db

        else:
            n_out_dim = y.shape[1]
            w = np.zeros((n, n_out_dim))
            b = np.zeros(n_out_dim).reshape(1, -1)

            for _ in range(self.iterations):
                y_pred = xx.dot(w) + b

                dw = (-(2 * xx.T.dot(y - y_pred)) + (2 * self.penalty * w)) / m
                db = -2 * np.sum((y - y_pred), axis=0).reshape(1, -1) / m

                w = w - self.learning_rate * dw
                b = b - self.learning_rate * db

        return w, b