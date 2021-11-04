from UQpy.Distributions import *
from UQpy.Surrogates import *
import numpy as np

np.random.seed(1)
max_degree, n_samples = 2, 10
dist = Uniform(loc=0, scale=10)
pce = PolyChaosExp(dist)

def func(x):
    return x * np.sin(x) / 10

x = dist.rvs(n_samples)
x_test = dist.rvs(n_samples)
y = func(x)

def poly_td_func(pce, max_degree):
    construct_td_basis(pce, max_degree)
    p = pce.poly_basis
    return p

def poly_tp_func(pce, max_degree):
    construct_tp_basis(pce, max_degree)
    p = pce.poly_basis
    return p

def pce_coeff(pce, x, y):
    fit_lstsq(pce, x, y)
    return pce.coefficients

def pce_predict(pce,x):
    return pce.predict(x)

# Unit tests
def test_1():
    """
    Test td basis
    """
    assert round(poly_td_func(pce, max_degree)[1].evaluate(x)[0], 4) == -0.2874

def test_2():
    """
    Test tp basis
    """
    assert round(poly_tp_func(pce, max_degree)[1].evaluate(x)[0], 4) == -0.2874

def test_3():
    """
    Test PCE coefficients
    """
    assert round(pce_coeff(pce, x, y)[0][0], 4) == 0.2175

def test_4():
    """
    Test PCE prediction
    """
    y_test = pce_predict(pce, x_test)
    assert round(y_test[0][0], 4) == -0.1607

def test_5():
    """
    Test Sobol indices
    """
    assert round(pce_sobol_first(pce)[0][0], 3) == 1.0

def test_6():
    """
    Test Sobol indices
    """
    assert round(pce_sobol_total(pce)[0][0], 3) == 1.0

def test_7():
    """
    PCE mean
    """
    assert round(pce_mean(pce)[0], 3) == 0.218

def test_8():
    """
    PCE variance
    """
    assert round(pce_variance(pce)[0], 3) == 0.185

