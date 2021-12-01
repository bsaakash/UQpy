import pytest
from beartype.roar import BeartypeCallHintPepParamException

from UQpy.surrogates.kriging.Kriging import Kriging
from UQpy.utilities.strata.Rectangular import Rectangular
from UQpy.sampling.StratifiedSampling import StratifiedSampling
from UQpy.RunModel import RunModel
from UQpy.distributions.collection.Uniform import Uniform
import numpy as np
import shutil
from UQpy.surrogates.kriging.regression_models import Linear, Constant
from UQpy.surrogates.kriging.correlation_models import Gaussian


def test_kriging_constant_exponential():
    from UQpy.surrogates.kriging.regression_models.Constant import Constant
    from UQpy.surrogates.kriging.correlation_models.Exponential import Exponential
    marginals = [Uniform(loc=0., scale=1.), Uniform(loc=0., scale=1.)]
    strata = Rectangular(strata_number=[10, 10])
    x = StratifiedSampling(distributions=marginals, strata_object=strata,
                           samples_per_stratum_number=1, random_state=1)
    rmodel = RunModel(model_script='python_model_function.py', vec=False)
    rmodel.run(samples=x.samples)

    regression_model = Constant()
    correlation_model = Exponential()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizations_number=20, correlation_model_parameters=[1, 1])
    K.fit(samples=x.samples, values=rmodel.qoi_list)

    assert round(K.correlation_model_parameters[0], 5) == 3.99253
    assert round(K.correlation_model_parameters[1], 5) == 0.78878

    shutil.rmtree(rmodel.model_dir)

def test_kriging_linear_gaussian():
    from UQpy.surrogates.kriging.regression_models.Linear import Linear
    from UQpy.surrogates.kriging.correlation_models.Gaussian import Gaussian
    marginals = [Uniform(loc=0., scale=1.), Uniform(loc=0., scale=1.)]
    strata = Rectangular(strata_number=[10, 10])
    x = StratifiedSampling(distributions=marginals, strata_object=strata,
                           samples_per_stratum_number=1, random_state=1)
    rmodel = RunModel(model_script='python_model_function.py', vec=False)
    rmodel.run(samples=x.samples)

    regression_model = Linear()
    correlation_model = Gaussian()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizations_number=20, correlation_model_parameters=[1, 1])
    K.fit(samples=x.samples, values=rmodel.qoi_list)
    assert round(K.correlation_model_parameters[0], 2) == 52.63
    assert round(K.correlation_model_parameters[1], 2) == 3.03
    shutil.rmtree(rmodel.model_dir)


def test_kriging_quadratic_linear():
    from UQpy.surrogates.kriging.regression_models.Quadratic import Quadratic
    from UQpy.surrogates.kriging.correlation_models.Linear import Linear
    marginals = [Uniform(loc=0., scale=1.), Uniform(loc=0., scale=1.)]
    strata = Rectangular(strata_number=[10, 10])
    x = StratifiedSampling(distributions=marginals, strata_object=strata,
                           samples_per_stratum_number=1, random_state=1)
    rmodel = RunModel(model_script='python_model_function.py', vec=False)
    rmodel.run(samples=x.samples)

    regression_model = Quadratic()
    correlation_model = Linear()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizations_number=20, correlation_model_parameters=[1, 1], random_state=0)
    K.fit(samples=x.samples, values=rmodel.qoi_list)
    assert round(K.correlation_model_parameters[0], 3) == 12.294
    assert round(K.correlation_model_parameters[1], 3) == 0.6
    shutil.rmtree(rmodel.model_dir)


def test_kriging_constant_spherical():
    from UQpy.surrogates.kriging.regression_models.Constant import Constant
    from UQpy.surrogates.kriging.correlation_models.Spherical import Spherical
    marginals = [Uniform(loc=0., scale=1.), Uniform(loc=0., scale=1.)]
    strata = Rectangular(strata_number=[10, 10])
    x = StratifiedSampling(distributions=marginals, strata_object=strata,
                           samples_per_stratum_number=1, random_state=1)
    rmodel = RunModel(model_script='python_model_function.py', vec=False)
    rmodel.run(samples=x.samples)

    regression_model = Constant()
    correlation_model = Spherical()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizations_number=20, correlation_model_parameters=[1, 1])
    K.fit(samples=x.samples, values=rmodel.qoi_list)
    assert round(K.correlation_model_parameters[0], 3) == 2.315
    assert round(K.correlation_model_parameters[1], 3) == 0.536
    shutil.rmtree(rmodel.model_dir)


def test_kriging_constant_spline():
    from UQpy.surrogates.kriging.regression_models.Constant import Constant
    from UQpy.surrogates.kriging.correlation_models.Spline import Spline
    marginals = [Uniform(loc=0., scale=1.), Uniform(loc=0., scale=1.)]
    strata = Rectangular(strata_number=[10, 10])
    x = StratifiedSampling(distributions=marginals, strata_object=strata,
                           samples_per_stratum_number=1, random_state=1)
    rmodel = RunModel(model_script='python_model_function.py', vec=False)
    rmodel.run(samples=x.samples)

    regression_model = Constant()
    correlation_model = Spline()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizations_number=20, correlation_model_parameters=[1, 1])
    K.fit(samples=x.samples, values=rmodel.qoi_list)
    assert round(K.correlation_model_parameters[0], 3) == 2.104
    assert round(K.correlation_model_parameters[1], 3) == 0.387
    shutil.rmtree(rmodel.model_dir)


def test_kriging_constant_cubic():
    from UQpy.surrogates.kriging.regression_models.Constant import Constant
    from UQpy.surrogates.kriging.correlation_models.Cubic import Cubic
    marginals = [Uniform(loc=0., scale=1.), Uniform(loc=0., scale=1.)]
    strata = Rectangular(strata_number=[10, 10])
    x = StratifiedSampling(distributions=marginals, strata_object=strata,
                           samples_per_stratum_number=1, random_state=1)
    rmodel = RunModel(model_script='python_model_function.py', vec=False)
    rmodel.run(samples=x.samples)

    regression_model = Constant()
    correlation_model = Cubic()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizations_number=20, correlation_model_parameters=[1, 1], random_state=0)
    K.fit(samples=x.samples, values=rmodel.qoi_list)
    assert round(K.correlation_model_parameters[0], 3) == 0.001
    assert round(K.correlation_model_parameters[1], 3) == 338.74
    shutil.rmtree(rmodel.model_dir)


samples = np.linspace(0, 5, 20).reshape(-1, 1)
values = np.cos(samples)
krig = Kriging(regression_model=Linear(), correlation_model=Gaussian(),
               correlation_model_parameters=[0.14], optimize=False, random_state=1)
krig.fit(samples=samples, values=values, correlation_model_parameters=[0.3])

krig2 = Kriging(regression_model=Constant(), correlation_model=Gaussian(),
                correlation_model_parameters=[0.3], bounds=[[0.01, 5]],
                optimize=False, optimizations_number=100, normalize=False,
                random_state=2)
krig2.fit(samples=samples, values=values)


# Using the in-built linear regression model as a function
linear_regression_model = Kriging(regression_model=Linear(), correlation_model=Gaussian(),
                                  correlation_model_parameters=[1]).regression_model
gaussian_corrleation_model = Kriging(regression_model=Linear(), correlation_model=Gaussian(),
                                     correlation_model_parameters=[1]).correlation_model

krig3 = Kriging(regression_model=linear_regression_model, correlation_model=gaussian_corrleation_model,
                correlation_model_parameters=[1], optimize=False, normalize=False, random_state=0)
krig3.fit(samples=samples, values=values)


def test_predict():
    prediction = np.round(krig.predict([[1], [np.pi/2], [np.pi]], True), 3)
    expected_prediction = np.array([[0.54,  0., -1.], [0.,  0.,  0.]])
    assert (expected_prediction == prediction).all()


def test_predict1():
    prediction = np.round(krig2.predict([[1], [2*np.pi], [np.pi]], True), 3)
    expected_prediction = np.array([[0.54,  1.009, -1.], [0.,  0.031,  0.]])
    assert (expected_prediction == prediction).all()


def test_predict2():
    prediction = np.round(krig3.predict([[1], [np.pi/2], [np.pi]]), 3)
    expected_prediction = np.array([[0.54, -0., -1.]])
    assert (expected_prediction == prediction).all()


def test_jacobian():
    jacobian = np.round(krig.jacobian([[np.pi], [np.pi/2]]), 3)
    expected_jacobian = np.array([-0., -1.])
    assert (expected_jacobian == jacobian).all()


def test_jacobian1():
    jacobian = np.round(krig3.jacobian([[np.pi], [np.pi/2]]), 3)
    expected_jacobian = np.array([0., -1.])
    assert (expected_jacobian == jacobian).all()


def test_regress():
    from UQpy.surrogates.kriging.regression_models import Constant, Linear, Quadratic
    krig.regression_model = Constant()
    tmp = krig.regression_model.r([[0], [1]])
    tmp_test1 = (tmp[0] == np.array([[1.], [1.]])).all() and (tmp[1] == np.array([[[0.]], [[0.]]])).all()

    krig.regression_model = Linear()
    tmp = krig.regression_model.r([[0], [1]])
    tmp_test2 = (tmp[0] == (np.array([[1., 0.], [1., 1.]]))).all() and \
                (tmp[1] == np.array([[[0., 1.]], [[0., 1.]]])).all()

    krig.regression_model = Quadratic()
    tmp = krig.regression_model.r([[-1, 1], [2, -0.5]])
    tmp_test3 = (tmp[0] == np.array([[1., -1.,  1.,  1., -1.,  1.], [1.,  2., -0.5,  4., -1.,  0.25]])).all() and \
                (tmp[1] == np.array([[[0., 1., 0., -2., 1., 0.],
                                      [0., 0., 1., 0., -1., 2.]],
                                     [[0., 1., 0., 4., -0.5, 0.],
                                      [0., 0., 1., 0., 2., -1.]]])).all()

    assert tmp_test1 and tmp_test2 and tmp_test3


def test_corr():
    from UQpy.surrogates.kriging.correlation_models import Exponential, Linear, Spherical, Cubic, Spline
    krig.correlation_model = Exponential()
    rx_exponential = (np.round(krig.correlation_model.c([[0], [1], [2]], [[2]], np.array([1])), 3) ==
                      np.array([[0.135], [0.368], [1.]])).all()
    drdt_exponential = (np.round(krig.correlation_model.c([[0], [1], [2]], [[2]], np.array([1]), dt=True)[1], 3) ==
                        np.array([[[-0.271]], [[-0.368]], [[0.]]])).all()
    drdx_exponential = (np.round(krig.correlation_model.c([[0], [1], [2]], [[2]], np.array([1]), dx=True)[1], 3) ==
                        np.array([[[0.135]], [[0.368]], [[0.]]])).all()
    expon = rx_exponential and drdt_exponential and drdx_exponential

    krig.correlation_model = Linear()
    rx_linear = (np.round(krig.correlation_model.c([[0], [1], [2]], [[2]], np.array([1])), 3) ==
                 np.array([[0.], [0.], [1.]])).all()
    drdt_linear = (np.round(krig.correlation_model.c([[0.4], [0.5], [0.6]], [[0.5]], np.array([1]), dt=True)[1], 3) ==
                   np.array([[[-0.1]], [[-0.]], [[-0.1]]])).all()
    drdx_linear = (np.round(krig.correlation_model.c([[0.4], [0.5], [0.6]], [[0.5]], np.array([1]), dx=True)[1], 3) ==
                   np.array([[[1.]], [[-0.]], [[-1.]]])).all()
    linear = rx_linear and drdt_linear and drdx_linear

    krig.correlation_model = Spherical()
    rx_spherical = (np.round(krig.correlation_model.c([[0], [1], [2]], [[2]], np.array([1])), 3) ==
                    np.array([[0.], [0.], [1.]])).all()
    drdt_spherical = (np.round(krig.correlation_model.c([[0.4], [0.5], [0.6]], [[0.5]], np.array([1]), dt=True)[1], 3)
                      == np.array([[[-0.148]], [[-0.]], [[-0.148]]])).all()
    drdx_spherical = (np.round(krig.correlation_model.c([[0.4], [0.5], [0.6]], [[0.5]], np.array([1]), dx=True)[1], 3)
                      == np.array([[[1.485]], [[-0.]], [[-1.485]]])).all()
    spherical = rx_spherical and drdt_spherical and drdx_spherical

    krig.correlation_model = Cubic()
    rx_cubic = (np.round(krig.correlation_model.c([[0.2], [0.5], [1]], [[0.5]], np.array([1])), 3) ==
                np.array([[0.784], [1.], [0.5]])).all()
    drdt_cubic = (np.round(krig.correlation_model.c([[0.4], [0.5], [0.6]], [[0.5]], np.array([1]), dt=True)[1], 3) ==
                  np.array([[[-0.054]], [[0.]], [[-0.054]]])).all()
    drdx_cubic = (np.round(krig.correlation_model.c([[0.4], [0.5], [0.6]], [[0.5]], np.array([1]), dx=True)[1], 3) ==
                  np.array([[[0.54]], [[0.]], [[-0.54]]])).all()
    cubic = rx_cubic and drdt_cubic and drdx_cubic

    krig.correlation_model = Spline()
    rx_spline = (np.round(krig.correlation_model.c([[0.2], [0.5], [1]], [[0.5]], np.array([1])), 3) ==
                 np.array([[0.429], [1.], [0.156]])).all()
    drdt_spline = (np.round(krig.correlation_model.c([[0.4], [0.5], [0.6]], [[0.5]], np.array([1]), dt=True)[1], 3) ==
                   np.array([[[-0.21]], [[0.]], [[-0.21]]])).all()
    drdx_spline = (np.round(krig.correlation_model.c([[0.4], [0.5], [0.6]], [[0.5]], np.array([1]), dx=True)[1], 3) ==
                   np.array([[[2.1]], [[0.]], [[-2.1]]])).all()
    spline = rx_spline and drdt_spline and drdx_spline

    assert expon and linear and spherical and cubic and spline


def test_reg_model():
    """
        Raises an error if reg_model is not callable or a string of an in-built model.
    """
    with pytest.raises(BeartypeCallHintPepParamException):
        Kriging(regression_model='A', correlation_model=Gaussian(), correlation_model_parameters=[1])


def test_corr_model():
    """
        Raises an error if corr_model is not callable or a string of an in-built model.
    """
    with pytest.raises(BeartypeCallHintPepParamException):
        Kriging(regression_model=Linear(), correlation_model='A', correlation_model_parameters=[1])


def test_corr_model_params():
    """
        Raises an error if corr_model_params is not defined.
    """
    with pytest.raises(TypeError):
        Kriging(regression_model=Linear(), correlation_model=Gaussian(), bounds=[[0.01, 5]],
                optimizations_number=100, random_state=1)


def test_optimizer():
    """
        Raises an error if corr_model_params is not defined.
    """
    with pytest.raises(TypeError):
        Kriging(regression_model=Linear(), correlation_model=Gaussian(),
                correlation_model_parameters=[1], optimizer='A')


def test_random_state():
    """
        Raises an error if type of random_state is not correct.
    """
    with pytest.raises(BeartypeCallHintPepParamException):
        Kriging(regression_model=Linear(), correlation_model=Gaussian(),
                correlation_model_parameters=[1], random_state='A')


def test_loglikelihood():
    prediction = np.round(krig3.log_likelihood(np.array([1.5]), krig3.correlation_model, np.array([[1], [2]]),
                                               np.array([[1], [1]]), np.array([[1], [2]]))[0], 3)
    expected_prediction = 1.679
    assert (expected_prediction == prediction).all()


def test_loglikelihood_derivative():
    prediction = np.round(krig3.log_likelihood(np.array([1.5]), krig3.correlation_model, np.array([[1], [2]]),
                                               np.array([[1], [1]]), np.array([[1], [2]]))[1], 3)
    expected_prediction = np.array([-0.235])
    assert (expected_prediction == prediction).all()
