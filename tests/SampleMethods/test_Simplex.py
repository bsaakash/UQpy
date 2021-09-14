import numpy as np
import pytest
from UQpy.SampleMethods import Simplex

vertex = np.array([[0, 0], [0.5, 1], [1, 0]])
x = Simplex(nodes=vertex, nsamples=10, random_state=1)

vertex1 = np.array([[0], [1]])
x1 = Simplex(nodes=vertex1, nsamples=3, random_state=2)


# Unit tests
def test_samples_2d():
    """
    Test the 10 samples generated by simplex class.
    """
    tmp = (np.round(x.samples, 3) == np.array([[0.555, 0.181], [0.007, 0.007], [0.209, 0.348], [0.29, 0.282],
                                               [0.485, 0.29], [0.546, 0.204], [0.425, 0.055], [0.138, 0.055],
                                               [0.503, 0.285], [0.224, 0.3]])).all()
    assert tmp


def test_samples_1d():
    """
    Test the 3 samples generated by simplex class.
    """
    tmp = (np.round(x1.samples, 3) == np.array([[0.436], [0.026], [0.55]])).all()
    assert tmp


def test_dimension():
    """
    Test the number of nodes should be one more than the dimension.
    """
    with pytest.raises(NotImplementedError):
        Simplex(nodes=np.array([[0], [1], [2]]), nsamples=3, random_state=2)


def test_nsamples():
    """
    Test the number of nodes should be one more than the dimension.
    """
    with pytest.raises(NotImplementedError):
        Simplex(nodes=np.array([[0], [1]]), nsamples=-3, random_state=2)


def test_random_state():
    """
        Check 'random_state' is an integer or RandomState object.
    """
    with pytest.raises(TypeError):
        Simplex(nodes=np.array([[0], [1]]), nsamples=1, random_state='abc')
