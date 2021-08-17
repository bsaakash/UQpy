
import scipy.stats as stats
from beartype import beartype
from UQpy.utilities.ValidationTypes import *
from UQpy.utilities.Utilities import process_random_state

class SimplexSampling:
    """
    Generate uniform random samples inside an n-dimensional simplex.


    **Inputs:**

    * **nodes** (`ndarray` or `list`):
        The vertices of the simplex.

    * **nsamples** (`int`):
        The number of samples to be generated inside the simplex.

        If `nsamples` is provided when the object is defined, the ``run`` method will be called automatically. If
        `nsamples` is not provided when the object is defined, the user must invoke the ``run`` method and specify
        `nsamples`.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    **Attributes:**

    * **samples** (`ndarray`):
        New random samples distributed uniformly inside the simplex.

    **Methods:**

    """
    @beartype
    def __init__(self,
                 nodes=None,
                 samples_number: PositiveInteger = None,
                 random_state: RandomStateType = None):
        self.nodes = np.atleast_2d(nodes)
        self.samples_number = samples_number

        if self.nodes.shape[0] != self.nodes.shape[1] + 1:
            raise NotImplementedError("UQpy: Size of simplex (nodes) is not consistent.")

        self.random_state = process_random_state(random_state)

        if samples_number is not None:
            self.samples = self.run(samples_number=samples_number)

    @beartype
    def run(self, samples_number: PositiveInteger):
        """
        Execute the random sampling in the ``Simplex`` class.

        The ``run`` method is the function that performs random sampling in the ``Simplex`` class. If `nsamples` is
        provided called when the ``Simplex`` object is defined, the ``run`` method is automatically. The user may also
        call the ``run`` method directly to generate samples. The ``run`` method of the ``Simplex`` class can be invoked
        many times and each time the generated samples are appended to the existing samples.

        **Input:**

        * **nsamples** (`int`):
            Number of samples to be generated inside the simplex.

            If the ``run`` method is invoked multiple times, the newly generated samples will be appended to the
            existing samples.

        **Output/Return:**

        The ``run`` method has no returns, although it creates and/or appends the `samples` attribute of the ``Simplex``
        class.

        """
        self.samples_number = samples_number
        dimension = self.nodes.shape[1]
        if dimension > 1:
            sample = np.zeros([self.samples_number, dimension])
            for i in range(self.samples_number):
                r = np.zeros([dimension])
                ad = np.zeros(shape=(dimension, len(self.nodes)))
                for j in range(dimension):
                    b_ = list()
                    for k in range(1, len(self.nodes)):
                        ai = self.nodes[k, j] - self.nodes[k - 1, j]
                        b_.append(ai)
                    ad[j] = np.hstack((self.nodes[0, j], b_))
                    r[j] = stats.uniform.rvs(loc=0, scale=1, random_state=self.random_state) ** (1 / (dimension - j))
                d = np.cumprod(r)
                r_ = np.hstack((1, d))
                sample[i, :] = np.dot(ad, r_)
        else:
            a = min(self.nodes)
            b = max(self.nodes)
            sample = a + (b - a) * stats.uniform.rvs(size=[self.samples_number, dimension], random_state=self.random_state)
        return sample

    def __copy__(self):
        new = self.__class__(nodes=self.nodes,
                             random_state=self.random_state)
        new.__dict__.update(self.__dict__)

        return new