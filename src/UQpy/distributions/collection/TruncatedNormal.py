from typing import Union

import scipy.stats as stats
from beartype import beartype

from UQpy.distributions.baseclass import DistributionContinuous1D


class TruncatedNormal(DistributionContinuous1D):

    @beartype
    def __init__(
        self,
        a: Union[None, float, int],
        b: Union[None, float, int],
        loc: Union[None, float, int] = 0.0,
        scale: Union[None, float, int] = 1.0,
    ):
        """

        :param float a: shape parameter
        :param float b: shape parameter
        :param float loc: location parameter
        :param float scale: scale parameter
        """
        super().__init__(
            a=a,
            b=b,
            loc=loc,
            scale=scale,
            ordered_parameters=("a", "b", "loc", "scale"),
        )
        self._construct_from_scipy(scipy_name=stats.truncnorm)
