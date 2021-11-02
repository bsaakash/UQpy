import numpy as np
import scipy

from UQpy.dimension_reduction.grassmann_manifold.interpolations.baseclass.InterpolationMethod import (
    InterpolationMethod,
)


class LinearInterpolation(InterpolationMethod):
    def interpolate(self, coordinates, samples, point):

        """
        Interpolate a point using the linear interpolation.

        For the linear interpolation the user are asked to provide the data points, the coordinates of the data points,
        and the coordinate of the point to be interpolated.

        **Input:**

        * **coordinates** (`ndarray`)
            Coordinates of the input data points.

        * **samples** (`ndarray`)
            Matrices corresponding to the points on the grassmann_manifold manifold.

        * **point** (`ndarray`)
            Coordinates of the point to be interpolated.

        **Output/Returns:**

        * **interp_point** (`ndarray`)
            Interpolated point.

        """

        if not isinstance(coordinates, list) and not isinstance(
            coordinates, np.ndarray
        ):
            raise TypeError("UQpy: `coordinates` must be either list or ndarray.")
        else:
            coordinates = np.array(coordinates)

        if not isinstance(samples, list) and not isinstance(samples, np.ndarray):
            raise TypeError("UQpy: `samples` must be either list or ndarray.")
        else:
            samples = np.array(samples)

        if not isinstance(point, list) and not isinstance(point, np.ndarray):
            raise TypeError("UQpy: `point` must be either list or ndarray.")
        else:
            point = np.array(point)

        myInterpolator = scipy.interpolate.LinearNDInterpolator(coordinates, samples)
        interp_point = myInterpolator(point)
        interp_point = interp_point[0]

        return interp_point