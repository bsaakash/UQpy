import numpy as np

from UQpy.dimension_reduction.distances.grassmanian.baseclass.RiemannianDistance import (
    RiemannianDistance,
)


class BinetCauchy(RiemannianDistance):
    def compute_distance(self, xi, xj) -> float:

        RiemannianDistance.check_points(xi, xj)

        r = np.dot(xi.T, xj)
        (ui, si, vi) = np.linalg.svd(r, full_matrices=True)
        si[np.where(si > 1)] = 1.0
        theta = np.arccos(si)

        cos_sq = np.cos(theta) ** 2
        d = np.sqrt(1 - np.prod(cos_sq))

        return d
