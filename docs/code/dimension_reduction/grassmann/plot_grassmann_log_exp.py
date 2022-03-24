"""

Mappings to & from tangent space
==================================

This example shows how to use the UQpy Grassmann class to use the logarithmic map and the exponential map
"""

#%% md
#
# Import the necessary libraries. Here we import standard libraries such as numpy and matplotlib, but also need to
# import the Grassmann class from UQpy implemented in the DimensionReduction module.

#%%

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from UQpy.dimension_reduction.grassmann_manifold.projections.SvdProjection import SvdProjection
from UQpy.dimension_reduction import Grassmann
import sys

#%% md
#
# Generate four random matrices with reduced rank corresponding to the different samples. The samples are stored in
# `matrices`.

#%%

D1 = 6
r0 = 2  # rank sample 0
r1 = 3  # rank sample 1
r2 = 4  # rank sample 2
r3 = 3  # rank sample 2

np.random.seed(1111)  # For reproducibility.
# Solutions: original space.
Sol0 = np.dot(np.random.rand(D1, r0), np.random.rand(r0, D1))
Sol1 = np.dot(np.random.rand(D1, r1), np.random.rand(r1, D1))
Sol2 = np.dot(np.random.rand(D1, r2), np.random.rand(r2, D1))
Sol3 = np.dot(np.random.rand(D1, r3), np.random.rand(r3, D1))

# Creating a list of matrices.
matrices = [Sol0, Sol1, Sol2, Sol3]

# Plot the matrices
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
ax1.title.set_text('Matrix 0')
ax1.imshow(Sol0)
ax2.title.set_text('Matrix 1')
ax2.imshow(Sol1)
ax3.title.set_text('Matrix 2')
ax3.imshow(Sol2)
ax4.title.set_text('Matrix 3')
ax4.imshow(Sol3)
plt.show()

#%% md
#
# Instantiate the UQpy class Grassmann.

#%%

manifold_projection = SvdProjection(matrices, p="max")

# Plot the points on the Grassmann manifold defined by the left singular eigenvectors.
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
ax1.title.set_text('Matrix 0')
ax1.imshow(manifold_projection.psi[0].data)
ax2.title.set_text('Matrix 1')
ax2.imshow(manifold_projection.psi[0].data)
ax3.title.set_text('Matrix 2')
ax3.imshow(manifold_projection.psi[0].data)
ax4.title.set_text('Matrix 3')
ax4.imshow(manifold_projection.psi[0].data)
plt.show()

#%% md
#
# Project $\Psi$, the left singular eigenvectors, on the tangent space centered at $\Psi_0$.

#%%

points_tangent = Grassmann.log_map(grassmann_points=manifold_projection.psi,
                                   reference_point=manifold_projection.psi[0])

print(points_tangent[0])
print(points_tangent[1])
print(points_tangent[2])
print(points_tangent[3])

# Plot the matrices
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
ax1.title.set_text('Matrix 0')
ax1.imshow(points_tangent[0])
ax2.title.set_text('Matrix 1')
ax2.imshow(points_tangent[1])
ax3.title.set_text('Matrix 2')
ax3.imshow(points_tangent[2])
ax4.title.set_text('Matrix 3')
ax4.imshow(points_tangent[3])
plt.show()

#%% md
#
# Map the points back to the Grassmann manifold.

#%%

points_grassmann = Grassmann.exp_map(tangent_points=points_tangent,
                                     reference_point=manifold_projection.psi[0])

# Plot the matrices
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
ax1.title.set_text('Matrix 0')
ax1.imshow(points_grassmann[0].data)
ax2.title.set_text('Matrix 1')
ax2.imshow(points_grassmann[1].data)
ax3.title.set_text('Matrix 2')
ax3.imshow(points_grassmann[2].data)
ax4.title.set_text('Matrix 3')
ax4.imshow(points_grassmann[3].data)
plt.show()