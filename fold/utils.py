"""
Created on Sat Oct 29 22:08:46 2022 @author: john.obrecht
"""

import json
import numpy as np

# Set 3D Axes Equal - function
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# Get tree coordinates from a file
def get_tree_coords(file):
    f = open(file, 'r')
    tree = np.array(json.loads('[' + f.read().replace('\n', ',') + ']'), dtype='float')
    
    # Scale Z
    tree[:, 2] -= np.min(tree[:, 2])
    z_sc = np.max(tree[:, 2])
    tree[:, 2] /= z_sc
    
    # Scale X & Y
    tree[:, 0] -= np.mean(tree[:, 0])  # Center the x-direction
    tree[:, 1] -= np.mean(tree[:, 1])  # Center the y-direction
    tree[:, 0] /= z_sc  # Scale x by the same as z was scaled
    tree[:, 1] /= z_sc  # Scale y by the same as z was scaled
    
    # Sort array order by column 2
    tree = tree[np.argsort(tree[:, 2])]

    return tree
