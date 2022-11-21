"""
Created on Sat Oct 29 22:08:46 2022 @author: john.obrecht
"""

import cv2
import json
import numpy as np
import pandas as pd

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
    # f = open(file, 'r')
    # tree = np.array(json.loads('[' + f.read().replace('\n', ',') + ']'), dtype='float')

    tree_0 = pd.read_csv(file)
    tree = tree_0[['x', 'y', 'z']].values
    
    # # Scale Z
    # tree[:, 2] -= np.min(tree[:, 2])
    # z_sc = np.max(tree[:, 2])
    # tree[:, 2] /= z_sc
    
    # # Scale X & Y
    # tree[:, 0] -= np.mean(tree[:, 0])  # Center the x-direction
    # tree[:, 1] -= np.mean(tree[:, 1])  # Center the y-direction
    # tree[:, 0] /= z_sc  # Scale x by the same as z was scaled
    # tree[:, 1] /= z_sc  # Scale y by the same as z was scaled
    
    # Sort array order by column 2
    # tree = tree[np.argsort(tree[:, 2])]

    return tree


def get_x_y(img, gx, idx_x, idx_y, thr):
    g0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g0 = g0.astype(float)
    g = g0 - gx
    g[g<thr] = 0
    sum_x = np.sum(g, axis=0)
    sum_y = np.sum(g, axis=1)
    sum_sum = np.sum(sum_y) + 1E-6
    x = np.dot(sum_x, idx_y) / sum_sum
    y = np.dot(sum_y, idx_x) / sum_sum
    return x, y


def convert_rgba_to_rgb(seq):
    a = seq[:, 3, :]
    num_pts, _, num_frames = np.shape(seq)
    rgb = np.zeros([num_pts, 3, num_frames])
    for i in range(3):
        rgb[:, i, :] = 255 * a * seq[:, i, :]
    return rgb
