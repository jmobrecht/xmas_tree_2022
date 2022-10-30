"""
Created on Sat Oct 29 23:07:53 2022 @author: john.obrecht
"""

import numpy as np
from matplotlib import cm

# Rainbow
def rainbow(tree, num_pts, num_frames):
    color_map = cm.get_cmap('hsv', num_pts)
    seq = np.zeros([num_pts, 4, num_frames])
    seq[:, :, 0] = color_map(np.linspace(0, 1, num_pts))
    for i in range(1, num_frames):
        seq[:, :, i] = np.roll(seq[:, :, i-1], 1, axis=0)
    return seq

# Slice
def moving_slice(tree, num_pts, num_frames):
    h0 = np.linspace(1, 0, num_frames)
    def slice(z, i):
        return np.exp(-(z - h0[i])**2 / (0.01 * np.max(tree[:, 2]))**2)
    seq = np.ones([num_pts, 4, num_frames])
    for i in range(1, num_frames):
        seq[:, 3, i] = np.round(slice(tree[:, 2], i))
    return seq

# Rain
def falling_rain(tree, num_pts, num_frames):
    def dist(x, y, z, i):
        return np.sqrt( (x - x0[i])**2 + (y - y0[i])**2 + (z - z0[i])**2)
    def rain(x, y, z, i):
        return np.exp(-(dist(x, y, z, i))**2 / (0.05)**2)
    # Radial scale
    r_sc = np.mean([-np.min(tree[:, 0]), np.max(tree[:, 0]), -np.min(tree[:, 1]), np.max(tree[:, 1])])
    z0 = np.linspace(1, 0, num_frames)
    y0 = np.linspace(0, 0, num_frames)
    x0 = np.linspace(0, r_sc, num_frames)
    seq = np.ones([num_pts, 4, num_frames])
    for i in range(1, num_frames):
        seq[:, 3, i] = np.round(rain(tree[:, 0], tree[:, 1], tree[:, 2], i))
    return seq