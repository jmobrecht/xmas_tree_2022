"""
Created on Fri Nov  4 13:07:33 2022 @author: john.obrecht
"""

import numpy as np
from matplotlib import cm
from fold.basic_functions import *

#%% Rainbow: Uniform color change
def rainbow_00(tree, num_pts, num_frames):
    slices = np.linspace(0, 1, num_frames)
    color_map = cm.get_cmap('hsv', num_frames)
    rainbow = color_map(slices)
    seq = np.zeros([num_pts, 4, num_frames])
    for i in range(num_frames):
        seq[:, :, i] = rainbow[i]
    return seq

#%% Rainbow: Vertical gradient color change
def rainbow_01(tree, num_pts, num_frames):
    slices = np.linspace(0, 1, num_frames)
    bin_nums = np.digitize(tree[:, 2], slices, right=True)
    color_map = cm.get_cmap('hsv', num_pts)
    rainbow = color_map(slices)
    seq = np.zeros([num_pts, 4, num_frames])
    for i in range(num_pts):
        b = bin_nums[i]
        for j in range(num_frames):
            seq[i, :, j] = rainbow[np.mod(b + j, num_frames)]
    return seq

#%% Rainbow: Radial gradient color change
def rainbow_02(tree, num_pts, num_frames):
    th_t = np.mod(180 / np.pi * np.arctan2(tree[:, 1], tree[:, 0]), 360)
    slices = np.linspace(0, 1, num_frames)
    bin_nums = np.digitize(th_t, 360 * slices, right=True)
    color_map = cm.get_cmap('hsv', num_pts)
    rainbow = color_map(slices)
    seq = np.zeros([num_pts, 4, num_frames])
    for i in range(num_pts):
        b = bin_nums[i]
        for j in range(num_frames):
            seq[i, :, j] = rainbow[np.mod(b + j, num_frames)]
    return seq

#%% Three-Arm Point-Spiral
def spiral_02(tree, num_pts, num_frames):
    x_t, y_t, z_t = tree[:, 0], tree[:, 1], tree[:, 2]
    r_sc = np.max(np.abs(tree[:, 0:2]))  # Radial scale = largest radial extent    
    r0 = r_sc * np.linspace(0, 1, num_frames)  # R moves from 0 (at top) to outside at bottom
    r1 = np.roll(r0, int(num_frames / 3))
    r2 = np.roll(r0, int(2 * num_frames / 3))
    th0 = np.linspace(0, 2, num_frames) * 360  # Theta circles X rotations
    th1 = np.mod(th0 + 120, 360)
    th2 = np.mod(th0 + 240, 360)
    x0, y0 = np.zeros(num_frames), np.zeros(num_frames)
    x1, y1 = np.zeros(num_frames), np.zeros(num_frames)
    x2, y2 = np.zeros(num_frames), np.zeros(num_frames)
    for i in range(num_frames):
        x0[i], y0[i], _ = np.dot(rz(th0[i]), [r0[i], 0, 0])
        x1[i], y1[i], _ = np.dot(rz(th1[i]), [r1[i], 0, 0])
        x2[i], y2[i], _ = np.dot(rz(th2[i]), [r2[i], 0, 0])
    z0 = np.linspace(1, 0, num_frames)  # Z falls from 1 to 0
    z1 = np.roll(z0, int(num_frames / 3))
    z2 = np.roll(z0, int(2 * num_frames / 3))    
    rgb = ((1, 0, 0), (0, 1, 0), (1, 1, 1))  # Set color
    seq = np.zeros([num_pts, 4, num_frames])
    sz = 0.1
    thr = 1E-2
    for i in range(num_frames):
        # 1st arm
        a0 = point(x_t, y_t, z_t, x0, y0, z0, sz, i)  # Only change col. 3 (alpha value)
        f0 = a0 > thr
        seq[f0, 0, i], seq[f0, 1, i], seq[f0 > thr, 2, i] = rgb[0][0], rgb[0][1], rgb[0][2]
        # 2nd arm
        a1 = point(x_t, y_t, z_t, x1, y1, z1, sz, i)  # Only change col. 3 (alpha value)
        f1 = a1 > thr
        seq[f1, 0, i], seq[f1, 1, i], seq[f1 > thr, 2, i] = rgb[1][0], rgb[1][1], rgb[1][2]
        # 3rd arm
        a2 = point(x_t, y_t, z_t, x2, y2, z2, sz, i)  # Only change col. 3 (alpha value)
        f2 = a2 > thr
        seq[f2, 0, i], seq[f2, 1, i], seq[f2 > thr, 2, i] = rgb[2][0], rgb[2][1], rgb[2][2]
        # Alpha values
        seq[:, 3, i] = a0 + a1 + a2
    return seq