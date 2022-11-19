"""
Created on Sat Oct 29 23:07:53 2022 @author: john.obrecht
"""

import numpy as np
from matplotlib import cm
from fold.basic_functions import *

# Point-Spiral: single arm
def spiral_01(tree, num_pts, num_frames):
    r_sc = np.max(np.abs(tree[:, 0:2]))  # Radial scale = largest radial extent    
    r0 = np.linspace(0, r_sc, num_frames)  # R moves from 0 (at top) to outside at bottom
    th0 = np.linspace(0, 2, num_frames) * 360  # Theta circles X rotations
    x0, y0 = np.zeros(num_frames), np.zeros(num_frames)
    for i in range(num_frames):
        x0[i], y0[i], _ = np.dot(rz(th0[i]), [r0[i], 0, 0])
    z0 = np.linspace(1, 0, num_frames)  # Z falls from 1 to 0
    rgb = (1, 0, 0)  # Set color
    seq = np.zeros([num_pts, 4, num_frames])
    seq[:, 0, :], seq[:, 1, :], seq[:, 2, :] = rgb[0], rgb[1], rgb[2]
    for i in range(num_frames):
        seq[:, 3, i] = point(tree[:, 0], tree[:, 1], tree[:, 2], x0, y0, z0, 0.08, i)  # Only change col. 3 (alpha value)
    return seq

### moving down the list

# Rain
def falling_rain(tree, num_pts, num_frames):
    x_t, y_t, z_t = tree[:, 0], tree[:, 1], tree[:, 2]
    r_sc = np.max(np.abs(tree[:, 0:2]))  # Radial scale = largest radial extent    
    z0 = np.linspace(1, 0, num_frames)  # Z falls from 1 to 0
    y0 = np.linspace(0, 0, num_frames)  # Y remains fixed
    x0 = np.linspace(0, r_sc, num_frames)  # X moves from 0 (at top) to outside at bottom
    seq = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    for i in range(1, num_frames):
        seq[:, 3, i] = point(x_t, y_t, z_t, x0, y0, z0, 0.05, i)  # Only change col. 3 (alpha value)
    return seq

# Illuminate points near (threshold) to cone surface only
def cone_01(tree, num_pts, num_frames):
    def dist(r, z, r_sc):
        return np.abs(r / r_sc + z - 1) / np.sqrt(1 + r_sc**-2)
    r_sc = np.max(np.abs(tree[:, 0:2]))  # Radial scale = largest radial extent
    r = np.sqrt(tree[:, 0]**2 + tree[:, 1]**2)
    d = dist(r, tree[:, 2], r_sc)
    seq = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    seq[d > 0.04, :, :] = 0
    return seq

# Alpha inversely proportional to distance to cone surface
def cone_02(tree, num_pts, num_frames):
    def dist(r, z, r_sc):
        return np.abs(r / r_sc + z - 1) / np.sqrt(1 + r_sc**-2)
    r_sc = np.max(np.abs(tree[:, 0:2]))  # Radial scale = largest radial extent
    r = np.sqrt(tree[:, 0]**2 + tree[:, 1]**2)
    d = dist(r, tree[:, 2], r_sc)
    a = 1 - d / np.max(d)
    seq = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    for i in range(num_frames):
        seq[:, 3, i] = a
    return seq

# Swirling Vertical Stripes: hard-coded pulse waveform
def swirl_01(tree, num_pts, num_frames):
    th_t = np.mod(180 / np.pi * np.arctan2(tree[:, 1], tree[:, 0]), 360)
    num_stripes = 6
    span = 360 / num_stripes / 2
    t0 = np.linspace(0, 1, num_frames) * 360  # Theta circles X rotations
    lim_up, lim_dn = np.mod(t0 + span, 360), np.mod(t0 - span, 360)
    seq = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    seq[:, 3, :] = 0
    for i in range(num_frames):
        upper, lower = lim_up[i], lim_dn[i]
        filt = (upper > th_t) | (th_t > lower) if lower > upper else (upper > th_t) & (th_t > lower)
        seq[filt, 3, i] = 1
    return seq

# Swirling Vertical Stripes: pulse waveform
def swirl_02(tree, num_pts, num_frames):
    th_t = np.mod(180 / np.pi * np.arctan2(tree[:, 1], tree[:, 0]), 360)
    num_stripes = 6
    span = 360 / num_stripes / 2
    t0 = np.linspace(0, 1, num_frames) * 360  # Theta circles X rotations
    lim_up, lim_dn = t0 + span, t0 - span
    seq = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    seq[:, 3, :] = 0
    for i in range(num_pts):
        seq[i, 3, :] = wf_pulse_th(th_t[i], lim_up, lim_dn)
    return seq

# Swirling Vertical Stripes: gaussian waveform
def swirl_03(tree, num_pts, num_frames):
    th_t = np.mod(180 / np.pi * np.arctan2(tree[:, 1], tree[:, 0]), 360)
    num_stripes = 6
    span = 360 / num_stripes / 2
    t0 = np.linspace(0, 1, num_frames) * 360  # Theta circles X rotations
    seq = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    seq[:, 3, :] = 0
    for i in range(num_pts):
        seq[i, 3, :] = wf_gaussian(th_t[i], t0, span)
    return seq

# Swirling Vertical Stripes: decay waveform
def swirl_04(tree, num_pts, num_frames):
    th_t = np.mod(180 / np.pi * np.arctan2(tree[:, 1], tree[:, 0]), 360)
    num_stripes = 6
    span = 360 / num_stripes / 2
    t0 = np.linspace(0, 1, num_frames) * 360  # Theta circles X rotations
    seq = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    seq[:, 3, :] = 0
    for i in range(num_pts):
        seq[i, 3, :] = wf_decay_2(th_t[i], t0, span, 360)
    return seq

# Rising Horizontal Stripes: pulse waveform
def slice_01(tree, num_pts, num_frames):
    num_stripes = 8
    span = 1 / num_stripes / 2
    z0 = np.linspace(0, 1, num_frames)  # Height rising linearly
    lim_up, lim_dn = z0 + span, z0 - span
    seq = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    seq[:, 3, :] = 0
    for i in range(num_pts):
        seq[i, 3, :] = wf_pulse_th(tree[i, 2], lim_up, lim_dn)
    return seq

# Falling Horizontal Stripes: gaussian waveform
def slice_02(tree, num_pts, num_frames):
    z0 = np.linspace(1, 0, num_frames)
    seq = np.ones([num_pts, 4, num_frames])
    for i in range(num_pts):
        seq[i, 3, :] = wf_gaussian(tree[i, 2], z0, 0.01)
    return seq

# Falling Horizontal Stripes: decay waveform
def melt_01(tree, num_pts, num_frames):
    z0 = np.linspace(1, 0, num_frames)
    seq = np.ones([num_pts, 4, num_frames])
    for i in range(num_pts):
        seq[i, 3, :] = wf_decay_2(tree[i, 2], z0, 0.1, 1)
    return seq