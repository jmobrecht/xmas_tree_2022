"""
Created on Sat Oct 29 23:07:53 2022 @author: john.obrecht
"""

import numpy as np
from matplotlib import cm
from scipy import signal

### Basic Functions

def cosd(x):
    return 2 * np.pi * x

def dist(x, y, z, x0, y0, z0, i):
    return np.sqrt( (x - x0[i])**2 + (y - y0[i])**2 + (z - z0[i])**2)

def point(x, y, z, x0, y0, z0, sz, i):
    return np.exp(-(dist(x, y, z, x0, y0, z0, i))**2 / sz**2), 

def sheet(z, z0, i):
    return np.exp(-(z - z0[i])**2 / (0.01 * np.max(z))**2)

def wf_triangle(x, x0, w):
    # Normal triangle
    left = (x > x0 - w) & (x <= x0)
    right = (x > x0) & (x < x0 + w)
    y = np.zeros(np.shape(x))
    y[left] = 1 + (x[left] - x0) / w
    y[right] = 1 - (x[right] - x0) / w
    # Triangle > 1 --- when triangle goes off screen right
    if x0 + w > 1:
        right_2 = (x < np.mod(x0 + w, 1))
        y[right_2] = 1 - (x[right_2] - (x0 - 1)) / w
    # Triangle < 0 -- when triangle goes off screen left
    if x0 - w < 0:
        left_2 = (x > np.mod(x0 - w, 1))
        y[left_2] = 1 + (x[left_2] - (x0 + 1)) / w
    return y

def rx(th):
    th *= np.pi / 180
    return np.array([[1, 0, 0], [0, np.cos(th), np.sin(th)], [0, -np.sin(th), np.cos(th)]])
    
def ry(th):
    th *= np.pi / 180
    return np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])

def rz(th):
    th *= np.pi / 180
    return np.array([[np.cos(th), np.sin(th), 0], [-np.sin(th), np.cos(th), 0], [0, 0, 1]])

# Rainbow
def rainbow(tree, num_pts, num_frames):
    color_map = cm.get_cmap('hsv', num_pts)
    seq = np.zeros([num_pts, 4, num_frames])
    seq[:, :, 0] = color_map(np.linspace(0, 1, num_pts))
    for i in range(num_frames):
        seq[:, :, i] = np.roll(seq[:, :, i-1], 1, axis=0)
    return seq

# Slice
def moving_slice(tree, num_pts, num_frames):
    z0 = np.linspace(1, 0, num_frames)
    seq = np.ones([num_pts, 4, num_frames])
    for i in range(num_frames):
        seq[:, 3, i] = np.round(sheet(tree[:, 2], z0, i))
    return seq

# Rain
def falling_rain(tree, num_pts, num_frames):
    x_t, y_t, z_t = tree[:, 0], tree[:, 1], tree[:, 2]
    r_sc = np.max(np.abs(tree[:, 0:2]))  # Radial scale = largest radial extent    
    z0 = np.linspace(1, 0, num_frames)  # Z falls from 1 to 0
    y0 = np.linspace(0, 0, num_frames)  # Y remains fixed
    x0 = np.linspace(0, r_sc, num_frames)  # X moves from 0 (at top) to outside at bottom
    seq = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    for i in range(1, num_frames):
        seq[:, 3, i] = np.round(point(x_t, y_t, z_t, x0, y0, z0, 0.05, i))  # Only change col. 3 (alpha value)
    return seq

# Spiral
def spiral(tree, num_pts, num_frames):
    x_t, y_t, z_t = tree[:, 0], tree[:, 1], tree[:, 2]
    r_sc = np.max(np.abs(tree[:, 0:2]))  # Radial scale = largest radial extent    
    z0 = np.linspace(1, 0, num_frames)  # Z falls from 1 to 0
    t0 = np.linspace(0, 2, num_frames) * 360  # Theta circles X rotations
    r0 = np.linspace(0, r_sc, num_frames)  # R moves from 0 (at top) to outside at bottom
    x0, y0 = np.zeros(num_frames), np.zeros(num_frames)
    for i in range(num_frames):
        x0[i], y0[i], _ = np.dot(rz(t0[i]), [r0[i], 0, 0])
    seq = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    for i in range(num_frames):
        seq[:, 3, i] = np.round(point(x_t, y_t, z_t, x0, y0, z0, 0.1, i))  # Only change col. 3 (alpha value)
    return seq

# Illuminate points near to cone surface only
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

# Blink all lights on / off
def blink_01(tree, num_pts, num_frames):
    blink = 25  # number of frames for on / off
    filt = (np.mod(np.arange(num_frames), 2 * blink) < blink)
    seq = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    seq[:, :, filt] = 0* seq[:, :, filt]
    return seq

# Breathing Tree
def breathe_01(tree, num_pts, num_frames):
    period = 200
    p = 4
    seq = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    for i in range(num_frames):
        seq[:, 3, i] = (np.sin(2 * np.pi * i / period))**(2*p)  # Only change col. 3 (alpha value)
    return seq

# def sparkle_01(tree, num_pts, num_frames):
#     period = 200
#     phase = np.random.random(num_pts) * 2 * np.pi
    
    