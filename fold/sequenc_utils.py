"""
Created on Sat Oct 29 23:07:53 2022 @author: john.obrecht
"""

import numpy as np
from matplotlib import cm

### Basic Functions

def dist(x, y, z, x0, y0, z0, i):
    return np.sqrt( (x - x0[i])**2 + (y - y0[i])**2 + (z - z0[i])**2)

def point(x, y, z, x0, y0, z0, sz, i):
    return np.exp(-(dist(x, y, z, x0, y0, z0, i))**2 / sz**2), 

def sheet(z, z0, i):
    return np.exp(-(z - z0[i])**2 / (0.01 * np.max(tree[:, 2]))**2)


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
    for i in range(1, num_frames):
        seq[:, :, i] = np.roll(seq[:, :, i-1], 1, axis=0)
    return seq

# Slice
def moving_slice(tree, num_pts, num_frames):
    z0 = np.linspace(1, 0, num_frames)
    seq = np.ones([num_pts, 4, num_frames])
    for i in range(1, num_frames):
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
    for i in range(1, num_frames):
        seq[:, 3, i] = np.round(point(x_t, y_t, z_t, x0, y0, z0, 0.1, i))  # Only change col. 3 (alpha value)
    return seq
