"""
Created on Sat Oct 29 23:07:53 2022 @author: john.obrecht
"""

import numpy as np
from matplotlib import cm
from scipy import signal

### Basic Functions

def cosd(x):
    return np.cos(x * np.pi / 180)

def sind(x):
    return np.sin(x * np.pi / 180)

def dist(x, y, z, x0, y0, z0, i):
    return np.sqrt( (x - x0[i])**2 + (y - y0[i])**2 + (z - z0[i])**2)

def point(x, y, z, x0, y0, z0, sz, i):
    return np.exp(-(dist(x, y, z, x0, y0, z0, i))**2 / sz**2), 

def sheet(z, z0, i):
    return np.exp(-(z - z0[i])**2 / (0.01 * np.max(z))**2)

def wf_gaussian(x, x0, sz):
    return np.exp(-((x - x0)**2 / sz**2))

# Triangle through time coordinates
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

# Decay through time coordinates
def wf_decay(x, x0, tau):
    y = np.zeros(np.shape(x))
    y[x >= x0] = np.exp(-(x[x >= x0] - x0) / tau)
    # Handling the continuity past x = 1    
    x2 = x + 1
    y2 = y.copy()
    y2[x2 >= x0] = np.exp(-(x2[x2 >= x0] - x0) / tau)
    y = np.max((y, y2), axis=0)   
    return y

# Decay through spatial coordinates
def wf_decay_2(x, x0, tau, lim):
    x0 = lim - x0
    y = np.zeros(np.shape(x0))
    y[x <= x0] = np.exp((x - x0[x <= x0]) / tau)
    # Handling the continuity past x = 1    
    x2 = x0 + lim
    y2 = y.copy()
    y2[x <= x2] = np.exp((x - x2[x <= x2]) / tau)
    y = np.max((y, y2), axis=0)  
    return y[::-1]

def wf_pulse(x, xu, xl):
    y = np.zeros(np.shape(xu))
    y[(xu > x) & (x > xl)] = 1
    y[(xu > x + 360) & (x + 360 > xl)] = 1  # Edge case
    y[(xu > x - 360) & (x - 360 > xl)] = 1  # Edge case
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

#%% EFFECTS ###

# Rainbow: horizontal falling
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

# Rainbow: swirling vertical
def rainbow_02(tree, num_pts, num_frames):
    x_t, y_t = tree[:, 0], tree[:, 1]
    th_t = np.mod(180 / np.pi * np.arctan2(y_t, x_t), 360)
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
    p = 2
    seq = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    for i in range(num_frames):
        seq[:, 3, i] = (np.sin(2 * np.pi * i / period))**(2*p)  # Only change col. 3 (alpha value)
    return seq

# Sparkle: Triangle wave
def sparkle_01(tree, num_pts, num_frames):
    t = np.arange(0, num_frames)
    phase = np.random.random(num_pts)
    seq = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    for i in range(num_pts):
        seq[i, 3, :] = wf_triangle(t / num_frames, phase[i], 0.1)
    return seq

# Sparkle: Decay wave
def sparkle_02(tree, num_pts, num_frames):
    t = np.arange(0, num_frames)
    phase = np.random.random(num_pts)
    seq = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    for i in range(num_pts):
        seq[i, 3, :] = wf_decay(t / num_frames, phase[i], 0.15)
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
        seq[i, 3, :] = wf_pulse(th_t[i], lim_up, lim_dn)
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
        seq[i, 3, :] = wf_pulse(tree[i, 2], lim_up, lim_dn)
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