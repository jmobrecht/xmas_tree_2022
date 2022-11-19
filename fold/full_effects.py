"""
Created on Fri Nov  4 13:07:33 2022 @author: john.obrecht
"""

import random
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
def spiral_02(tree, num_pts, num_frames, rgb):
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

#%% Blink all lights on / off single color
def blink_00(tree, num_pts, num_frames, rgb):
    blink = int(num_frames / 2)  # number of frames for on / off
    filt = (np.mod(np.arange(num_frames), 2 * blink) < blink)
    seq = np.ones([num_pts, 4, num_frames])
    for i in range(3):
        seq[:, i, :] *= rgb[i]
    seq[:, :, filt] = 0 * seq[:, :, filt]
    return seq

#%% Blink all lights on / off single random color
def blink_01(tree, num_pts, num_frames):
    num_colors = 100
    slices = np.linspace(0, 1, num_colors)
    color_map = cm.get_cmap('hsv', num_colors)
    rainbow = color_map(slices)
    num_random_colors = 25
    random_numbers = (np.random.random(num_random_colors) * num_colors).astype(int)
    blink = int(num_frames / 2)
    filt = (np.mod(np.arange(num_frames), 2 * blink) < blink)
    for j in range(num_random_colors):
        rgb = rainbow[random_numbers[j]]
        seq = np.ones([num_pts, 4, num_frames])
        for i in range(3):
            seq[:, i, :] *= rgb[i]
        seq[:, :, filt] = 0 * seq[:, :, filt]
        if j == 0:
            seq_all = seq.copy()
        else:
            seq_all = np.append(seq_all, seq, axis=2)
    return seq_all

#%% Blink all lights on / off three colors
def blink_02(tree, num_pts, num_frames, rgb):
    blink = int(num_frames / 2)
    filt = (np.mod(np.arange(num_frames), 2 * blink) < blink)
    for j in range(3):
        rgb_i = rgb[j]
        seq = np.ones([num_pts, 4, num_frames])
        for i in range(3):
            seq[:, i, :] *= rgb_i[i]
        seq[:, :, filt] = 0 * seq[:, :, filt]
        if j == 0:
            seq_all = seq.copy()
        else:
            seq_all = np.append(seq_all, seq, axis=2)
    return seq_all

#%% Breathing Tree
def breathe_00(tree, num_pts, num_frames, rgb):
    period = num_frames * 2
    p = 2
    seq = np.ones([num_pts, 4, num_frames])
    for i in range(3):
        seq[:, i, :] *= rgb[i]
    for i in range(num_frames):
        seq[:, 3, i] = (np.sin(2 * np.pi * i / period))**(2*p)  # Only change col. 3 (alpha value)
    return seq

#%% Breathe all lights on / off single random color
def breathe_01(tree, num_pts, num_frames):
    num_colors = 100
    slices = np.linspace(0, 1, num_colors)
    color_map = cm.get_cmap('hsv', num_colors)
    rainbow = color_map(slices)
    num_random_colors = 25
    random_numbers = (np.random.random(num_random_colors) * num_colors).astype(int)
    period = num_frames * 2
    p = 2
    for j in range(num_random_colors):
        rgb = rainbow[random_numbers[j]]
        seq = np.ones([num_pts, 4, num_frames])
        for i in range(3):
            seq[:, i, :] *= rgb[i]
        for k in range(num_frames):
            seq[:, 3, k] = (np.sin(2 * np.pi * k / period))**(2*p)  # Only change col. 3 (alpha value)
        if j == 0:
            seq_all = seq.copy()
        else:
            seq_all = np.append(seq_all, seq, axis=2)
    return seq_all

#%% Breathe all lights on / off three colors
def breathe_02(tree, num_pts, num_frames, rgb):
    period = num_frames * 2
    p = 2
    for j in range(3):
        rgb_i = rgb[j]
        seq = np.ones([num_pts, 4, num_frames])
        for i in range(3):
            seq[:, i, :] *= rgb_i[i]
        for k in range(num_frames):
            seq[:, 3, k] = (np.sin(2 * np.pi * k / period))**(2*p)  # Only change col. 3 (alpha value)
        if j == 0:
            seq_all = seq.copy()
        else:
            seq_all = np.append(seq_all, seq, axis=2)
    return seq_all

#%% Sparkle: Pulse wave
def sparkle_00(tree, num_pts, num_frames, rgb):
    t = np.arange(0, num_frames) / num_frames
    phase = np.random.random(num_pts)
    span = 1 / 10
    lim_up, lim_dn = span, -span
    seq = np.ones([num_pts, 4, num_frames])
    for i in range(3):
        seq[:, i, :] *= rgb[i]
    for i in range(num_pts):
        seq[i, 3, :] = wf_pulse_x(t + phase[i], lim_up, lim_dn)
    return seq

#%% Sparkle: Triangle wave
def sparkle_01(tree, num_pts, num_frames, rgb):
    t = np.arange(0, num_frames)
    phase = np.random.random(num_pts)
    seq = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    for i in range(3):
        seq[:, i, :] *= rgb[i]
    for i in range(num_pts):
        seq[i, 3, :] = wf_triangle(t / num_frames, phase[i], 0.15)
    return seq

#%% Sparkle: Decay wave
def sparkle_02(tree, num_pts, num_frames, rgb):
    t = np.arange(0, num_frames)
    phase = np.random.random(num_pts)
    seq = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    for i in range(3):
        seq[:, i, :] *= rgb[i]
    for i in range(num_pts):
        seq[i, 3, :] = wf_decay(t / num_frames, phase[i], 0.15)
    return seq

#%% Sparkle: Pulse wave - Random colors
def sparkle_00_R(tree, num_pts, num_frames):
    slices = np.linspace(0, 1, num_pts)
    random.shuffle(slices)
    color_map = cm.get_cmap('hsv', num_pts)
    rainbow = color_map(slices)
    t = np.arange(0, num_frames) / num_frames
    phase = np.random.random(num_pts)
    span = 1 / 10
    lim_up, lim_dn = span, -span
    seq = np.ones([num_pts, 4, num_frames])
    for i in range(num_pts):
        for j in range(3):
            seq[i, j, :] *= rainbow[i][j]
        seq[i, 3, :] = wf_pulse_x(t + phase[i], lim_up, lim_dn)
    return seq

#%% Sparkle: Triangle wave - Random colors
def sparkle_01_R(tree, num_pts, num_frames):
    slices = np.linspace(0, 1, num_pts)
    random.shuffle(slices)
    color_map = cm.get_cmap('hsv', num_pts)
    rainbow = color_map(slices)
    t = np.arange(0, num_frames)
    phase = np.random.random(num_pts)
    seq = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    for i in range(num_pts):
        for j in range(3):
            seq[i, j, :] *= rainbow[i][j]
        seq[i, 3, :] = wf_triangle(t / num_frames, phase[i], 0.15)
    return seq

#%% Sparkle: Decay wave - Random colors
def sparkle_02_R(tree, num_pts, num_frames):
    slices = np.linspace(0, 1, num_pts)
    random.shuffle(slices)
    color_map = cm.get_cmap('hsv', num_pts)
    rainbow = color_map(slices)
    t = np.arange(0, num_frames)
    phase = np.random.random(num_pts)
    seq = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    for i in range(num_pts):
        for j in range(3):
            seq[i, j, :] *= rainbow[i][j]
        seq[i, 3, :] = wf_decay(t / num_frames, phase[i], 0.15)
    return seq