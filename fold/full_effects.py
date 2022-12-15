"""
Created on Fri Nov  4 13:07:33 2022 @author: john.obrecht
"""

import random
import numpy as np
from matplotlib import cm
from fold.basic_functions import *
from scipy.interpolate import RegularGridInterpolator

#%% Rainbow: Uniform color change
def all_on(tree, num_pts, num_frames, rgb):
    seq = np.ones([num_pts, 4, num_frames])
    for i in range(3):
        seq[:, i, :] *= rgb[i]
    return seq

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

#%% Rainbow: Angular gradient color change
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

#%% Rainbow: Radial gradient color change
def rainbow_03(tree, num_pts, num_frames):
    rh = np.sqrt(tree[:, 1]**2 + tree[:, 0]**2)
    slices = np.linspace(1, 0, num_frames)
    bin_nums = np.digitize(rh, slices, right=True)
    color_map = cm.get_cmap('hsv', num_pts)
    rainbow = color_map(slices)
    seq = np.zeros([num_pts, 4, num_frames])
    for i in range(num_pts):
        b = bin_nums[i]
        for j in range(num_frames):
            seq[i, :, j] = rainbow[np.mod(b + j, num_frames)]
    return seq

#%% Sparkle: Pulse wave
def rainbow_random(tree, num_pts, num_frames):
    slices = np.linspace(1, 0, num_frames)
    ph = np.round(np.random.random(num_pts) * 255, 0).astype(int)
    color_map = cm.get_cmap('hsv', num_pts)
    rainbow = color_map(slices)
    seq = np.ones([num_pts, 4, num_frames])
    for i in range(num_pts):
        seq[i, 0:2, :] = np.transpose(np.roll(rainbow[:, 0:2], ph[i]))
    return seq

#%% Twilight: Uniform color change
def twilight_00(tree, num_pts, num_frames):
    slices = np.linspace(0, 1, num_frames)
    color_map = cm.get_cmap('twilight', num_frames)
    rainbow = color_map(slices)
    seq = np.zeros([num_pts, 4, num_frames])
    for i in range(num_frames):
        seq[:, :, i] = rainbow[i]
    return seq

#%% Twilight: Vertical gradient color change
def twilight_01(tree, num_pts, num_frames):
    slices = np.linspace(0, 1, num_frames)
    bin_nums = np.digitize(tree[:, 2], slices, right=True)
    color_map = cm.get_cmap('twilight', num_pts)
    rainbow = color_map(slices)
    seq = np.zeros([num_pts, 4, num_frames])
    for i in range(num_pts):
        b = bin_nums[i]
        for j in range(num_frames):
            seq[i, :, j] = rainbow[np.mod(b + j, num_frames)]
    return seq

#%% Twilight: Radial gradient color change
def twilight_02(tree, num_pts, num_frames):
    th_t = np.mod(180 / np.pi * np.arctan2(tree[:, 1], tree[:, 0]), 360)
    slices = np.linspace(0, 1, num_frames)
    bin_nums = np.digitize(th_t, 360 * slices, right=True)
    color_map = cm.get_cmap('twilight', num_pts)
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
    th0 = np.linspace(0, 6, num_frames) * 360  # Theta circles X rotations
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

#%% Sparkle: Reverse Pulse wave
def sparkle_00b(tree, num_pts, num_frames, rgb):
    t = np.arange(0, num_frames) / num_frames
    phase = np.random.random(num_pts)
    span = 1 / 10
    lim_up, lim_dn = span, -span
    seq = np.ones([num_pts, 4, num_frames])
    for i in range(3):
        seq[:, i, :] *= rgb[i]
    for i in range(num_pts):
        seq[i, 3, :] = 1 - wf_pulse_x(t + phase[i], lim_up, lim_dn)
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

#%% Sparkle: Reverse Triangle wave
def sparkle_01b(tree, num_pts, num_frames, rgb):
    t = np.arange(0, num_frames)
    phase = np.random.random(num_pts)
    seq = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    for i in range(3):
        seq[:, i, :] *= rgb[i]
    for i in range(num_pts):
        seq[i, 3, :] = 1 - wf_triangle(t / num_frames, phase[i], 0.15)
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

#%% Sparkle: Reverse Decay wave
def sparkle_02b(tree, num_pts, num_frames, rgb):
    t = np.arange(0, num_frames)
    phase = np.random.random(num_pts)
    seq = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    for i in range(3):
        seq[:, i, :] *= rgb[i]
    for i in range(num_pts):
        seq[i, 3, :] = 1 - wf_decay(t / num_frames, phase[i], 0.15)
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

#%% Swirling Vertical Stripes: pulse waveform
def stripes_V_00(tree, num_pts, num_frames, rgb, stripes=6, thickness=1):
    th_t = np.mod(180 / np.pi * np.arctan2(tree[:, 1], tree[:, 0]), 360)
    span = 360 / stripes * thickness
    t0 = np.linspace(0, 1, num_frames) * 360  # Theta circles X rotations
    seq_a = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    seq_b = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    seq_a[:, 3, :] = 0
    seq_b[:, 3, :] = 0
    for i in range(num_pts):
        pulse_a = np.zeros(num_frames)
        pulse_b = np.zeros(num_frames)
        for k in range(int(stripes / 2)):

#             pulse_a += wf_pulse_th(th_t[i], t0 + (2 * k + 1) * span, t0 + (2 * k + 0) * span)
#             pulse_b += wf_pulse_th(th_t[i], t0 + (2 * k + 2) * span, t0 + (2 * k + 1) * span)

            pulse_a += wf_pulse_th(th_t[i], t0 + 360 * (2 * k + 0) / stripes + span, t0 + 360 * (2 * k + 0) / stripes)
            pulse_b += wf_pulse_th(th_t[i], t0 + 360 * (2 * k + 1) / stripes + span, t0 + 360 * (2 * k + 1) / stripes)
        seq_a[i, 3, :] = pulse_a
        seq_b[i, 3, :] = pulse_b
    for j in range(3):
        seq_a[:, j, :] *= rgb[0][j]
        seq_b[:, j, :] *= rgb[1][j]
        seq_a[:, j, :] *= seq_a[:, 3, :]
        seq_b[:, j, :] *= seq_b[:, 3, :]
    seq = seq_a + seq_b
    return seq

#%% Falling Stripes: pulse waveform
def stripes_H_00(tree, num_pts, num_frames, rgb, stripes=6, thickness=1):
    span = 1 / stripes * thickness
    z0 = np.linspace(1, 0, num_frames) # Height rising linearly
    seq_a = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    seq_b = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    seq_a[:, 3, :] = 0
    seq_b[:, 3, :] = 0
    for i in range(num_pts):
        pulse_a = np.zeros(num_frames)
        pulse_b = np.zeros(num_frames)
        for k in range(stripes):
            pulse_a += wf_pulse_th(tree[i, 2], 1 + z0 - (2 * k + 0) / stripes, 1 + z0 - (2 * k + 0) / stripes - span)
            pulse_b += wf_pulse_th(tree[i, 2], 1 + z0 - (2 * k + 1) / stripes, 1 + z0 - (2 * k + 1) / stripes - span)
        seq_a[i, 3, :] = pulse_a
        seq_b[i, 3, :] = pulse_b
    for j in range(3):
        seq_a[:, j, :] *= rgb[0][j]
        seq_b[:, j, :] *= rgb[1][j]
        seq_a[:, j, :] *= seq_a[:, 3, :]
        seq_b[:, j, :] *= seq_b[:, 3, :]
    seq = seq_a + seq_b
    return seq

#%% Alpha inversely proportional to distance to cone surface
# def cone_02(tree, num_pts, num_frames, rgb):
#     r_sc0 = np.max(np.abs(tree[:, 0:2]))  # Radial scale = largest radial extent
#     r = np.sqrt(tree[:, 0]**2 + tree[:, 1]**2)
#     b = wf_triangle(np.arange(0, num_frames, 1), num_frames / 2, num_frames / 2)
#     seq = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
#     for i in range(num_frames):
#         r_sc = b[i] * r_sc0
#         a = 1 - 4 * dist_cone_surface(r, tree[:, 2], r_sc) / r_sc0
#         a[a < 0] = 0
#         seq[:, 3, i] = a
#     for j in range(3):
#         seq[:, j, :] *= rgb[j]
#     return seq

#%% Face
def face_00(tree, num_pts, num_frames, rgb):
    z_eyes = 0.50
    r_eyes = 0.1
    th_eyes = 90  # degrees
    z_mouth = 0.30
    r_mouth = 0.20
    z_mouth_offset = 0.07
    th_0 = 345  # degrees
    r_sc = np.max(np.abs(tree[:, 0:2]))
    t_surf = 0.1
    # Eyes
    rho_eyes = z_eyes * r_sc  # radial extent of eyes
    eye = [0, rho_eyes, z_eyes]
    eye_l = rz(th_0 - th_eyes / 2).dot(eye)
    eye_r = rz(th_0 + th_eyes / 2).dot(eye)
    # Mouth
    z_mouth_0 = z_mouth + z_mouth_offset
    rho_mouth_0 = z_mouth_0 * r_sc
    mouth_0 = rz(th_0).dot([0, rho_mouth_0, z_mouth_0])    
    d_eye_l = np.zeros(num_pts)
    d_eye_r = np.zeros(num_pts)
    d_mouth = np.zeros(num_pts)
    for i in range(num_pts):
        d_eye_l[i] = dist2(eye_l, tree[i, :])
        d_eye_r[i] = dist2(eye_r, tree[i, :])
        d_mouth[i] = dist2(mouth_0, tree[i, :])
    r = np.sqrt(tree[:, 0]**2 + tree[:, 1]**2)
    d_cs = dist_cone_surface(r, tree[:, 2], r_sc)
    seq_a = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    seq_b = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    seq_a[:, 3, :] = 0
    for i in range(num_frames):
        filt_mouth = (d_mouth < r_mouth) & (tree[:, 2] < z_mouth) & (d_cs < t_surf)
        seq_a[filt_mouth, 3, i] = 1
        seq_b[filt_mouth, 3, i] = 0
        filt_eye_l = (d_eye_l < r_eyes) & (d_cs < t_surf)
        seq_a[filt_eye_l, 3, i] = 1
        seq_b[filt_eye_l, 3, i] = 0
        filt_eye_r = (d_eye_r < r_eyes) & (d_cs < t_surf)
        seq_a[filt_eye_r, 3, i] = 1
        seq_b[filt_eye_r, 3, i] = 0
    for j in range(3):
        seq_a[:, j, :] *= rgb[0][j]
        seq_b[:, j, :] *= rgb[1][j]
        seq_a[:, j, :] *= seq_a[:, 3, :]
        seq_b[:, j, :] *= seq_b[:, 3, :]
    seq = seq_a + seq_b
    return seq

#%% Rain
def rain_00(tree, num_pts, num_frames, drops=10):
    x_t, y_t, z_t = tree[:, 0], tree[:, 1], tree[:, 2]
    r_sc = np.max(np.abs(tree[:, 0:2]))  # Radial scale = largest radial extent    
    ph_posr = np.random.random(3 * drops) * 0.5 + 0.5
    ph_post = np.random.random(3 * drops) * 2 * np.pi
    ph_posz = np.random.random(3 * drops) * 2 - 1
    ph_size = np.random.random(3 * drops) * 0.04 + 0.04
    z0 = np.linspace(1, 0, num_frames)  # Z falls from 1 to 0
    seq = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    seq[:, 3, :] = 0
    for k in range(3 * drops):
        r0 = (1 - (z0 + ph_posz[k])) * r_sc * ph_posr[k]
        r0[r0 < 0] = 0
        p = [r0 * np.cos(ph_post[k]), r0 * np.sin(ph_post[k]), z0 + ph_posz[k]]
        for i in range(1, num_frames):
            d = dist(x_t, y_t, z_t, p[0], p[1], p[2], i)
            seq[d < ph_size[k], 3, i] = 1  # Only change col. 3 (alpha value)
    tmp = seq[:, 3, :]
    tmp[tmp > 1] = 1
    seq[:, 3, :] = tmp
    return seq

#%% Rain
def rain_01(tree, num_pts, num_frames, drops=10):
    x_t, y_t, z_t = tree[:, 0], tree[:, 1], tree[:, 2]
    r_sc = np.max(np.abs(tree[:, 0:2]))  # Radial scale = largest radial extent    
    ph_posr = np.random.random(3 * drops) * 0.5 + 0.5
    ph_post = np.random.random(3 * drops) * 2 * np.pi
    ph_posz = np.random.random(3 * drops) * 2 - 1
    ph_size = np.random.random(3 * drops) * 0.04 + 0.04
    z0 = np.linspace(1, 0, num_frames)  # Z falls from 1 to 0
    seq = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    seq[:, 3, :] = 0
    for k in range(3 * drops):
        r0 = (1 - (z0 + ph_posz[k])) * r_sc * ph_posr[k]
        r0[r0 < 0] = 0
        p = [r0 * np.cos(ph_post[k]), r0 * np.sin(ph_post[k]), z0 + ph_posz[k]]
        for i in range(1, num_frames):
            a = np.round(point(x_t, y_t, z_t, p[0], p[1], p[2], ph_size[k], i), 3)
            seq[:, 3, i] += a  # Only change col. 3 (alpha value)
    tmp = seq[:, 3, :]
    tmp[tmp > 1] = 1
    seq[:, 3, :] = tmp
    return seq

#%% Fill tree
def fill_00(tree, num_pts, num_frames):
    slices = np.linspace(0.02, 1.02, 100)
    bin_nums = np.digitize(tree[:, 2], slices, right=True)
    color_map = cm.get_cmap('bwr', num_pts)
    rainbow = color_map(slices)
    z1 = np.linspace(0, 1, num_frames) # Height rising linearly
    z2 = z1[::-1]
    seq1 = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    seq1[:, 3, :] = 0
    for j in range(num_pts):
        b = bin_nums[j]
        for k in range(3):
            seq1[j, k, :] = rainbow[b, k]
        for i in range(num_frames):
            seq1[tree[:, 2] < z1[i], 3, i] = 1
    seq2 = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    seq2[:, 3, :] = 0
    for j in range(num_pts):
        b = bin_nums[j]
        for k in range(3):
            seq2[j, k, :] = rainbow[b, k]
        for i in range(num_frames):
            seq2[tree[:, 2] < z2[i], 3, i] = 1
    rainbow = color_map(slices[::-1])
    seq3 = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    seq3[:, 3, :] = 0
    for j in range(num_pts):
        b = bin_nums[j]
        for k in range(3):
            seq3[j, k, :] = rainbow[b, k]
        for i in range(num_frames):
            seq3[tree[:, 2] < z1[i], 3, i] = 1
    seq4 = np.ones([num_pts, 4, num_frames])  # Start all white (1, 1, 1, 1)
    seq4[:, 3, :] = 0
    for j in range(num_pts):
        b = bin_nums[j]
        for k in range(3):
            seq4[j, k, :] = rainbow[b, k]
        for i in range(num_frames):
            seq4[tree[:, 2] < z2[i], 3, i] = 1
    seq = np.concatenate([seq1, seq2, seq3, seq4], axis=2)
    return seq

#%% Fill tree 2
def fill_02(tree, num_pts):
    slices = np.arange(0, num_pts, 1)
    color_map = cm.get_cmap('rainbow', num_pts)
    rainbow = color_map(slices)
    seq = np.ones([num_pts, 4, num_pts])  # Start all white (1, 1, 1, 1)
    seq[:, 3, :] = 0
    for j in range(num_pts):
        for k in range(3):
            seq[j, k, :] = rainbow[j, k]
        seq[:j, 3, j] = 1
    return seq

#%% Camoflage: Rainbow
def camoflage_rainbow(tree, num_pts, num_frames):
    # Z-pos
    d_z = 1 / 101
    z_max = 3
    z = np.arange(0, z_max, d_z)
    d_z = z[1] - z[0]
    freq_max = 2 / d_z
    num_pts_z = len(z)
    num_pts_half_z = int((num_pts_z + 1) / 2)
    # Z-freq
    freq_z = np.linspace(0, freq_max, num_pts_z)
    freq1_z = freq_z[:num_pts_half_z]
    # XY-pos
    xy_max = 0.35
    x = np.arange(-xy_max, xy_max, d_z)
    num_pts_xy = len(x)
    num_pts_half_xy = int((num_pts_xy + 1) / 2)
    # XY-freq
    freq_xy = np.linspace(0, freq_max, num_pts_xy)
    freq1_xy = freq_xy[:num_pts_half_xy]
    # 3D
    w3 = 1 / 30 * freq_max
    freqX, freqY, freqZ = np.meshgrid(freq1_xy, freq1_xy, freq1_z)
    freq_amp_all_1 = gaussian_3d(freqX, freqY, freqZ, w3)
    tmp_x = np.concatenate([freq_amp_all_1, freq_amp_all_1[1:, :, :][::-1, :, :]])
    tmp_y = np.concatenate([tmp_x, tmp_x[:, 1:, :][:, ::-1, :]], axis=1)
    tmp_z = np.concatenate([tmp_y, tmp_y[:, :, 1:][:, :, ::-1]], axis=2)
    ph_z = np.random.random(np.shape(tmp_z)) * np.pi
    freq_complex_3d = tmp_z * np.exp(-1j * ph_z)
    field = np.real(np.fft.ifftn(freq_complex_3d))
    field = np.concatenate([field, field], axis=2)
    # Scale field
    field -= np.min(field)
    field /= (np.max(field) / 100)
    # Interpolation
    zp = np.concatenate([z, z + z_max])
    fn = RegularGridInterpolator((x, x, zp), field)
    # Mapping field onto tree points
    step_size = z_max / num_frames
    tree[tree[:, 2] > 1, 2] = 1
    tree[tree[:, 2] < 0, 2] = 0
    color_map = cm.get_cmap('hsv', 100)
    seq = np.ones([num_pts, 4, num_frames])
    for j in range(num_frames):
        for i in range(num_pts):
            seq[i, :3, j] = color_map(int(fn(tree[i, :3])))[:3]
        tree[:, 2] += step_size
    return seq
