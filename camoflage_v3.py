"""
Created on Wed Nov 30 21:49:14 2022 @author: john.obrecht
"""

import numpy as np
from numpy.fft import ifftn
import matplotlib.pyplot as plt

def gaussian_3d(fx, fy, fz, s):
    return np.exp(-(fx**2 + fy**2 + fz**2) / s**2)

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
xy_max = 0.3
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
freq_ifft_3d = ifftn(freq_complex_3d)

#%%

plt.figure(tight_layout=True, figsize=(20, 4))
slices = np.arange(0, num_pts_z, 30)
for i, layer in enumerate(slices):
    plt.subplot(2, len(slices), i + 1)
    plt.pcolor(np.real(freq_ifft_3d[:, :, layer]), cmap='hsv')
    plt.axis('equal')
for i, layer in enumerate(slices):
    plt.subplot(2, len(slices), i + 1 + len(slices))
    plt.pcolor(np.imag(freq_ifft_3d[:, :, layer]), cmap='hsv')
    plt.axis('equal')
    
plt.figure(tight_layout=True, figsize=(20, 4))
slices = np.arange(0, num_pts_xy, 10)
for i, layer in enumerate(slices):
    plt.subplot(2, len(slices), i + 1)
    plt.pcolor(np.real(freq_ifft_3d[:, layer, :]), cmap='hsv')
    plt.axis('equal')
for i, layer in enumerate(slices):
    plt.subplot(2, len(slices), i + 1 + len(slices))
    plt.pcolor(np.imag(freq_ifft_3d[:, layer, :]), cmap='hsv')
    plt.axis('equal')
