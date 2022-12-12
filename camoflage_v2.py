"""
Created on Wed Nov 30 21:49:14 2022 @author: john.obrecht
"""

import numpy as np
from numpy.fft import ifft
import matplotlib.pyplot as plt

# Z-pos
d_z = 1 / 101
z_max = 1
z = np.arange(0, z_max, d_z)
d_z = z[1] - z[0]
freq_max = 2 / d_z
num_pts_z = len(z)
num_pts_half_z = int((num_pts_z + 1) / 2)

# Z-freq
d_freq_z = freq_max / (num_pts_z - 1)
freq_z = np.linspace(0, freq_max, num_pts_z)
freq1_z = freq_z[:num_pts_half_z]

# XY-pos
xy_max = 0.3
x = np.arange(-xy_max, xy_max, d_z)
y = x.copy()
num_pts_xy = len(x)
num_pts_half_xy = int((num_pts_xy + 1) / 2)

# XY-freq
d_freq_xy = freq_max / (num_pts_xy - 1)
freq_xy = np.linspace(0, freq_max, num_pts_xy)
freq1_xy = freq_xy[:num_pts_half_xy]

#%% 

# Calculate frequency spectrum - gaussian amplitude
def gaussian(f, s):
    return np.exp(-f**2 / s**2)

w = 1 / 20 * freq_max
freq = np.concatenate([freq1_z, freq1_z[1:][::-1]])
freq_amp = gaussian(freq, w)

# Calculate frequency spectrum - random phase (note the negative on the 2nd half phases)
phase1 = np.random.random(num_pts_half_z) * np.pi
phase = np.concatenate([phase1, -phase1[1:][::-1]])
phase[0] = 0

# Construct the complex frequency spectrum
freq_complex = freq_amp * np.exp(-1j * phase)
freq_ifft = ifft(freq_complex)

# Plot
plt.figure(tight_layout=True, figsize=(9, 6))
plt.plot(z, freq_ifft, 'r.:')
plt.plot(z + z_max, freq_ifft, 'b.:')

#%%

def gaussian_3d(fx, fy, fz, s):
    return np.exp(-(1 * fx**2 + 1 * fy**2 + 1 * fz**2) / s**2)

w = 1 / 20 * freq_max
freqX, freqY, freqZ = np.meshgrid(freq1_xy, freq1_xy, freq1_z)
freq_amp_all_1 = gaussian_3d(freqX, freqY, freqZ, w)
tmp_x = np.concatenate([freq_amp_all_1, freq_amp_all_1[1:, :, :][::-1, :, :]])
tmp_y = np.concatenate([tmp_x, tmp_x[:, 1:, :][:, ::-1, :]], axis=1)
tmp_z = np.concatenate([tmp_y, tmp_y[:, :, 1:][:, :, ::-1]], axis=2)

phase1_all = np.random.random(np.shape(freqX)) * np.pi
# phase1_z = np.random.random(num_pts_half_z) * np.pi
# phase1_z[0] = 0
# phase = phase1_z.reshape(1, 1, num_pts_half_z)
# phase1_all = np.tile(phase, (num_pts_half_xy, num_pts_half_xy, 1))
# phase1_z = np.random.random(num_pts_half_xy) * np.pi
# phase1_z[0] = 0
# phase = phase1_z.reshape(1, num_pts_half_xy, 1)
# phase1_all = np.tile(phase, (num_pts_half_xy, 1, num_pts_half_z))

ph_x = np.concatenate([phase1_all, phase1_all[1:, :, :][::-1, :, :]])
ph_y = np.concatenate([ph_x, ph_x[:, 1:, :][:, ::-1, :]], axis=1)
ph_z = np.concatenate([ph_y, ph_y[:, :, 1:][:, :, ::-1]], axis=2)

freq_complex_3d = tmp_z * np.exp(-1j * ph_z)
freq_ifft_3d = ifft(freq_complex_3d)

#%%

plt.figure(tight_layout=True, figsize=(9, 6))

plt.subplot(3, 1, 1)
plt.plot(x, freq_ifft_3d[:, 0, 0], 'r.:')
plt.plot(x + 2 * xy_max, freq_ifft_3d[:, 0, 0], 'k.:')

plt.subplot(3, 1, 2)
plt.plot(y, freq_ifft_3d[0, :, 0], 'g.:')
plt.plot(y + 2 * xy_max, freq_ifft_3d[0, :, 0], 'k.:')

plt.subplot(3, 1, 3)
plt.plot(z, freq_ifft_3d[0, 0, :], 'b.:')
plt.plot(z + z_max, freq_ifft_3d[0, 0, :], 'k.:')

