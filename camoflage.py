"""
Created on Wed Nov 30 21:49:14 2022 @author: john.obrecht
"""

import numpy as np
from numpy.fft import fftshift, ifft
import matplotlib.pyplot as plt

num_pts = 101
num_pts_half = int((num_pts + 1) / 2)
d_th = 2 * np.pi / num_pts
th = np.arange(0, 2 * np.pi, d_th)

freq_max = 2 / d_th
d_freq = freq_max / (num_pts - 1)
freq = np.linspace(0, freq_max, num_pts)
freq0 = freq[0]
freq1 = freq[1:num_pts_half]
freq2 = freq[num_pts_half:]

#%% TEST

JOHN = np.sin(8 * th)
fft1 = np.fft.fft(JOHN)
JOHN2 = ifft(fft1)

plt.figure(tight_layout=True, figsize=(9, 6))

plt.subplot(2, 1, 1)
plt.plot(th, JOHN, 'r.:')
plt.plot(th, JOHN2, 'b.:')

plt.subplot(2, 1, 2)
plt.plot(freq, np.abs(fft1), 'r.:')

#%% 

w = 0.05 * freq_max
freq_amp1 = np.exp(-(freq1**2 / w**2))
freq_amp = np.concatenate([np.array([1.0]), freq_amp1, freq_amp1[::-1]])

freq_ph1 = np.random.random(num_pts_half - 1) * np.pi
freq_ph = np.concatenate([np.array([0.0]), freq_ph1, -freq_ph1[::-1]])

freq_complex = freq_amp * np.exp(-1j * freq_ph)
freq_ifft = ifft(freq_complex)

plt.figure(tight_layout=True, figsize=(9, 6))
plt.plot(th, freq_ifft, 'r.:')
plt.plot(th + 2 * np.pi, freq_ifft, 'b.:')
