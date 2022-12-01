# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 21:49:14 2022

@author: john.obrecht
"""

import numpy as np
from scipy.fft import fft, ifft

kx = 1
ky = 1
kz = 1

x, y, z = np.meshgrid(np.arange(-10,10,1), np.arange(-10,10,1), np.arange(-10,10,1))

s = np.exp(-x**2 / kx**2 - y**2 / ky**2 - z**2 / kz**2)

ph = np.random.random([20, 20, 20]) * 2 * np.pi

JOHN = fft(s * np.exp(-1j * ph))
J2 = np.abs(JOHN)**2
