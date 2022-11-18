"""
Created on Tue Nov 15 21:09:07 2022 @author: john.obrecht
"""

import board
import neopixel
from matplotlib import cm
import numpy as np
from time import sleep

seq = np.load('repo/rainbow_02.npy')
num_pixels, _, num_steps = np.shape(seq)

pixels = neopixel.NeoPixel(board.D12, 650, brightness=1, pixel_order='RGB')

while True:
    for j in range(num_steps):
        pixels[:] = seq[:, :, j]
        pixels.show()
        sleep(0.01)

