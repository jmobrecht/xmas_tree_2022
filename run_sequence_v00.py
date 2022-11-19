"""
Created on Tue Nov 15 21:09:07 2022 @author: john.obrecht
"""

import board
import neopixel
from matplotlib import cm
import numpy as np
from time import sleep

seq = np.load('repo/sparkle_02.npy')
num_pixels, _, num_steps = np.shape(seq)

pixels = neopixel.NeoPixel(board.D12, 650, brightness=1, pixel_order='RGB')

while True:
    for j in range(num_steps):
        pixels[:] = seq[:, :, j]
        pixels.show()
<<<<<<< HEAD
        sleep(0.001)  # 0.5 good for rainbows... bad for breathe, sparkle
=======
        sleep(0.01)

>>>>>>> 7e11f1d1efff73ad95c8e219eee5d32e5656ea71
