"""
Created on Tue Nov 15 21:09:07 2022 @author: john.obrecht
"""

import board
import neopixel
import argparse
import numpy as np
from matplotlib import cm
from time import sleep

def run_sequence(effect, brightness, delay):

    seq = np.load('repo/' + effect + '.npy')
    num_pixels, _, num_steps = np.shape(seq)
    pixels = neopixel.NeoPixel(board.D12, 650, brightness=brightness, pixel_order='RGB')       
    while True:
        for j in range(num_steps):
            pixels[:] = seq[:, :, j]
            pixels.show()
            sleep(delay)

parser = argparse.ArgumentParser()
parser.add_argument('--effect', default = 'rainbow_01', type = str)
parser.add_argument('--brightness', default = 1, type = float)
parser.add_argument('--delay', default = 0.001, type = float)

if __name__ == '__main__':
    args = parser.parse_args()
else:
    args = parser.parse_args('')

run_sequence(args.effect, args.brightness, args.delay)
