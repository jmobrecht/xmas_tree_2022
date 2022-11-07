import imageio as iio
import board
import neopixel
import numpy as np
import pandas as pd
import time
import cv2
import matplotlib.pyplot as plt
from time import sleep
from fold.utils import *

# Camera properties
camera = iio.get_reader('<video0>')
meta = camera.get_meta_data()
delay = 1 / meta['fps']

# LED properties
rgb = (255, 255, 255)
num_pixels = 50
pixels = neopixel.NeoPixel(board.D12, num_pixels, brightness=1, pixel_order='RGB')

# Props
sx, sy = (768, 1024)  # np.shape(g)
ind_x = np.arange(0, sx, 1)
ind_y = np.arange(0, sy, 1)
thr = 235

# Variables
d0 = 1.1 # Initial delay
d1 = 0.5 # Pre-pixel-on / Post-camera / Post-all-off delay
d2 = 0.5 # Post-pixel-on delay (minimum: 0.3)
d3 = 0.0 # Post-all-off delay

#################################################################################

tA = time.time()

# Turn all pixels off (to reset)
pixels.fill((0, 0, 0))

# Take backrgound image (to be subtracted off)
sleep(d0)
bkg = camera.get_next_data()
#np.save('../data/Image_0X', bkg)
gx = cv2.cvtColor(bkg, cv2.COLOR_BGR2GRAY)
sleep(d0)

output_list = []
for i in range(num_pixels):

    sleep(d1)

    # Turn pixel on
    pixels[i] = rgb
    pixels.show()

    sleep(d2)
    
    # Capture image w/ pixel on (approx. 30 ms between "sleeps" here)
    img = camera.get_next_data()
    x, y = get_x_y(img, gx, ind_x, ind_y, thr)
#    np.save('../data/Image_{}'.format(str(i).zfill(2)), img)
    output_list.append({'pixel': i, 'x': x, 'y': y}) 
    print('Pixel {:03} - X: {:1.2f}, Y: {:1.2f}'.format(i, x, y))
    # Turn all pixels off (to reset)
    pixels.fill((0, 0, 0))

    sleep(delay) # minimum delay of the inverse frame rate needed
   
camera.close()

sleep(d3)

# Turn all pixels off (to reset)
pixels.fill((0, 0, 0))  # All off

tB = time.time()
print('Time Total: {:1.2f} s'.format((tB - tA) * 1E0))

df = pd.DataFrame(output_list)
df.set_index('pixel', inplace=True, drop=True)
df.to_csv('Output.csv')
