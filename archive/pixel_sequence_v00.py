import imageio as iio
import board
import neopixel
import numpy as np
import time
from time import sleep
import matplotlib.pyplot as plt

# Camera properties
camera = iio.get_reader('<video0>')
meta = camera.get_meta_data()
delay = 1 / meta['fps']

# LED properties
rgb = (255, 255, 255)
num_pixels = 350
pixels = neopixel.NeoPixel(board.D12, num_pixels, brightness=1, pixel_order='RGB')

# Variables
d0 = 0.1 # Initial delay
d1 = 0.1 # Pre-pixel-on / Post-camera / Post-all-off delay
d2 = 0.3 # Post-pixel-on delay (minimum: 0.3)
d3 = 0.0 # Post-all-off delay

#################################################################################

tA = time.time()

# Turn all pixels off (to reset)
pixels.fill((0, 0, 0))

# Take backrgound image (to be subtracted off)
sleep(d0)
bkg = camera.get_next_data()
np.save('../data/Image_0X', bkg)
sleep(d0)

for i in range(5): #num_pixels):

    sleep(d1)

    # Turn pixel on
    pixels[i] = rgb
    pixels.show()

    sleep(d2)
    
    # Capture image w/ pixel on (approx. 30 ms between "sleeps" here)
    img = camera.get_next_data()
    np.save('../data/Image_{}'.format(str(i).zfill(2)), img)
    # Turn all pixels off (to reset)
    pixels.fill((0, 0, 0))

    sleep(delay) # minimum delay of the inverse frame rate needed
   
camera.close()

sleep(d3)

# Turn all pixels off (to reset)
pixels.fill((0, 0, 0))  # All off

tB = time.time()
print('Time Total: {:1.2f} s'.format((tB - tA) * 1E0))
