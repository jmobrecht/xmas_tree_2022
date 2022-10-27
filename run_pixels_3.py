import board
import neopixel
from time import sleep

def wheel(pos):
    # Input a value 0 to 255 to get a color value.
    # The colours are a transition r - g - b - back to r.
    if pos < 0 or pos > 255:
        r = g = b = 0
    elif pos < 85:
        r = int(pos * 3)
        g = int(255 - pos * 3)
        b = 0
    elif pos < 170:
        pos -= 85
        r = int(255 - pos * 3)
        g = 0
        b = int(pos * 3)
    else:
        pos -= 170
        r = 0
        g = int(pos * 3)
        b = int(255 - pos * 3)
    return (r, g, b)

def rainbow_cycle(wait, num_pixels):
    import time
    for j in range(255):
        t0 = time.time()
        for i in range(num_pixels):
            pixel_index = (i * 256 // num_pixels) + j
            pixels[i] = wheel(pixel_index & 255)
        pixels.show()
        t1 = time.time()
        print('Time: {:1.2f} ms'.format((t1 - t0) * 1E3))
        sleep(wait)

def rainbow_cycle_2(wait, steps, num_pixels, xp):
    import time

    t0 = time.time()
    
    for j in range(steps):
        pixels[:] = np.roll(xp, j, axis=0)
        pixels.show()
        sleep(wait)
    
    t1 = time.time()
    print('Time: {:1.2f} ms'.format((t1 - t0) * 1E3))
    

pixels = neopixel.NeoPixel(board.D18, 50, brightness=1, pixel_order='RGB')

from matplotlib import cm
import numpy as np
steps = 500
color_map = cm.get_cmap('hsv', steps)
x = color_map(np.linspace(0, 1, steps))
xp = 255 * x[:, 0:3]

while True:
    # Comment this line out if you have RGBW/GRBW NeoPixels
    pixels.fill((255, 0, 0))
    # Uncomment this line if you have RGBW/GRBW NeoPixels
    # pixels.fill((255, 0, 0, 0))
    pixels.show()
    sleep(1)

    # Comment this line out if you have RGBW/GRBW NeoPixels
    pixels.fill((0, 255, 0))
    # Uncomment this line if you have RGBW/GRBW NeoPixels
    # pixels.fill((0, 255, 0, 0))
    pixels.show()
    sleep(1)

    # Comment this line out if you have RGBW/GRBW NeoPixels
    pixels.fill((0, 0, 255))
    # Uncomment this line if you have RGBW/GRBW NeoPixels
    # pixels.fill((0, 0, 255, 0))
    pixels.show()
    sleep(1)

    rainbow_cycle_2(0.1, 500, 50, xp)  # rainbow cycle with 1ms delay per step
