import board
import neopixel
from matplotlib import cm
import numpy as np
from time import sleep

def run_first_x_pixels(x, rgb):
    pixels = neopixel.NeoPixel(board.D12, 650, brightness=1, pixel_order='RGB')
    pixels.fill((0, 0, 0))
    sleep(2)
    pixels.fill(rgb)
    pixels.show()

def run_pixel_x(x, rgb):
    pixels = neopixel.NeoPixel(board.D12, 650, brightness=1, pixel_order='RGB')
    pixels.fill((0, 0, 0))
    sleep(2)
    pixels[x-1] = rgb
    pixels.show()
    
def run_pixel_x_fast(x, rgb):
    pixels = neopixel.NeoPixel(board.D12, 650, brightness=1, pixel_order='RGB')
    pixels.fill((0, 0, 0))
    pixels[x-1] = rgb
    pixels.show()

def all_off():
    pixels = neopixel.NeoPixel(board.D12, 650, brightness=0, pixel_order='RGB')
    pixels.fill((0, 0, 0))

def run_all(rgb):
    for i in range(650):
        run_pixel_x_fast(i, rgb)
        sleep(0.01)

def rainbow_cycle_2(wait, steps, num_pixels, xp):
    pixels = neopixel.NeoPixel(board.D12, num_pixels, brightness=1, pixel_order='RGB')
    for j in range(steps):
        pixels[:] = np.roll(xp, j, axis=0)
        pixels.show()
        sleep(wait)

steps = 650
color_map = cm.get_cmap('hsv', steps)
x = color_map(np.linspace(0, 1, steps))
xp = 255 * x[:, 0:3]
while True:
    rainbow_cycle_2(0.1, steps, 650, xp)
#run_first_x_pixels(650, (255, 255, 255))
#run_pixel_x(10, (255, 255, 255))
#run_all((255, 255, 255))
#all_off()
