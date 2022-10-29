import cv2
import board
import neopixel
import numpy as np
from time import sleep

# Camera properties
cam = cv2.VideoCapture()   # 1 -> index of camera
sx, sy = (480, 640)  # np.shape(g)
ind_x = np.arange(0, sx, 1)
ind_y = np.arange(0, sy, 1)
gX = 255 * np.ones([sx, sy])
thr = 220

# LED properties
rgb = (255, 255, 255)
num_pixels = 50
pixels = neopixel.NeoPixel(board.D18, num_pixels, brightness=1, pixel_order='RGB')

#pixels[12] = rgb
#pixels.show()
#
#sleep(1)
#
#s, img = cam.read()
#np.save('image', img)

sleep(1)

pixels.fill((0, 0, 0))  # All off

sleep(1)

s, img = cam.read()
np.save('background', img)

sleep(1)

pixels.fill((0, 0, 0))  # All off

