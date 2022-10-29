#from picamera import PiCamera
#import picamera
#from time import sleep
#import numpy as np
#
#cam = PiCamera()

#cam.start_preview()
#sleep(5)
#cam.stop_preview()

#cam.start_preview()
#sleep(5)
#cam.capture('/home/pi/Documents/image.jpg')
#cam.stop_preview()

#with picamera.PiCamera() as cam:
#    cam.resolution = (320, 240)
#    cam.framerate = 24
#    time.sleep(2)
#    output = np.empty((240, 320, 3), dtype=np.uint8)
##    cam.capture(output, 'rgb')

import matplotlib.pyplot as plt
import pygame
import pygame.camera
import pygame.image
from pygame.locals import *
import time

pygame.init()
pygame.camera.init()

camlist = pygame.camera.list_cameras()
if camlist:
    cam = pygame.camera.Camera(camlist[0], (640, 480))  # to 2592 x 1944 pix
    cam.start()
    t0 = time.time()
    
    image = cam.get_image()
    
    t1 = time.time()
    print('Time: {:1.2f} ms'.format((t1 - t0) * 1E3))
    cam.stop()
    
#    i2 = pygame.PixelArray(image)
#    plt.imshow(i2)
#    plt.show()