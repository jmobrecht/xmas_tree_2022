"""
Created on Fri Oct 14 20:20:46 2022 @author: john.obrecht
Take a picture of a single-LED and find its x-y position on image
"""
    
#%% Grayscale --- works well.

import cv2
import numpy as np
import matplotlib.pyplot as plt

def take_picture(cam, gX, save=False):
    s, img = cam.read()
    g0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if s else gX
    if save:
        np.save('test', g0)
    return g0

def get_x_y(img, idx_x, idx_y, thr):
    g = img.copy()
    g[g<thr] = 0
    sum_x = np.sum(g, axis=0)
    sum_y = np.sum(g, axis=1)
    sum_sum = np.sum(sum_y) + 1E-6
    x = np.dot(sum_x, idx_y) / sum_sum
    y = np.dot(sum_y, idx_x) / sum_sum
    return x, y

def analyze_one_image(cam, gX, ind_x, ind_y, thr, pixel_no, rgb, show=False):
    
    # TODO Initiate single pixel on (perhaps run through R, G, & B colors for analysis?)
    
    g0 = take_picture(cam, gX)  # Take picture
    # g0 = np.load('test.npy')  # Load picture (debugging)
    x, y = get_x_y(g0, ind_x, ind_y, thr)  # Analyze grayscale image
    if show:
        plt.imshow(g0, cmap='bone')
        plt.plot(x, y, 'rx', markersize=8)
    return x, y

#%% Camera properties

cam = cv2.VideoCapture(1)   # 1 -> index of camera
sx, sy = (480, 640)  # np.shape(g)
ind_x = np.arange(0, sx, 1)
ind_y = np.arange(0, sy, 1)
gX = 255 * np.ones([sx, sy])
thr = 220

#%% 

import time
t0 = time.time()

pixel_no = 0
rgb = 255 * (1, 1, 1)
x, y = analyze_one_image(cam, gX, ind_x, ind_y, thr, pixel_no, rgb, True)

t1 = time.time()
print('Time: {:1.2f} ms'.format((t1 - t0) * 1E3))
