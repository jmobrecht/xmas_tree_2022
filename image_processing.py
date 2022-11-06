"""
Created on Thu Oct 27 20:28:05 2022 @author: john.obrecht
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_x_y(img, idx_x, idx_y, thr):
    g = img.copy()
    g[g<thr] = 0
    sum_x = np.sum(g, axis=0)
    sum_y = np.sum(g, axis=1)
    sum_sum = np.sum(sum_y) + 1E-6
    x = np.dot(sum_x, idx_y) / sum_sum
    y = np.dot(sum_y, idx_x) / sum_sum
    return x, y

folder = r'C:\Users\john.obrecht\Downloads'
bkg = np.load(folder + os.sep + 'background.npy')  # Load picture (debugging)
img = np.load(folder + os.sep + 'image.npy')  # Load picture (debugging)
gB = cv2.cvtColor(bkg, cv2.COLOR_BGR2GRAY)
gI = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gI = gI.astype(float)
gB = gB.astype(float)

gI[gI == 0] = 1
gB[gB == 0] = 1

gM = gI - gB
gD = gM.copy()

sx, sy = (480, 640)  # np.shape(g)
ind_x = np.arange(0, sx, 1)
ind_y = np.arange(0, sy, 1)
gX = 255 * np.ones([sx, sy])
thr = 220

x, y = get_x_y(gM, ind_x, ind_y, thr)
gD[gD<thr] = 0

#%%

f, axarr = plt.subplots(2, 2)
axarr[0, 0].imshow(gB, cmap='bone')
axarr[0, 1].imshow(gI, cmap='bone')
axarr[1, 0].imshow(gM, cmap='bone')
axarr[1, 1].imshow(gD, cmap='bone')
axarr[1, 1].plot(x, y, 'rx', markersize=12)
