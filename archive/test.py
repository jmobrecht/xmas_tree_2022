import matplotlib.pyplot as plt
import numpy as np
import cv2

def get_x_y(img, idx_x, idx_y, thr):
    g = img.copy()
    g[g<thr] = 0
    sum_x = np.sum(g, axis=0)
    sum_y = np.sum(g, axis=1)
    sum_sum = np.sum(sum_y) + 1E-6
    x = np.dot(sum_x, idx_y) / sum_sum
    y = np.dot(sum_y, idx_x) / sum_sum
    return x, y

sx, sy = (768, 1024)  # np.shape(g)
ind_x = np.arange(0, sx, 1)
ind_y = np.arange(0, sy, 1)
gX = 255 * np.ones([sx, sy])
thr = 150

iX = np.load('data/Image_0X.npy')
gX = cv2.cvtColor(iX, cv2.COLOR_BGR2GRAY)
gX = gX.astype(float)
gXp = gX - gX
xX, yX = get_x_y(gXp, ind_x, ind_y, thr)
gXp[gXp<thr] = 0

i0 = np.load('data/Image_00.npy')
g0 = cv2.cvtColor(i0, cv2.COLOR_BGR2GRAY)
g0 = g0.astype(float)
g0p = g0 - gX
x0, y0 = get_x_y(g0p, ind_x, ind_y, thr)
g0p[g0p<thr] = 0

i1 = np.load('data/Image_01.npy')
g1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
g1 = g1.astype(float)
g1p = g1 - gX
x1, y1 = get_x_y(g1p, ind_x, ind_y, thr)
g1p[g1p<thr] = 0

i2 = np.load('data/Image_02.npy')
g2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
g2 = g2.astype(float)
g2p = g2 - gX
x2, y2 = get_x_y(g2p, ind_x, ind_y, thr)
g2p[g2p<thr] = 0

i3 = np.load('data/Image_03.npy')
g3 = cv2.cvtColor(i3, cv2.COLOR_BGR2GRAY)
g3 = g3.astype(float)
g3p = g3 - gX
x3, y3 = get_x_y(g3p, ind_x, ind_y, thr)
g3p[g3p<thr] = 0

i4 = np.load('data/Image_04.npy')
g4 = cv2.cvtColor(i4, cv2.COLOR_BGR2GRAY)
g4 = g4.astype(float)
g4p = g4 - gX
x4, y4 = get_x_y(g4p, ind_x, ind_y, thr)
g4p[g4p<thr] = 0

#########################################################################

f, axarr = plt.subplots(1, 6, tight_layout=True, figsize=(25, 5))

axarr[0].imshow(gX, cmap='bone')
axarr[1].imshow(g0, cmap='bone')
axarr[2].imshow(g1, cmap='bone')
axarr[3].imshow(g2, cmap='bone')
axarr[4].imshow(g3, cmap='bone')
axarr[5].imshow(g4, cmap='bone')

plt.show()

#########################################################################

#f, axarr = plt.subplots(1, 6, tight_layout=True, figsize=(25, 5))
#
#ms = 6
#axarr[0].imshow(gXp, cmap='bone')
#axarr[0].plot(xX, yX, 'rx', markersize=ms)
#axarr[1].imshow(g0p, cmap='bone')
#axarr[1].plot(x0, y0, 'rx', markersize=ms)
#axarr[2].imshow(g1p, cmap='bone')
#axarr[2].plot(x1, y1, 'rx', markersize=ms)
#axarr[3].imshow(g2p, cmap='bone')
#axarr[3].plot(x2, y2, 'rx', markersize=ms)
#axarr[4].imshow(g3p, cmap='bone')
#axarr[4].plot(x3, y3, 'rx', markersize=ms)
#axarr[5].imshow(g4p, cmap='bone')
#axarr[5].plot(x4, y4, 'rx', markersize=ms)
#
#plt.show()

#########################################################################