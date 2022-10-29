import numpy as np
import cv2
import matplotlib.pyplot as plt

bkg = np.load('background.npy')  # Load picture (debugging)
img = np.load('image.npy')  # Load picture (debugging)

gB = cv2.cvtColor(bkg, cv2.COLOR_BGR2GRAY)
gI = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(gB, cmap='bone')
plt.show()
