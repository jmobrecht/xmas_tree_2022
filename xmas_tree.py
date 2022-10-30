"""
Created on Sun Oct 23 11:12:29 2022 @author: john.obrecht
"""

import re
import os
import numpy as np
import matplotlib.pyplot as plt
from fold.utils import set_axes_equal, get_tree_coords
from fold.sequenc_utils import rainbow, moving_slice, falling_rain

#%%  Load tree & properties

folder = r'C:\Users\john.obrecht\Downloads\xmastree2020-main'
file = 'coords.txt'
path = os.sep.join([folder, file])

# Get tree coordinates
tree = get_tree_coords(path)
num_pts = len(tree)

# Radial scale
r_sc = np.mean([-np.min(tree[:, 0]), np.max(tree[:, 0]), -np.min(tree[:, 1]), np.max(tree[:, 1])])

# XYZ scale
tree_range = [[np.min(tree[:, 0]), np.max(tree[:, 0])], [np.min(tree[:, 1]), np.max(tree[:, 1])], [np.min(tree[:, 2]), np.max(tree[:, 2])]]

#%% Animation

def update_color(i):
    graph._facecolors = seq[:, :, i]
    title.set_text('Frame: {} of {}'.format(i, num_frames))

# Sequence properties
num_frames = 500

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import pandas as pd

seq = rainbow(tree, num_pts, num_frames)

frame_rate = 0.01

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor((0.2, 0.2, 0.2))
title = ax.set_title('3D Test')

graph = ax.scatter(tree[:, 0], tree[:, 1], tree[:, 2], s=4, c = seq[:, :, 0])
graph._offsets3d = (tree[:, 0], tree[:, 1], tree[:, 2])
ax.set_xlim(np.min(tree[:, 0]), np.max(tree[:, 0]))
ax.set_ylim(np.min(tree[:, 1]), np.max(tree[:, 1]))
ax.set_zlim(np.min(tree[:, 2]), np.max(tree[:, 2]))
set_axes_equal(ax)
ax.axis('off')
ani = matplotlib.animation.FuncAnimation(fig, update_color, num_frames, interval=frame_rate, blit=False)
plt.show()

# ani.save('rainbow.gif')

#%% Static View

# import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(projection='3d')

# ax.set_facecolor((0.2, 0.2, 0.2))
# ax.scatter(tree[:, 0], tree[:, 1], tree[:, 2], s=4, c=xp, cmap='jet')
# ax.set_xlim(np.min(tree[:, 0]), np.max(tree[:, 0]))
# ax.set_ylim(np.min(tree[:, 1]), np.max(tree[:, 1]))
# ax.set_zlim(np.min(tree[:, 2]), np.max(tree[:, 2]))
# set_axes_equal(ax)
# ax.axis('off')
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
# plt.show()

#%%
