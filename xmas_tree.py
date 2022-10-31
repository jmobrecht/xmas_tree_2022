"""
Created on Sun Oct 23 11:12:29 2022 @author: john.obrecht
"""

import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation
from mpl_toolkits.mplot3d import Axes3D
from fold.utils import set_axes_equal, get_tree_coords
from fold.sequenc_utils import rainbow, moving_slice, falling_rain, spiral, cone_01, cone_02, blink_01, breathe_01

# Get tree coordinates
path = r'C:\Users\john.obrecht\Downloads\xmastree2020-main\coords.txt'
tree = get_tree_coords(path)
num_pts = len(tree)

#%% Animation

def update_color(i):
    graph._facecolors = seq[:, :, i]
    title.set_text('Frame: {} of {}'.format(i, num_frames))

# Sequence properties
num_frames = 100

# Load sequence
seq = breathe_01(tree, num_pts, num_frames)

# Animation frame rate: how fast the animation progresses through sequence
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

#%%

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from fold.sequenc_utils import wf_triangle
t = np.linspace(0, 1, 500)
triangle = wf_triangle(t, 0.05, 0.1)
plt.plot(t, triangle)
