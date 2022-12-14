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
os.chdir(r'C:\Users\john.obrecht\OneDrive - envision\Documents\GitHub\xmas_tree_2022')
from fold.utils import *
from fold.sequenc_utils import *
from fold.full_effects import *

# Get tree coordinates

# path = r'C:\Users\john.obrecht\OneDrive - envision\Documents\GitHub\xmastree2020-main\coords.txt'
# tree = get_tree_coords_0(path)
# tree = np.concatenate([tree, tree[:150, :]], axis=0)
tree = get_tree_coords('Master_Output.csv')
num_pts = len(tree)
rgb_warm = tuple(x / 255 for x in [255, 170, 70])

#%% Animation

# Load sequence
# seq = all_on(tree, num_pts, num_frames=10, rgb=(0.1,1,1))
# seq = rainbow_00(tree, num_pts, num_frames=100)
# seq = rainbow_01(tree, num_pts, num_frames=100)
# seq = rainbow_02(tree, num_pts, num_frames=100)
# seq = rainbow_03(tree, num_pts, num_frames=100)
# seq = rainbow_random(tree, num_pts, num_frames=100)
# seq = spiral_02(tree, num_pts, num_frames=250, rgb=((1,0,0), (0,1,0), (1,1,1)))
# seq = blink_00(tree, num_pts, num_frames=10, rgb=rgb_warm)
# seq = blink_01(tree, num_pts, num_frames=10)
# seq = blink_02(tree, num_pts, num_frames=10, rgb=((1,0,0), (0,1,0), (1,1,1)))
# seq = breathe_00(tree, num_pts, num_frames=250, rgb=rgb_warm)
# seq = breathe_01(tree, num_pts, num_frames=100)
# seq = breathe_02(tree, num_pts, num_frames=100, rgb=((1,0,0), (0,1,0), (1,1,1)))
# seq = sparkle_00(tree, num_pts, num_frames=250, rgb=rgb_warm)
# seq = sparkle_00b(tree, num_pts, num_frames=250, rgb=rgb_warm)
# seq = sparkle_01(tree, num_pts, num_frames=250, rgb=rgb_warm)
# seq = sparkle_01b(tree, num_pts, num_frames=250, rgb=rgb_warm)
# seq = sparkle_02(tree, num_pts, num_frames=250, rgb=rgb_warm)
# seq = sparkle_02b(tree, num_pts, num_frames=250, rgb=rgb_warm)
# seq = sparkle_00_R(tree, num_pts, num_frames=250)
# seq = sparkle_01_R(tree, num_pts, num_frames=250)
# seq = sparkle_02_R(tree, num_pts, num_frames=250)
# seq = stripes_V_00(tree, num_pts, num_frames=250, rgb=((1,0,0), (1,1,1)), stripes=6, thickness=1)
# seq = stripes_H_00(tree, num_pts, num_frames=250, rgb=((1,0,0), (1,1,1)), stripes=6, thickness=1)
# seq = face_00(tree, num_pts, num_frames=250, rgb=((1,0,0), (0.6,0.6,0.6)))
# seq = rain_00(tree, num_pts, num_frames=250, drops=10)
# seq = rain_01(tree, num_pts, num_frames=250, drops=12)
# seq = twilight_00(tree, num_pts, num_frames=100)
# seq = twilight_01(tree, num_pts, num_frames=100)
# seq = twilight_02(tree, num_pts, num_frames=100)
# seq = fill_00(tree, num_pts, num_frames=100)
# seq = fill_02(tree, num_pts)
# seq = camoflage_rainbow(tree, num_pts, num_frames=200)
# seq = gradual_00(tree, num_pts, num_frames=None, rgb=((1,1,0), (0,0,1)))
seq = gradual_S_00(tree, num_pts, num_frames=100, rgb=((1,1,1), rgb_warm))
# seq = coins(tree, num_pts, num_frames=100)

def update_color(i):
    graph._facecolors = seq[:, :, i]
    title.set_text('Frame: {} of {}'.format(i, num_frames))

# Animation frame rate: how fast the animation progresses through sequence
frame_rate = 0.01
num_pts, _, num_frames = np.shape(seq)

fig = plt.figure(tight_layout=True, figsize=(9,9))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor((0.2, 0.2, 0.2))
title = ax.set_title('3D Test')
graph = ax.scatter(tree[:, 0], tree[:, 1], tree[:, 2], s=18, c = seq[:, :, 0])
graph._offsets3d = (tree[:, 0], tree[:, 1], tree[:, 2])
ax.set_xlim(np.min(tree[:, 0]), np.max(tree[:, 0]))
ax.set_ylim(np.min(tree[:, 1]), np.max(tree[:, 1]))
ax.set_zlim(np.min(tree[:, 2]), np.max(tree[:, 2]))
set_axes_equal(ax)
ax.axis('off')
ani = matplotlib.animation.FuncAnimation(fig, update_color, num_frames, interval=frame_rate, blit=False)
plt.show()

# ani.save('rainbow.gif')
# np.save('repo/gradual_S_w', convert_rgba_to_rgb(seq))
