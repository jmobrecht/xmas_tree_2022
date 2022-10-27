"""
Created on Sun Oct 23 11:12:29 2022 @author: john.obrecht
"""

import re
import json
import os
import numpy as np
import matplotlib.pyplot as plt

folder = r'C:\Users\john.obrecht\Downloads\xmastree2020-main'
file = 'coords.txt'
path = os.sep.join([folder, file])
f = open(path, 'r')
tree = np.array(json.loads('[' + f.read().replace('\n', ',') + ']'), dtype='float')
num_pts = len(tree)

# Scale Z
tree[:, 2] -= np.min(tree[:, 2])
z_sc = np.max(tree[:, 2])
tree[:, 2] /= z_sc

# Scale X & Y by z-scale
tree[:, 0] -= np.mean(tree[:, 0])
tree[:, 1] -= np.mean(tree[:, 1])
tree[:, 0] /= z_sc
tree[:, 1] /= z_sc
r_sc = np.mean([-np.min(tree[:, 0]), np.max(tree[:, 0]), -np.min(tree[:, 1]), np.max(tree[:, 1])])
theta = np.arctan(r_sc / 1) * 180 / np.pi

# tree_range = [[np.min(tree[:, 0]), np.max(tree[:, 0])], [np.min(tree[:, 1]), np.max(tree[:, 1])], [np.min(tree[:, 2]), np.max(tree[:, 2])]]
tree_range = [[np.min(tree[:, 0]), np.max(tree[:, 0])], [np.min(tree[:, 1]), np.max(tree[:, 1])], [np.min(tree[:, 2]), np.max(tree[:, 2])]]

# Sort array order by column 2
tree = tree[np.argsort(tree[:, 2])]

#%% Set 3D Axes Equal - function

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

#%%

# Rainbow
from matplotlib import cm

steps = num_pts
color_map = cm.get_cmap('hsv', steps)
xp = color_map(np.linspace(0, 1, steps))
# xp = xp[:, 0:3]

# Sequence
num_frames = 50
seq1 = np.zeros([num_pts, 4, num_frames])
seq1[:, :, 0] = xp
for i in range(1, num_frames):
    seq1[:, :, i] = np.roll(seq1[:, :, i-1], 1, axis=0)

# Slice
h0 = np.linspace(np.max(tree[:, 2]), np.min(tree[:, 2]), num_frames)
def slice(z, i):
    return np.exp(-(z - h0[i])**2 / (0.01 * np.max(tree[:, 2]))**2)

seq2 = np.ones([num_pts, 4, num_frames])
for i in range(1, num_frames):
    # np.round(slice(tree[:,2], 0))
    seq2[:, 3, i] = np.round(slice(tree[:, 2], i))

# Rain
z0 = np.linspace(1, 0, num_frames)
y0 = np.linspace(0, 0, num_frames)
x0 = np.linspace(0, r_sc, num_frames)
def dist(x, y, z, i):
    return np.sqrt( (x - x0[i])**2 + (y - y0[i])**2 + (z - z0[i])**2)
def rain(x, y, z, i):
    return np.exp(-(dist(x, y, z, i))**2 / (0.05)**2)

seq3 = np.ones([num_pts, 4, num_frames])
for i in range(1, num_frames):
    seq3[:, 3, i] = np.round(rain(tree[:, 0], tree[:, 1], tree[:, 2], i))

#%% Animation

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import pandas as pd

frame_rate = 0.01

def update_color(i):
    graph._facecolors = seq3[:, :, i]
    title.set_text('Frame: {} of {}'.format(i, num_frames))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor((0.2, 0.2, 0.2))
title = ax.set_title('3D Test')

graph = ax.scatter(tree[:, 0], tree[:, 1], tree[:, 2], s=4, c = seq3[:, :, 0])
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

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection='3d')

ax.set_facecolor((0.2, 0.2, 0.2))
ax.scatter(tree[:, 0], tree[:, 1], tree[:, 2], s=4, c=xp, cmap='jet')
ax.set_xlim(np.min(tree[:, 0]), np.max(tree[:, 0]))
ax.set_ylim(np.min(tree[:, 1]), np.max(tree[:, 1]))
ax.set_zlim(np.min(tree[:, 2]), np.max(tree[:, 2]))
set_axes_equal(ax)
ax.axis('off')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.show()

#%%
