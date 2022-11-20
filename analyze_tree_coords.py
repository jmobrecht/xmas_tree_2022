"""
Created on Sat Nov 19 12:16:37 2022 @author: john.obrecht
"""

import os
import numpy as np
import pandas as pd
os.chdir(r'C:\Users\john.obrecht\OneDrive - envision\Documents\GitHub\xmas_tree_2022')
from fold.utils import *

def cosd(th):
    return np.cos(np.pi / 180 * th)

def sind(th):
    return np.sin(np.pi / 180 * th)

def dist(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

#%%

folder = r'C:\Users\john.obrecht\OneDrive - envision\Documents\GitHub\xmas_tree_2022'
file_list = [x for x in os.listdir(folder) if x.endswith('.csv') and x.startswith('Output')]

clock_dict = {
    '0000': 0,
    '0130': 45,
    '0300': 90,
    '0430': 135,
    '0600': 180,
    '0730': 225,
    '0900': 270,
    '1030': 315,
    }

df_x = pd.DataFrame()
df_z = pd.DataFrame()
for file in file_list:
    clock = file[file.find('_') + 1:-4]
    tmp = pd.read_csv(folder + os.sep + file)
    df_x[clock] = tmp['x']
    df_z[clock] = tmp['y']

df_z.replace(0, float('NaN'), inplace=True)
df_x.replace(0, float('NaN'), inplace=True)

#%%

df_master_out = pd.DataFrame(columns=['x', 'y', 'z'])
# df_master_in = pd.read_csv(folder + os.sep + 'master.csv')

#%%

colors = 'rgbkcymr'
import matplotlib.pyplot as plt

plt.figure(tight_layout=True, figsize=(16, 12))
for i, col in enumerate(df_z.columns):
    plt.subplot(3, 3, i + 1)
    plt.plot(df_x[col], df_z[col], colors[i], marker='.', markersize=8, linewidth=0)
    plt.xlim([250, 650])
    plt.ylim([0, 700])
    plt.axis('equal')
    
for i, col in enumerate(df_z.columns):
    plt.subplot(3, 3, 9)
    plt.plot(df_x[col], df_z[col], colors[i], marker='.', markersize=8, linewidth=0)
    plt.xlim([250, 650])
    plt.ylim([0, 700])
    plt.axis('equal')

#%% Stats & metrics

z_stats = pd.DataFrame()
z_stats['max'] = df_z.max()
z_stats['median'] = df_z.median()
z_stats['min'] = df_z.min()
z_stats['range'] = z_stats['max'] - z_stats['min']

z_range = z_stats['range'].median()
z_min = z_stats['min'].median()

x_stats = pd.DataFrame()
x_stats['max'] = df_x.max()
x_stats['median'] = df_x.median()
x_stats['min'] = df_x.min()
x_stats['range'] = x_stats['max'] - x_stats['min']

x_median = x_stats['median'].median()

#%% Normalize

df_z = 1 - (df_z - z_min) / z_range
df_x = -(df_x - x_median) / z_range
df_y = df_x.copy()

plt.figure(tight_layout=True, figsize=(16, 12))
for i, col in enumerate(df_z.columns):
    plt.subplot(3, 3, i + 1)
    plt.plot(df_x[col], df_z[col], colors[i], marker='.', markersize=8, linewidth=0)
    plt.xlim([250, 650])
    plt.ylim([0, 700])
    plt.axis('equal')
    
for i, col in enumerate(df_z.columns):
    plt.subplot(3, 3, 9)
    plt.plot(df_x[col], df_z[col], colors[i], marker='.', markersize=8, linewidth=0)
    plt.xlim([250, 650])
    plt.ylim([0, 700])
    plt.axis('equal')

#%% Z

df_zp = df_z.copy()
count = df_zp.count(axis=1)
df_zp['median'] = df_zp.median(axis=1)
df_zp['count'] = count

df_master_out['z'] = df_zp['median']

#%% X

df_xp = pd.DataFrame()
for col in df_x.columns:
    if col in ['0000', '0600']:
        df_xp[col] = cosd(clock_dict[col]) * df_x[col]
count = df_xp.count(axis=1)
df_xp['median'] = df_xp.median(axis=1)
df_xp['count'] = count

# X replacement

df_xp2 = pd.DataFrame()
for col in df_x.columns:
    if col in ['0130', '0730']:
        df_xp2[col] = cosd(clock_dict[col]) * df_x[col]
count = df_xp2.count(axis=1)
df_xp2['median'] = df_xp2.median(axis=1)
df_xp2['count'] = count

df_xp3 = pd.DataFrame()
for col in df_x.columns:
    if col in ['0430', '1030']:
        df_xp3[col] = cosd(clock_dict[col]) * df_x[col]
count = df_xp3.count(axis=1)
df_xp3['median'] = df_xp3.median(axis=1)
df_xp3['count'] = count

# Combine

df_master_out['x'] = df_xp['median'].fillna(df_xp2['median'].fillna(df_xp3['median']))

#%% Y

df_yp = pd.DataFrame()
for col in df_y.columns:
    if col in ['0300', '0900']:
        df_yp[col] = sind(clock_dict[col]) * df_y[col]
count = df_yp.count(axis=1)
df_yp['median'] = df_yp.median(axis=1)
df_yp['count'] = count

# Y replacement

df_yp2 = pd.DataFrame()
for col in df_y.columns:
    if col in ['0130', '0730']:
        df_yp2[col] = sind(clock_dict[col]) * df_y[col]
count = df_yp2.count(axis=1)
df_yp2['median'] = df_yp2.median(axis=1)
df_yp2['count'] = count

df_yp3 = pd.DataFrame()
for col in df_y.columns:
    if col in ['0430', '1030']:
        df_yp3[col] = sind(clock_dict[col]) * df_y[col]
count = df_yp3.count(axis=1)
df_yp3['median'] = df_yp3.median(axis=1)
df_yp3['count'] = count

# Combine

df_master_out['y'] = df_yp['median'].fillna(df_yp2['median'].fillna(df_yp3['median']))

#%% Interpolate

df_master_out.interpolate(axis=0, inplace=True)

# Dimension the data & calculate distance between points

df_master_out[['xp', 'yp', 'zp']] = df_master_out[['x', 'y', 'z']] * 72  # Convert to inches
df_tmp = df_master_out[['xp', 'yp', 'zp']].copy()
df_tmp[['xp2', 'yp2', 'zp2']] = df_tmp.shift(1)
df_master_out['dist'] = df_tmp.apply(lambda x: dist(x['xp'], x['yp'], x['zp'], x['xp2'], x['yp2'], x['zp2']), axis=1)

#%% Plot X, Y, Z w/ color & size by point distance

plt.figure(tight_layout=True, figsize=(16, 12))
plt.subplot(3, 1, 1)
plt.plot(df_master_out['x'], 'r.:', linewidth=0.5, markersize=0)
plt.scatter(df_master_out.index, df_master_out['x'], 3 * df_master_out['dist'], df_master_out['dist'], cmap='jet')
plt.ylabel('X')
plt.title('Before Adjustment')
plt.subplot(3, 1, 2)
plt.plot(df_master_out['y'], 'g.:', linewidth=0.5, markersize=0)
plt.scatter(df_master_out.index, df_master_out['y'], 3 * df_master_out['dist'], df_master_out['dist'], cmap='jet')
plt.ylabel('Y')
plt.subplot(3, 1, 3)
plt.plot(df_master_out['z'], 'b.:', linewidth=0.5, markersize=0)
plt.scatter(df_master_out.index, df_master_out['z'], 3 * df_master_out['dist'], df_master_out['dist'], cmap='jet')
plt.ylabel('Z')

#%% Export for manual adjustment

# df_master_out.to_csv('Master_Output.csv')
df_master_in = pd.read_csv('Master_Output.csv')

df_master_in[['xp', 'yp', 'zp']] = df_master_in[['x', 'y', 'z']] * 72  # Convert to inches
df_tmp = df_master_in[['xp', 'yp', 'zp']].copy()
df_tmp[['xp2', 'yp2', 'zp2']] = df_tmp.shift(1)
df_master_in['dist'] = df_tmp.apply(lambda x: dist(x['xp'], x['yp'], x['zp'], x['xp2'], x['yp2'], x['zp2']), axis=1)

#%% Plot X, Y, Z w/ color & size by point distance

plt.figure(tight_layout=True, figsize=(16, 12))
plt.subplot(3, 1, 1)
plt.plot(df_master_in['x'], 'r.:', linewidth=0.5, markersize=0)
plt.scatter(df_master_in.index, df_master_in['x'], 3 * df_master_in['dist'], df_master_in['dist'], cmap='jet')
plt.ylabel('X')
plt.title('Before Adjustment')
plt.subplot(3, 1, 2)
plt.plot(df_master_in['y'], 'g.:', linewidth=0.5, markersize=0)
plt.scatter(df_master_in.index, df_master_in['y'], 3 * df_master_in['dist'], df_master_in['dist'], cmap='jet')
plt.ylabel('Y')
plt.subplot(3, 1, 3)
plt.plot(df_master_in['z'], 'b.:', linewidth=0.5, markersize=0)
plt.scatter(df_master_in.index, df_master_in['z'], 3 * df_master_in['dist'], df_master_in['dist'], cmap='jet')
plt.ylabel('Z')

#%%

from mpl_toolkits import mplot3d

fig = plt.figure(tight_layout=True, figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')
graph = ax.scatter(df_master_in['x'], df_master_in['y'], df_master_in['z'], s=4, c = (1, 0, 0))
graph._offsets3d = (df_master_in['x'], df_master_in['y'], df_master_in['z'])
ax.plot3D(df_master_in['x'], df_master_in['y'], df_master_in['z'], 'gray')
ax.set_xlim(df_master_in['x'].min(), df_master_in['x'].max())
ax.set_ylim(df_master_in['y'].min(), df_master_in['y'].max())
ax.set_zlim(df_master_in['z'].min(), df_master_in['z'].max())
set_axes_equal(ax)
# ax.axis('off')
plt.show()