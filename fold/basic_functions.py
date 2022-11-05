"""
Created on Fri Nov  4 12:57:51 2022 @author: john.obrecht
"""

import numpy as np

def cosd(x):
    return np.cos(x * np.pi / 180)

def sind(x):
    return np.sin(x * np.pi / 180)

def rx(th):
    th *= np.pi / 180
    return np.array([[1, 0, 0], [0, np.cos(th), np.sin(th)], [0, -np.sin(th), np.cos(th)]])
    
def ry(th):
    th *= np.pi / 180
    return np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])

def rz(th):
    th *= np.pi / 180
    return np.array([[np.cos(th), np.sin(th), 0], [-np.sin(th), np.cos(th), 0], [0, 0, 1]])

def dist(x, y, z, x0, y0, z0, i):
    return np.sqrt( (x - x0[i])**2 + (y - y0[i])**2 + (z - z0[i])**2)

def point(x, y, z, x0, y0, z0, sz, i):
    return np.exp(-(dist(x, y, z, x0, y0, z0, i))**2 / sz**2)

def wf_gaussian(x, x0, sz):
    return np.exp(-((x - x0)**2 / sz**2))

# Triangle through time coordinates
def wf_triangle(x, x0, w):
    # Normal triangle
    left = (x > x0 - w) & (x <= x0)
    right = (x > x0) & (x < x0 + w)
    y = np.zeros(np.shape(x))
    y[left] = 1 + (x[left] - x0) / w
    y[right] = 1 - (x[right] - x0) / w
    # Triangle > 1 --- when triangle goes off screen right
    if x0 + w > 1:
        right_2 = (x < np.mod(x0 + w, 1))
        y[right_2] = 1 - (x[right_2] - (x0 - 1)) / w
    # Triangle < 0 -- when triangle goes off screen left
    if x0 - w < 0:
        left_2 = (x > np.mod(x0 - w, 1))
        y[left_2] = 1 + (x[left_2] - (x0 + 1)) / w
    return y

# Decay through time coordinates
def wf_decay(x, x0, tau):
    y = np.zeros(np.shape(x))
    y[x >= x0] = np.exp(-(x[x >= x0] - x0) / tau)
    # Handling the continuity past x = 1    
    x2 = x + 1
    y2 = y.copy()
    y2[x2 >= x0] = np.exp(-(x2[x2 >= x0] - x0) / tau)
    y = np.max((y, y2), axis=0)   
    return y

# Decay through spatial coordinates
def wf_decay_2(x, x0, tau, lim):
    x0 = lim - x0
    y = np.zeros(np.shape(x0))
    y[x <= x0] = np.exp((x - x0[x <= x0]) / tau)
    # Handling the continuity past x = 1    
    x2 = x0 + lim
    y2 = y.copy()
    y2[x <= x2] = np.exp((x - x2[x <= x2]) / tau)
    y = np.max((y, y2), axis=0)  
    return y[::-1]

def wf_pulse(x, xu, xl):
    y = np.zeros(np.shape(xu))
    y[(xu > x) & (x > xl)] = 1
    y[(xu > x + 360) & (x + 360 > xl)] = 1  # Edge case
    y[(xu > x - 360) & (x - 360 > xl)] = 1  # Edge case
    return y
