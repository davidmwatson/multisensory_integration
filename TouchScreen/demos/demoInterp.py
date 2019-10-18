#!/usr/bin/env python3

"""
Expanded version of test_touchscreen that demonstrates using cubic splines
to interpolate the touch co-ordinates over time.
"""

import sys
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Custom imports
sys.path.append('../')
from utils import TouchScreenReader

### Key details ###
monitorRate = 120
monitorInterval = 1/monitorRate



### Funcs ###

def colorline(x, y, z=None, ax=None, *args, **kwargs):
    """
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z

    Stolen from:
    https://nbviewer.jupyter.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(z.min(), z.max())

    kwargs['array'] = z
    kwargs['norm'] = norm
    lc = LineCollection(segments, *args, **kwargs)

    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)

    return lc


### Begin ###
# Setup reader and data lists
reader = TouchScreenReader()
T = []
X = []
Y = []

# Begin main loop
reader.flush()
print('Running demo\nPress Ctrl+C to quit')
while True:
    try:
        success, (t,x,y,ev_type) = reader.get_coords()
        if success:
            print(t[-1],x[-1],y[-1])  # just print last event in each array
            T.extend(t)
            X.extend(x)
            Y.extend(y)
    except KeyboardInterrupt:
        break

# Close port
reader.close()

# Convert lists to arrays
T = np.asarray(T)
T -= T[0]  # zero timepoints to first sample
X = np.asarray(X)
Y = np.asarray(Y)

# Interp
XY = np.c_[X, Y]
f = interp1d(T, XY, axis=0, kind='cubic', bounds_error=False, fill_value='nan')
interpT = np.arange(T.min(), T.max() + monitorInterval, monitorInterval)
interpXY = f(interpT)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
h = ax.scatter(X, Y, c=T, marker='.', cmap='jet')
colorline(interpXY[:,0], interpXY[:,1], interpT, linewidth=2, ax=ax, cmap='jet')
ax.set_xlim([-1920//2, 1920//2])
ax.set_ylim([-1080//2, 1080//2])
ax.set_aspect('equal')
cax = fig.colorbar(h)
cax.set_label('Time (s)')

plt.show()
