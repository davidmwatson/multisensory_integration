#!/usr/bin/env python3

"""
Simple demo of reading touch coords from screen.
"""

import sys
import matplotlib.pyplot as plt
import numpy as np

# Custom imports
sys.path.append('../')
from utils import TouchScreenReader


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

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
h = ax.scatter(X, Y, c=T, marker='.', cmap='jet')
ax.set_xlim([-1920//2, 1920//2])
ax.set_ylim([-1080//2, 1080//2])
ax.set_aspect('equal')
cax = fig.colorbar(h)
cax.set_label('Time (s)')
plt.show()
