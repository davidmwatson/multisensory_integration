#!/usr/bin/env python3

"""
Demonstrates RDK stimulus and touch screen inputs. Basically runs like
a trial of the main experiment, but it lasts forever.
"""

import sys
import numpy as np
from psychopy import visual, core, gui, event, monitors

# Local imports
sys.path.append('../')
from utils import TouchScreenReader, RDK


### Key vars ###

# Key paths
outdir = './logs'

# Stimuli details
nDots = 5000
dotSz = 4  # in pixels
dotDur = 10  # in frames
coherence = 0.4

# Monitor details
monitorRate = 120
monitor = monitors.Monitor('Display++')

### Custom funcs ###
def rotateVector(xy, theta):
    """Rotate vector xy by angle theta (in radians)"""
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return R @ np.asarray(xy)

### Begin setup ###

# Get info
infodict = {'Full Screen':True, 'Rotation':0}
dlg = gui.DlgFromDict(infodict)
if not dlg.OK:
    print('User cancelled')
    core.quit()

fullscr = infodict['Full Screen']
theta = np.deg2rad(float(infodict['Rotation']))

# Open connection to touch screen
reader = TouchScreenReader()

# Open window
win = visual.Window(fullscr=fullscr, allowGUI=False, color=-1,
                    units='pix', monitor=monitor)
winWidth, winHeight = win.size

# Init RDK
rdkStim = RDK(win, nElements=nDots, sizes=dotSz)


## Present stimulus

# Init some vars
touch_xy1 = None
dxy = np.zeros(2)
frameN = 0

# Begin main loop
while True:
    # Check for quit signal
    if event.getKeys(keyList=['escape','q']):
        break

    # Randomly update all dots every <dotDur> frames
    if frameN % dotDur == 0:
        rdkStim.update_dots()

    # Check for (most recent) touch input
    success, res = reader.get_coords()
    res = res[:, -1]  # get most recent
    if success:
        # Compute movement vector between this and previous touch
        touch_xy2 = res[1:3]
        if touch_xy1 is not None:  # not 1st touch
            dxy = rotateVector(touch_xy2 - touch_xy1, theta)
        touch_xy1 = touch_xy2

        # Update dots if necessary
        if any(dxy):
            rdkStim.update_dots(dxy, coherence)
    else:
        touch_xy1 = None

    # Display
    rdkStim.draw()
    t = win.flip()
    frameN += 1


## Finish
win.close()
reader.close()

