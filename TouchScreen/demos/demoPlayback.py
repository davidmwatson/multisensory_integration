#!/usr/bin/env python3

"""
Uses cubic spline interpolation of touch inputs to play animation of
dot stimulus back.

Recording trial lasts forever until screen is touched, then will record inputs
for 2 seconds before ending trial and moving on to playback.
"""

import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from psychopy import visual, core, gui, event, monitors

# Local imports
sys.path.append('../')
from utils import TouchScreenReader, MyElementArrayStim


### Key vars ###

# Key paths
outdir = './logs'

# Stimuli details
nDots = 5000
dotSz = 4  # in pixels
dotDur = 10  # in frames
coherence = 0.4

# Timings
trialDur = 2

# Monitor details
monitorRate = 120
monitorInterval = 1/monitorRate
monitor = monitors.Monitor('Display++')


### Custom funcs ###
def allowQuit(keyList=['escape','q']):
    if event.getKeys(keyList=keyList):
        print('User cancelled')
        win.close()
        try:
            reader.close()
        except:
            pass
        core.quit()


### Begin setup ###

# Get info
infodict = {'Full Screen':True}
dlg = gui.DlgFromDict(infodict)
if not dlg.OK:
    print('User cancelled')
    core.quit()

fullscr = infodict['Full Screen']

# Open connection to touch screen
reader = TouchScreenReader()

# Open window
win = visual.Window(fullscr=fullscr, allowGUI=False, color=-1,
                    units='pix', monitor=monitor)
winWidth, winHeight = win.size

# Init element stim
elementStim = MyElementArrayStim(win, nElements=nDots, sizes=dotSz)

# Text stim for recording status
recordTxt = visual.TextStim(win, pos=(0, 0.9), height=0.15, units='norm', bold=True)

# Circle for illustrating finger position
touchCirc = visual.Circle(win, radius=15, fillColor='red', lineColor='red')

# Clock
clock = core.Clock()
staticTimer = core.StaticPeriod(monitorRate)



## Main loop ##
while True:
    # Wait to begin
    reader.flush()
    txt = visual.TextStim(win, 'Touch to start', height=50, wrapWidth=winWidth)
    txt.draw()
    win.flip()
    while True:
        allowQuit()
        success, res = reader.get_coords()
        xy = res[1:3, -1]
        if success and txt.contains(*xy):
            break
    win.flip()
    core.wait(0.5)


    ## Record ##
    recordTxt.setText('Record')
    recordTxt.setColor('red')

    touch_record = []
    touch_xy1 = None
    dxy = np.zeros(2)
    frameN = 0
    t1 = float('inf')  # make trial infinite till touch input

    reader.flush()
    t0 = clock.getTime()
    while (clock.getTime() - t1) < trialDur:
        # Check for quit signal
        allowQuit()

        # Randomly update all dots every <dotDur> frames
        if frameN % dotDur == 0:
            elementStim.update_dots()

        # Check for (most recent) touch input
        success, res = reader.get_coords()
        res = res[:, -1]  # most recent
        if success:
            # Append to record
            touch_record.append(res)

            # Set t1 on 1st touch to begin countdown
            if np.isinf(t1):
                t1 = clock.getTime()

            # Compute movement vector between this and previous touch
            touch_xy2 = res[1:3]
            if touch_xy1 is not None:  # not 1st touch
                dxy = touch_xy2 - touch_xy1
            touch_xy1 = touch_xy2

            # Update dots if necessary
            if any(dxy != 0):
                elementStim.update_dots(dxy, coherence)

            # Update circle
            touchCirc.setPos(touch_xy1)

        else:
            touch_xy1 = None

        # Display
        elementStim.draw()
        recordTxt.draw()
        if success:
            touchCirc.draw()
        t = win.flip()
        frameN += 1

    # Clear screen and start ISI. Use static timer, as we'll use ISI to
    # compute spline interp and set up playback
    win.flip()
    staticTimer.start(0.5)

    # Convert touch_record to pandas dataframe
    touch_record = pd.DataFrame(touch_record, columns=['t','x','y','press'])
    touch_record['t'] += t1 - t0 - touch_record.loc[0,'t']  # zero to trial start


    ## Playback ##
    recordTxt.setText('Play')
    recordTxt.setColor('green')

    # Interp touch record for each monitor frame
    T = touch_record['t'].values
    XY = touch_record[['x','y']].values

    fullTrialDur = T.min() + trialDur
    interp_T = np.arange(0, fullTrialDur + monitorInterval, monitorInterval)
    interp_T = interp_T[interp_T <= fullTrialDur]  # clip out of range vals

    if len(XY) > 3:  # min DoF for cubic spline
        fun = interp1d(T, XY, axis=0, kind='cubic', bounds_error=False, fill_value='nan')
        interp_XY = fun(interp_T)
    else:
        interp_XY = np.full([len(interp_T), 2], np.nan)

    # Run playback
    staticTimer.complete()
    touch_xy1 = None
    for frameN, touch_xy2 in enumerate(interp_XY):

        # Check for quit signal
        allowQuit()

        # Randomly update all dots every <dotDur> frames
        if frameN % dotDur == 0:
            elementStim.update_dots()

        # Only update on frames we have touch signal for
        success = all(~np.isnan(touch_xy2))
        if success:
            # Compute movement vector between this and previous touch
            if touch_xy1 is not None:  # not 1st touch
                dxy = touch_xy2 - touch_xy1
            touch_xy1 = touch_xy2

            # Update dots if necessary
            if any(dxy != 0):
                elementStim.update_dots(dxy, coherence)

            # Update circle
            touchCirc.setPos(touch_xy1)

        else:
            touch_xy1 = None

        # Display
        elementStim.draw()
        recordTxt.draw()
        if success:
            touchCirc.draw()
        t = win.flip()

    # Clear screen, wait ISI
    win.flip()
    core.wait(0.5)
