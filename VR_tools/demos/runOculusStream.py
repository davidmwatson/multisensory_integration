"""
Runs a simple unaltered video stream on Oculus display

Requires psychxr module, and PsychoPy > v3
"""

import sys
from psychopy import visual, event

# local imports
sys.path.append('../')
from videostreaming import OpenCV_VideoStream, uEyeVideoStream, FrameStim


### Key settings

backend = 'opencv'  # choose from: opencv, uEye1, uEye2

mag_factor = 0.5

opencv_settings = {'fps':30,
                   'colour_mode':'mono',
                   'vertical_reverse':False,
                   'cam_num':0}

uEye1_settings = {'pixel_clock':'max',
                  'fps':90,
                  'colour_mode':'mono',
                  'block':False,
                  'auto_exposure':'camera',
                  'auto_gain_control':'camera'}

uEye2_settings = {'pixel_clock':68,
                  'fps':45,
                  'colour_mode':'bgr',
                  'block':False,
                  'exposure':6,
                  'auto_exposure':False,
                  'auto_gain_control':'software',
                  'auto_white_balance':'software',
                  'colour_correction':True}

if backend == 'opencv':
    camera_settings = opencv_settings
elif backend == 'uEye1':
    camera_settings = uEye1_settings
elif backend == 'uEye2':
    camera_settings = uEye2_settings


### Begin main script ###

# Open hmd window
hmd = visual.Rift(monoscopic=True, color=-1,  warnAppFrameDropped=False)


# Open video stream
if backend in ['uEye1','uEye2']:
    stream = uEyeVideoStream(**camera_settings)
else:
    stream = OpenCV_VideoStream(**opencv_settings)

# Set up framestim
disp_size = [x * mag_factor for x in hmd.size]
framestim = FrameStim(hmd, rescale='resize', display_size=disp_size,
                      interpolate=True)

# Begin main loop
timestamps = []
KEEPGOING = True
while KEEPGOING:
    # Update frame
    frame = stream.get_frame()
    framestim.frame = frame
    framestim.draw()
    t = hmd.flip()
    timestamps.append(t)

    # Check for quit events
    if len(event.getKeys(keyList=['escape','q'])) > 0:
        KEEPGOING = False

# Shut down
stream.close()
hmd.close()


import numpy as np
import matplotlib.pyplot as plt
frameDiffs = 1/np.diff(timestamps)
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,6))
ax1.plot(frameDiffs)
ax2.hist(frameDiffs, bins=50)
fig.show()
