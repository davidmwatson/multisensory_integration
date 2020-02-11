#!/usr/bin/env python3
"""
Runs a simple unaltered video stream on Oculus display via a combination
of the PsychoPy [1] and PsychXR [2] libraries.

!!! Updated for PsychXR v0.2.1+, which contains the necessary bug fixes for
ASW that prevent display shaking while head-locked.

Requirements
------------
* Psychopy >= 2020.1.0    (1st version with support for PsychXR v0.2)
* PsychXR >= 0.2.1    (1st version with ASW bug fix)
* Recommended Python == 3.6. PsychXR can be installed on Python > 3.6 but will
  need compiling from source. If using Python 3.6, PsychXR can be installed
  much more easily via pip.
  
References
----------
[1] Peirce et al. (2019). PsychoPy2: experiments in behavior made easy.
    Behavior Research Methods. DOI:10.3758/s13428-018-01193-y
    https://link.springer.com/article/10.3758%2Fs13428-018-01193-y

[2] Cutone, M. D. & Wilcox, L. M. (2019). PsychXR (Version 0.2.0) [Software].
    Available from https://github.com/mdcutone/psychxr
"""

import sys
from psychopy import visual, event, gui, core

# local imports
sys.path.append('../')
from videostreaming import OpenCV_VideoStream, uEyeVideoStream, FrameStim


### Key settings ###

# Params for each camera backend (will choose which backend at runtime)
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

uEye2_settings = {'pixel_clock':50,
                  'fps':30,
                  'colour_mode':'bgr',
                  'block':False,
                  'exposure':5,
                  'auto_exposure':False,
                  'auto_gain_control':'software',
                  'auto_white_balance':'software',
                  'colour_correction':True}




### Begin main script ###
    
# Runtime settings
config = {'Backend':['uEye2','uEye1','opencv'], 'Magnification':0.5}
dlg = gui.DlgFromDict(config, title='Runtime Settings')
if not dlg.OK:
    print('User cancelled')
    core.quit()
backend = config['Backend']
mag_factor = config['Magnification']

if backend == 'opencv':
    camera_settings = opencv_settings
elif backend == 'uEye1':
    camera_settings = uEye1_settings
elif backend == 'uEye2':
    camera_settings = uEye2_settings

# Open hmd window
hmd = visual.Rift(monoscopic=True, headLocked=True, color=-1, 
                  warnAppFrameDropped=False)

# Open video stream
if backend in ['uEye1','uEye2']:
    stream = uEyeVideoStream(**camera_settings)
else:
    stream = OpenCV_VideoStream(**opencv_settings)

# Set up framestim
disp_size = [x * mag_factor for x in hmd._hmdBufferSize]
framestim = FrameStim(hmd, rescale='resize', display_size=disp_size,
                      interpolate=True)

# Begin main loop
KEEPGOING = True
while KEEPGOING:
    # We have to update tracking on each frame (regardless of head-locking)
    trackingState = hmd.getTrackingState()
    hmd.calcEyePoses(trackingState.headPose.thePose)
    hmd.setDefaultView()
    
    # Update frame & display
    framestim.frame = stream.get_frame()
    framestim.draw()
    hmd.flip()

    # Check for quit events
    if len(event.getKeys(keyList=['escape','q'])) > 0:
        KEEPGOING = False

# Shut down
stream.close()
hmd.close()

