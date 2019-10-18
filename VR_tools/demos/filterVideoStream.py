#!/usr/bin/env python3

"""
Demonstration of running a real time spatial frequency filter of video feed.

Uses imageprocessing.pyfftw_FourierFilter class, which uses the pyfftw
module to implement optimised FFTs that are VERY fast (fast enough for 30fps,
and maybe even 60fps with a fast processor).  Recommend supplying a pyfftw
wisdom file if one is available otherwise script will take ~5mins to start!
Note - wisdom file must be correct for transform to be applied, e.g. for
correct image dimensions and python version. Would strongly recommend running
the video feed in monochrome to reduce processor load!
"""

from __future__ import division
import os, sys
from psychopy import visual, core, event, gui

# Custom imports
sys.path.append('../')
from videostreaming import OpenCV_VideoStream as VideoStream
from videostreaming import FrameStim
from imageprocessing import (fwhm2sigma, createFourierFilter,
                             pyfftw_FourierFilter)

wisdom_default = os.path.realpath('./py37_fftw_752x480_mono_wisdom.pkl')


###### Begin main script ######

### Present GUI to check setup ###
# Fields with some default parameters
info = {'fps':'30',
        'Camera Resolution':'(752, 480)',
        'Camera Number':'1',
        'Screen Number':'1',
        'Vertical Reverse':True,
        'Colour Mode':['mono','bgr'],
        'Wisdom File':wisdom_default,
        'Filter Mean':'0',
        'Filter FWHM':'50',
        'Invert Filter':True,
        'Fullscreen':True}

# Display dialogue
dlg = gui.DlgFromDict(info, title = 'Setup',
                      order = ['fps', 'Camera Resolution', 'Camera Number',
                               'Screen Number', 'Vertical Reverse',
                               'Colour Mode', 'Wisdom File', 'Filter Mean',
                               'Filter FWHM', 'Invert Filter', 'Fullscreen'],
                      fixed = ['fps'])
if dlg.OK:
    # Extract details to local vars for brevity
    fps = float(info['fps'])

    cam_res = info['Camera Resolution']
    # If it looks like a tuple or list, evaluate it as such
    if ( cam_res.startswith('(') and cam_res.endswith(')') ) or \
       ( cam_res.startswith('[') and cam_res.endswith(']') ):
           cam_res = eval(cam_res)

    cam_num = int(info['Camera Number'])
    screen_num = int(info['Screen Number'])
    vertical_reverse = info['Vertical Reverse']
    colour_mode = info['Colour Mode']
    wisdom_filepath = info['Wisdom File']
    filter_mean = float(info['Filter Mean'])
    filter_fwhm = float(info['Filter FWHM'])
    filter_sigma = fwhm2sigma(filter_fwhm)
    invert_filter = info['Invert Filter']
    fullscr = info['Fullscreen']
else:
    print('User cancelled')
    core.quit()

### Basic initialisation stuff ###

# Initialise stream
stream = VideoStream(
        cam_num=cam_num, fps=fps, cam_res=cam_res, colour_mode=colour_mode,
        vertical_reverse=vertical_reverse
        )
cam_res = stream.cam_res  # will assign from named resolutions too

# Determine image size from camera resolution
imsize = list(stream.cam_res[::-1])
if colour_mode in['rgb', 'bgr']:
    imsize.append(3)

# Create filter
filt = createFourierFilter(imsize=imsize, mode='sf',
                           filter_type='gaussian',
                           filter_kwargs={'mu':filter_mean,
                                          'sigma':filter_sigma},
                           invert=invert_filter, real=True)
# Set up filtering class
filterer = pyfftw_FourierFilter(imsize=imsize, filt=filt,
                                wisdom_filepath=wisdom_filepath)

# Add function into stream
stream.setPostproc(filterer.filter)

# Create window
win = visual.Window(fullscr=fullscr, color=-1, screen=screen_num,
                    allowGUI=False)

# Create framestim
framestim = FrameStim(win)

### Begin main execution ###

# Loop infinitely
KEEPGOING = True
while KEEPGOING:
    # Get frame, draw, display
    frame = stream.get_frame()
    framestim.setFrame(frame)
    framestim.draw()
    win.flip()

    # Check if we need to quit
    if len(event.getKeys(keyList = ['escape','q'])) > 0:
        print('User cancelled')
        win.close()
        stream.close()
        KEEPGOING = False
