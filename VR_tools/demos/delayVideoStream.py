#!/usr/bin/env python3

"""
Script runs video feed from camera, with option of a delay period.  Image is
displayed within a fullscreen PsychoPy window.


Usage instructions
------------------
Running script presents dialog with following options:
* Delay: The desired delay of the feed, specified in seconds.  Defaults to 0.
    Note that this requires the fps to be specified correctly.
* fps - Frames per second of the camera acquisition.  Currently we seem unable
    to increase this over 50 for the uEye camera, and then this doesn't divide
    into the refresh rate of the sensics headset, so for the moment this is
    fixed at 30 (the default of the camera anyway).
* Camera Resolution - (width, height) tuple specifying the resolution of the
    camera in pixels, or named camera.
* Camera Number - Index of camera device.
* Screen Number - Index of the display device.  0 will be the laptop monitor,
    and 1 will be the first external display device which should be the headset
    (if it is connected), so this is the default.
* Vertical Reverse - Choose whether to flip image updside-down or not before
    displaying.
* Image Rescale - How to rescale image to fit display device.
* Colour - Whether to display in colour or not.
* Fullscreen - whether to present in fullscreen mode or not
* Inherent Delay - The known inherent delay of the system.

"""

import sys
from psychopy import visual, core, event, gui
# Custom imports
sys.path.append('../')
from videostreaming import FrameStim, VideoDelayLine
from videostreaming import OpenCV_VideoStream as VideoStream


###### Begin main script ######

### Present GUI to check setup
# Fields with some default parameters
info = {'Delay (s)':'0',
        'fps':'30',
        'Camera Resolution':'(752, 480)',
        'Camera Number':'1',
        'Screen Number':'1',
        'Vertical Reverse':True,
        'Image Rescale':['resize','crop',None],
        'Colour':['mono','bgr'],
        'Fullscreen':True,
        'Inherent Delay (s)':'0.1'}

# Display dialogue
dlg = gui.DlgFromDict(info, title = 'Setup',
                      order = ['Delay (s)', 'fps', 'Camera Resolution',
                               'Camera Number', 'Screen Number',
                               'Vertical Reverse', 'Image Rescale','Colour',
                               'Fullscreen', 'Inherent Delay (s)'],
                      fixed = ['fps'])
if dlg.OK:
    # Extract details to local vars for brevity
    delay = float(info['Delay (s)'])
    fps = float(info['fps'])

    cam_res = info['Camera Resolution']
    # If it looks like a tuple or list, evaluate it as such
    if ( cam_res.startswith('(') and cam_res.endswith(')') ) or \
       ( cam_res.startswith('[') and cam_res.endswith(']') ):
           cam_res = eval(cam_res)

    cam_num = int(info['Camera Number'])
    screen_num = int(info['Screen Number'])
    vertical_reverse = info['Vertical Reverse']
    im_rescale = info['Image Rescale']
    colour = info['Colour']
    fullscr = info['Fullscreen']
    inherent_delay = float(info['Inherent Delay (s)'])
else:
    print('User cancelled')
    core.quit()


### Initial setup

# Initialise psychopy window
win = visual.Window(allowGUI=False, color=-1, fullscr=fullscr,
                    screen=screen_num)

# Intialise video stream
stream = VideoStream(cam_num=cam_num, cam_res=cam_res, fps=fps,
                     vertical_reverse=vertical_reverse, colour_mode=colour)

# Set up delay line if necessary
if delay > 0:
    # Instantiate class
    delay_line = VideoDelayLine(delay=delay, cam_res=stream.cam_res,
                                colour=colour, inherent_delay=inherent_delay,
                                delay_units='seconds', fps=fps)

    # Add function to stream
    stream.setPostproc(delay_line.collector)

# Set up framestim
framestim = FrameStim(win, rescale=im_rescale)


### Stimulus presentation ###

# Begin main loop
KEEPGOING = True
while KEEPGOING:
    # Acquire and draw frame
    frame = stream.get_frame()
    framestim.setFrame(frame)
    framestim.draw()

    # Flip to display
    win.flip()

    # Check if we need to stop
    for key in event.getKeys(keyList = ['escape','q']):
        if key:
            print('User cancelled')
            win.close()
            stream.close()
            KEEPGOING = False
