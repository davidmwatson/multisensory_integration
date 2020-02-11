#!/usr/bin/env python3

"""
Shows simple live video stream - useful when calibrating camera and / or
headset, or as a demo when familiarising new subjects with headset.

Options
-------
* Backend - whether to use the OpenCV or uEye SDK video streaming backends.
* FPS - frames per second to run camera at.
* Camera Resolution - (width, height) tuple or named camera resolution.
* Camera Number - Which camera to use; first available will be number 0.
* Screen Number - Which screen to use; first available will be number 0.
* Vertical Reverse - Whether to flip image updside-down, e.g. uEye camera
  running through OpenCV backend often returns image wrong way up, so this
  option allows the image to flipped back the right way.
* Image Rescale - How to fit the image to the window.
* Colour - Colour mode to use.
* Fullscreen - If True, will use fullscreen PsychoPy window.
* Fixation - If True, will overlay central fixation cross.  This is useful for
  marking the centre of the screen when calibrating the camera.
* Show FPS - If True, will display the working frame rate of the stream.
  This is calculated by taking a rolling mean of the time differences between
  the previous 10 window flips.  In addition, a plot of the flip times across
  the stream run will be produced at the end.
* Video Output - Can specify a path to an output file to record to.
* Output Codec - Codec to use for output file, if applicable.

Further details of these options can be found in the documentation for the
main video streaming classes in the utils module.

"""

from __future__ import division
import sys
import numpy as np
import matplotlib.pyplot as plt
from psychopy import visual, core, event, gui

# Custom imports
sys.path.append('../')
from videostreaming import (OpenCV_VideoStream, uEyeVideoStream,
                                  FrameStim, cam_res_lookup)

# Py2 <--> compatibility fixes
from past.builtins import unicode




###### Begin main script ######

### Present GUI to check setup
# Fields with some default parameters
info = {'Backend':['OpenCV', 'uEye'],
        'FPS':'30',
        'Camera Resolution':'(752, 480)',
        'Camera Number':'1',
        'Screen Number':'1',
        'Vertical Reverse':True,
        'Image Rescale':['resize','crop',None],
        'Colour':['mono', 'bgr'],
        'Fullscreen':False,
        'Fixation':False,
        'Show FPS':False,
        'Video Output':'',
        'Output Codec':'mp4v'}

# Display dialogue
dlg = gui.DlgFromDict(info, title = 'Setup',
                      order = ['Backend', 'FPS', 'Camera Resolution',
                               'Camera Number', 'Screen Number',
                               'Vertical Reverse', 'Image Rescale','Colour',
                               'Fullscreen','Fixation', 'Show FPS',
                               'Video Output', 'Output Codec'])
if dlg.OK:
    # Extract details to local vars for brevity
    backend = info['Backend']
    fps = float(info['FPS'])

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
    fixation = info['Fixation']
    show_fps = info['Show FPS']
    video_output = info['Video Output']
    output_codec = info['Output Codec']
else:
    print('User cancelled')
    core.quit()


### Initial setup

# Initialise psychopy window
win = visual.Window(allowGUI=False, fullscr=fullscr, screen=screen_num,
                    color=-1)

# Intialise video stream
if backend == 'OpenCV':
    stream = OpenCV_VideoStream(
            cam_num=cam_num, cam_res=cam_res, fps=fps, colour_mode=colour,
            vertical_reverse=vertical_reverse
            )
elif backend == 'uEye':
    if cam_res:
        if isinstance(cam_res, (str, unicode)):
            cam_res = cam_res_lookup[cam_res]
        aoi = [0,0] + list(cam_res)
    else:
        aoi = None
    stream = uEyeVideoStream(
            cam_num=cam_num, pixel_clock='max', fps=fps, aoi=aoi,
            colour_mode=colour, vertical_reverse=vertical_reverse
            )

# Open video writer if necessary
if video_output:
    stream.openVideoWriter(video_output, output_codec)
    stream.switchRecording()

# Set up framestim
framestim = FrameStim(win, rescale=im_rescale)

# Set up a fixation cross
diameter = 50
r = int(diameter / 2)
fix_objs = [visual.Line(win, (-r,0), (r,0), lineColor='red', lineWidth=2,
                        units='pix'),
            visual.Line(win, (0,-r), (0,r), lineColor='red', lineWidth=2,
                        units = 'pix'),
            visual.Circle(win, radius=r, lineColor='red', units='pix')]

# Set up text stim for fps (only used if showing FPS)
fps_txt = visual.TextStim(win, text='', pos=(-0.9,0.8), alignHoriz='left',
                          units='norm')

# List storing window flips (only used if showing FPS)
flip_times = []



### Stimulus presentation ###

# Begin main loop
KEEPGOING = True
while KEEPGOING:
    # Acquire and draw frame
    frame = stream.get_frame()
    framestim.setFrame(frame)
    framestim.draw()

    # Draw fixation if requested
    if fixation:
        for obj in fix_objs:
            obj.draw()

    # Update and draw FPS marker if requested
    if show_fps:
        with np.errstate(divide='ignore', invalid='ignore'):
            fps = 1 / np.mean(np.diff(flip_times[:-10]))
        fps_txt.setText('FPS = {:.02f}'.format(fps))
        fps_txt.draw()

    # Flip to display
    t = win.flip()
    if show_fps:
        flip_times.append(t)

    # Check if we need to stop
    if len(event.getKeys(keyList=['escape','q'])) > 0:
        print('User cancelled')
        win.close()
        stream.close()
        KEEPGOING = False

# Plot flip times if requested
if show_fps:
    flip_intervals = np.diff(flip_times)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

    # Raw flip times
    ax1.plot(flip_intervals)
    ax1.set_title('Flip intervals')
    ax1.set_ylabel('Flip interval (s)')
    ax1.set_xlabel('Frame N')

    # Histogram
    ax2.hist(flip_intervals, bins=50)
    ax2.set_title('Histogram')
    ax2.set_ylabel('Frequency')
    ax2.set_xlabel('Flip interval (s)')

    plt.show()


print('\nDone\n')
