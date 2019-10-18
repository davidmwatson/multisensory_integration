#!/usr/bin/env python3

"""
Script runs simultaneous high and low-pass filtering of image, with an option
to delay one or both streams.  Presented image is an average of the high- and
low-pass versions, following the outputs of the delay lines. Uses parallel
processes to allow application of low and high-pass filters in 2 simultaneous
child processes. Would strongly recommend running the video feed in monochrome
to reduce processor load!
"""

from __future__ import division
import os, sys, cv2, multiprocessing, traceback
import numpy as np

# Custom imports
sys.path.append('../')
from imageprocessing import (fwhm2sigma, createFourierFilter,
                             createSoftWindowMask, pyfftw_FourierFilter)

# Py2 <--> Py3 compatibility imports
if sys.version_info.major == 3:
    import queue
else:
    import Queue as queue

"""
!! Further psychopy-based imports are located in "if __name__ == '__main__'"
block to provide protection from multiprocessing !!
"""

wisdom_default = os.path.realpath('./py37_fftw_752x480_mono_wisdom.pkl')


##### Custom function and class definitions ######
def force_quit():
    """
    Terminates video stream and all filterer processes, then exits program.

    Function uses extensive error handling as none of the global vars are
    guaranteed to be accessible or even exist when function gets called.
    """
    global stream, filterers, win

    # Stop stream
    try:
        stream.close()
    except:
        pass

    # Kill all child processes
    try:
        for f in filterers:
            try:
                # Stop call gives process chance to exit itself
                f.stop()
                # Wait a bit for process to exit
                f.join(timeout=2)
                # If it's still alive, terminate it
                if f.is_alive():
                    f.terminate()
            except:
                pass
    except:
        pass

    # Close window
    try:
        win.close()
    except:
        pass

    # Quit main process
    core.quit()



def doFilterDelay(frame, mask, filterer1, filterer2, delayline1, delayline2):
    """
    Main function for applying filtering and delay lines to frame.  May be
    passed to VideoStream's postproc argument.
    """
    # Pass image to filters
    filterer1.input_queue.put(frame.copy(), timeout=1.0)
    filterer2.input_queue.put(frame.copy(), timeout=1.0)

    # Retrieve filtered images
    filt_frame1 = filterer1.output_queue.get(timeout=1.0)
    filt_frame2 = filterer2.output_queue.get(timeout=1.0)

    # Pass through delay lines
    filt_frame1 = delayline1.collector(filt_frame1)
    filt_frame2 = delayline2.collector(filt_frame2)

    # Sum frames together, cast to uint8
    new_frame = (filt_frame1 + filt_frame2).clip(0,255).astype(np.uint8)

    # Combine with mask into RGBA image, return
    if new_frame.ndim == 3:  # RGB -> RGBA
        new_frame = cv2.merge([new_frame, mask])
    else:  # mono -> RGBA
        new_frame = cv2.merge(3*[new_frame] + [mask])
    return new_frame



class filterProcess(multiprocessing.Process):
    """
    Class wraps pyfftw_FourierFilter and applies it in a parallel process.
    """
    def __init__(self, name, **filter_kwargs):
        # Super call implements inheritance from multiprocessing.Process
        super(filterProcess, self).__init__(name = name)

        # Add local vars to class
        self.filter_kwargs = filter_kwargs

        # Events
        self.ready_event = multiprocessing.Event()
        self.trigger_event = multiprocessing.Event()
        self.stop_event = multiprocessing.Event()

        # Queues
        self.input_queue = multiprocessing.Queue(maxsize=1)
        self.output_queue = multiprocessing.Queue(maxsize=1)


    def run(self):
        """
        Overwrite multiprocessing.Process method.  Our function gets called
        in its place when the process's .start() method is called.

        Because further initialisation of pyfftw objects is done once process
        is started, the process uses the .ready_event object to signal the
        main process when it is ready to begin proper. Call the class's
        .trigger() method to start child process's main loop.  When done, call
        the class's .stop() method.
        """
        # Instantiate filtering class
        filterer = pyfftw_FourierFilter(**self.filter_kwargs)

        # Report ready, wait for trigger
        self.ready_event.set()
        self.trigger_event.wait()

        # Loop continously until told to stop
        while not self.stop_event.is_set():
            # Try to get frame from input queue - might fail (e.g. if user
            # has quit since last iteration)
            try:
                frame = self.input_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Filter frame, place into output queue
            self.output_queue.put(filterer.filter(frame))


    def trigger(self):
        """
        Send signal to child process to begin main loop.  Call this once the
        class's .ready_event object has confirmed the child process is ready.
        """
        self.trigger_event.set()


    def stop(self):
        """
        Send signal to child process to stop main loop.
        """
        # Set stop event
        self.stop_event.set()
        # Clear output queue to prevent main loop blocking at end
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                pass



#### Begin main script ####

# All further code must be wrapped in "if __name__ == '__main__'" block to
# stop multiprocessing raising RuntimeError on Windows.
if __name__ == '__main__':
    __spec__ = None  # fixes multiprocessing in spyder

    # Imports for main process only
    from psychopy import visual, core, event, gui
    from videostreaming import VideoDelayLine, FrameStim
    from videostreaming import OpenCV_VideoStream as VideoStream

    ### Present GUI to check setup ###
    # Fields with some default parameters
    info = {'FPS':'30',
            'Camera Resolution':'(752, 480)',
            'Camera Number':'1',
            'Screen Number':'1',
            'Vertical Reverse':True,
            'Colour Mode':['mono', 'bgr'],
            'Wisdom File':wisdom_default,
            'High Filter Mean':'0',
            'High Filter FWHM':'50',
            'High Filter Delay':'0',
            'Low Filter Mean':'0',
            'Low Filter FWHM':'20',
            'Low Filter Delay':'0.5',
            'Fullscreen':True}

    # Display dialogue
    dlg = gui.DlgFromDict(info, title = 'Setup',
                          order = ['FPS', 'Camera Resolution', 'Camera Number',
                                   'Screen Number', 'Vertical Reverse',
                                   'Colour Mode', 'Wisdom File',
                                   'High Filter Mean', 'High Filter FWHM',
                                   'High Filter Delay', 'Low Filter Mean',
                                   'Low Filter FWHM', 'Low Filter Delay',
                                   'Fullscreen'],
                          fixed = ['FPS'])
    if dlg.OK:
        # Extract details to local vars for brevity
        fps = float(info['FPS'])
        cam_res = info['Camera Resolution']
        # If cam_res looks like a tuple or list, evaluate it as such
        if ( cam_res.startswith('(') and cam_res.endswith(')') ) or \
           ( cam_res.startswith('[') and cam_res.endswith(']') ):
               cam_res = eval(cam_res)

        cam_num = int(info['Camera Number'])
        screen_num = int(info['Screen Number'])
        vertical_reverse = info['Vertical Reverse']
        colour_mode = info['Colour Mode']
        wisdom_filepath = info['Wisdom File']

        high_filter_mean = float(info['High Filter Mean'])
        high_filter_fwhm = float(info['High Filter FWHM'])
        high_filter_sigma = fwhm2sigma(high_filter_fwhm)
        high_filter_delay = float(info['High Filter Delay'])

        low_filter_mean = float(info['Low Filter Mean'])
        low_filter_fwhm = float(info['Low Filter FWHM'])
        low_filter_sigma = fwhm2sigma(low_filter_fwhm)
        low_filter_delay = float(info['Low Filter Delay'])

        fullscr = info['Fullscreen']

    else:
        print('User cancelled')
        core.quit()

    # Set up video stream
    stream = VideoStream(cam_num=cam_num, cam_res=cam_res, fps=fps,
                         vertical_reverse=vertical_reverse,
                         colour_mode=colour_mode)
    cam_res = stream.cam_res

    # Work out image size from cam res
    imsize = list(cam_res[::-1])
    if colour_mode in ['bgr','rgb']:
        imsize.append(3)

    # Create mask, scale into range 0-255 and cast to uint8 so we can use it
    # as image alphachannel
    mask = createSoftWindowMask(imsize=imsize, maskshape='rect')
    mask = (mask * 255).astype(np.uint8)

    # Create filters
    filters = [createFourierFilter(imsize=imsize, mode='sf',
                                   filter_type='gaussian', invert=True,
                                   filter_kwargs={'sigma':high_filter_sigma}),
               createFourierFilter(imsize=imsize, mode='sf',
                                   filter_type='gaussian', invert=False,
                                   filter_kwargs={'sigma':low_filter_sigma})
              ]

    # Set DC components of filters to 0.5 (instead of 1) - this halves mean
    # luminance after filtering so allowing us to add the images together
    for f in filters:
        f[0,0] = 0.5

    # Set up delaylines
    isColour = colour_mode in ['rgb', 'bgr']
    delaylines = [VideoDelayLine(delay=high_filter_delay, fps=fps,
                                 cam_res=cam_res, colour=isColour),
                  VideoDelayLine(delay=low_filter_delay, fps=fps,
                                 cam_res=cam_res, colour=isColour)
                 ]


    ### Begin setting up child processes.  Everything below now in try block
    ### so we can exit child processes if something goes wrong.
    try:
        # Create class instances for filter child processes
        filterers = [filterProcess('highfilterer', imsize=imsize,
                                   filt=filters[0], apply_mask=False,
                                   wisdom_filepath=wisdom_filepath,
                                   output_dtype=float, output_range=None),
                     filterProcess('lowfilterer', imsize=imsize,
                                   filt=filters[1], apply_mask=False,
                                   wisdom_filepath=wisdom_filepath,
                                   output_dtype=float, output_range=None)
                    ]

        # Start each process in turn, wait for it to report ready.  Do each in
        # turn to prevent both attempting to read / write the wisdom file at
        # the same time. While doing this, continuously check for error. Need
        # to use while loop to check event status rather than using event's
        # wait method as startup time could be long if pyfftw wisdom needs
        # generating, so there isn't a sensible timeout value for the wait
        for f in filterers:
            f.start()
            while not f.ready_event.is_set():
                if not f.is_alive():
                    print(f.name + ' child process has errored, exiting now')
                    force_quit()

        # Add filter func as postproc to stream
        postproc_kwargs = {'mask':mask,
                           'filterer1':filterers[0],
                           'filterer2':filterers[1],
                           'delayline1':delaylines[0],
                           'delayline2':delaylines[1]
                           }
        stream.setPostproc(doFilterDelay, postproc_kwargs)

        # Create psychopy window
        win = visual.Window(fullscr=fullscr, allowGUI=False, color=-1,
                            screen=screen_num)

        # Create framestim
        framestim = FrameStim(win)

        # Signal child processes to begin
        for f in filterers:
            f.trigger()
        # Wait a little bit to give child processes a chance to start proper
        core.wait(0.1)

        ### Begin main loop
        KEEPGOING = True
        while KEEPGOING:
            # Check if any child processes have errored
            for f in filterers:
                if not f.is_alive():
                    print(f.name + ' child process has errored, exiting now')
                    force_quit()

            # Get frame and display
            frame = stream.get_frame()
            framestim.setFrame(frame)
            framestim.draw()
            win.flip()

            # Quit if requested
            if len(event.getKeys(keyList=['escape','q'])) > 0:
                KEEPGOING = False


        ### Main loop done
        # Close window
        win.close()

        # Stop video stream
        stream.close()

        # Stop child processes.  Force terminate if necessary
        for f in filterers:
            f.stop()
            f.join(timeout = 3.0)
            if f.is_alive():
                print('Force terminating %s process' % f.name)
                f.terminate()
            print('%s process stopped' % f.name)

        # Exit
        print('\nDone\n')
        core.quit()


    except Exception:
        # Something in main process errored - alert user and exit
        print('\nMain process errored - ending child processes and exiting\n')
        traceback.print_exc()
        force_quit()
