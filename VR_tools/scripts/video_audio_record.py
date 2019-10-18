#!/usr/bin/env python3

"""
Script provides methods for running video and audio recording in parallel.

Classes
-------
* VideoRecord - runs video capture in separate thread
* AudioRecord - runs audio capture in separate thread


Functions
---------
* doRecord - wrapper function for above classes - recommended that you mostly
  use this rather than the classes directly


See class / function help for further details and usage examples.

If the script is run directly, then a commandline interface to the doRecord
function is also provided

If you want to combine the video and audio outputs into a single file,
consider using ffmpeg (commandline tool), e.g.
    ffmpeg.exe -i video.mp4 -i audio.wav -c:v copy -c:a aac combined.mp4

See also mux_video_audio.ps1 - Powershell script for muxing video and audio
streams together.

"""

import sys, traceback, cv2, threading

# Custom imports
sys.path.append('../')
from videostreaming import OpenCV_VideoStream, uEyeVideoStream
from audiostreaming import AudioStream

# Py2 <--> Py3 compatibility fixes
from past.builtins import raw_input


class VideoRecord(threading.Thread):
    """
    Wrapper around the video streaming classes in the utils module, optimised
    for recording frames to a file.  Execution happens in a parallel thread so
    that recording can occur in the background whilst control of the main
    thread is retained by the user.  An option is provided for choosing which
    video streaming class to use as the backend here.

    Parameters
    ----------
    video_output : str, required
        Filepath to desired output video. Recommend using '.mp4' file
        extension (default if no extension provided).
    codec : str or -1, optional
        Fourcc (https://www.fourcc.org/codecs.php) code indicating codec
        to use for encoding video output. Codec must be appropriate for
        file type; recommend using 'mp4v' for MP4 files (default).
    backend : str {'opencv' (default) | 'uEye'}, optional
        Which video streaming class to use as the backend.
    wire : bool, optional
        If True, live feed of video will be displayed on screen.
    lr_flip_display : bool, optional
        If True, will left-right flip the display of the live feed on screen.
        Note this only affects the display - NOT the frame that is recorded
        to file.  Can be useful for avoiding working with a "mirror image"
        when setting the shot up.  Ignored if wire is false.
    **stream_kwargs
        Additional keyword arguments are passed to the selected video streaming
        backend.  See documentation for those classes for further details.

    Example usage
    -------------
    Begin by creating an instance of the class

    >>> video_thread = videoRecord('myvideo.mp4')

    Once initialisation has finished, you can call the thread's .start() method
    to begin the stream.

    >>> video_thread.start()

    This will execute in a separate thread to the main thread, meaning control
    of the terminal will be returned back to you.  You can start and end the
    recording using the .switchRecording() method. You may do this multiple
    times, although the cuts between recording sessions will be evident in the
    saved video.  The stream starts with the recording status set to OFF, so
    the first call to .switchRecording() will set the recording ON.

    >>> video_thread.switchRecording() # start recording
    >>> ### Do some stuff ###
    >>> video_thread.switchRecording() # stop recording

    When you are finished, call the .close() method to end the thread.

    >>> video_thread.close()

    """
    def __init__(self, output_file, backend='opencv', codec='mp4v', wire=True,
                 lr_flip_display=True, **stream_kwargs):
        # Assign local args into class
        self.output_file = output_file
        self.backend = backend
        self.codec = codec
        self.wire = wire
        self.lr_flip_display = lr_flip_display

        # Other internal variables
        self.KEEPGOING = True # should thread continue executing?

        # Instantiate parent threading class
        super(VideoRecord, self).__init__(name='VideoRecord')

        # Set up stream backend
        if self.backend == 'opencv':
            self.stream = OpenCV_VideoStream(**stream_kwargs)
        elif self.backend == 'uEye':
            if not stream_kwargs.has_key('pixel_clock'):
                stream_kwargs['pixel_clock'] = 'max'
            if not stream_kwargs.has_key('block'):
                stream_kwargs['block'] = True
            self.stream = uEyeVideoStream(**stream_kwargs)
        else:
            raise ValueError("backend must be one of 'opencv' or 'uEye'")

        # Create alias to stream's switch recording function for convenience
        self.switchRecording = self.stream.switchRecording

        # Work out mid-point of image (needed for fixation if we do a wire)
        self.midpoint = (int(self.stream.cam_res[0]//2),
                         int(self.stream.cam_res[1]//2))

        # Open video writer
        self.stream.openVideoWriter(self.output_file, self.codec)

        # If wiring, clear any existing opencv windows
        if self.wire:
            cv2.destroyAllWindows()


    def run(self):
        """
        Overwrite default Thread.run method - our version will get called in
        its place when the thread's .start() method is called.

        Set capture going.  Will display on screen if wire == True and will
        also write frames out to file recording status is ON.  Stops when
        KEEPGOING == False.
        """
        ## Loop continuously until we're told not to. Wrap everything in
        ## try-block so that we can close thread if something errors
        while self.KEEPGOING:
            try:
                # Acquire frame.  If recording status is active, this will
                # also automatically write frame out via video writer
                frame = self.stream.get_frame()

                # If wiring feed, display on screen
                if self.wire and frame is not None:

                    # Convert RGB -> BGR for display if necessary
                    if self.stream.colour_mode == 'rgb':
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # Overlay cross
                    if self.stream.colour_mode != 'mono':
                        c = (0,0,255) # red in BGR space
                    else:
                        c = 255 # white
                    cv2.line(img = frame, color = c, thickness = 1,
                             pt1 = (self.midpoint[0], self.midpoint[1]-15),
                             pt2 = (self.midpoint[0], self.midpoint[1]+15))
                    cv2.line(img = frame, color = c, thickness = 1,
                             pt1 = (self.midpoint[0]-15, self.midpoint[1]),
                             pt2 = (self.midpoint[0]+15, self.midpoint[1]))

                    # L/R flip if necessary
                    if self.lr_flip_display:
                        frame = cv2.flip(frame, 1)

                    # Display
                    cv2.imshow('Video Feed', frame)
                    cv2.waitKey(1) # needed to refresh display

            # Something errored - break out of main loop
            except Exception:
                self.close()
                traceback.print_exc()

        ## Main loop ended - clean up and finish
        try:  # if stream totally dead then trying to close it may error too
            self.stream.close()
        except Exception:
            traceback.print_exc()

        if self.wire:
            cv2.destroyAllWindows()


    def close(self):
        """Stop stream permanently."""
        self.KEEPGOING = False



class AudioRecord(AudioStream):
    """
    Class provides methods for recording audio to a file.  Process is near
    identical to that for the AudioStream class in the utils module, so here
    we simply sub-class that.  Usage is identical to that class, with the
    following exceptions:
        * The output_file argument is now mandatory
        * The live wire option defaults to False.

    Arguments
    ---------
    output_file : str, required
        Path to desired output file.
    wire : bool, optional
        Whether to playback a live wire of the audiostream.
    **kwargs
        Additional keyword arguments are passed to the
        utils.audiostreaming.AudioStream base class.

    """
    def __init__(self, output_file, wire=False, **kwargs):
        kwargs['output_file'] = output_file
        kwargs['wire'] = wire
        super(AudioRecord, self).__init__(**kwargs)





def doRecord(video_output='test_video.mp4', audio_output='test_audio.wav',
             video_kwargs={}, audio_kwargs={}):
    """
    Wrapper function for controlling video and audio stream classes.  Once
    running, press Enter to turn recording on / off, or type q then press
    Enter to finish.

    Parameters
    ----------
    video_output : str (filepath) or None, optional
        Path to desired video output file, recommend using .mp4 extension.
        If None, video stream will not be run.
    audio_output : str (filepath) or None, optional
        Path to desired audio output file, recommend using .wav extension
        If None, audio stream will not be run.
    video_kwargs : dict, optional
        Dict of further keyword arguments to pass to videoRecord class
        Ignored if video_output is None.
    audio_kwargs : dict, optional
        Dict of further keyword arguments to pass to audioRecord class
        Ignored if audio_output is None.


    Example usage
    -------------
    >>> doRecord(video_output='myvideo.mp4', audio_output='myaudio.wav',
    ...          audio_kwargs={'wire':True, 'blocksize':0, 'latency':'low'})

    """
    # Print instructions
    print('\n' + '#' * 79 + '\n' + \
          'Click back in terminal window\n' + \
          'Press Enter to change recording status\n' + \
          'Type q then Enter to quit\n' + \
          ('#' * 79) + '\n')

    # Initialise streams
    streams = []
    if video_output:
        vid = VideoRecord(output_file=video_output, **video_kwargs)
        streams.append(vid)
    else:
        vid = None

    if audio_output:
        aud = AudioRecord(output_file=audio_output, **audio_kwargs)
        streams.append(aud)
    else:
        aud = None

    for s in streams:
        s.start()


    # Begin recording period - loop infinitely
    keepgoing = True
    while keepgoing:
        # Block till Enter is pressed
        key = raw_input()

        # Quit if asked
        if key == 'q':
            print('User cancelled')

            # Switch recording off on all streams
            if vid and vid.stream.RECORDING:
                vid.switchRecording()
            if aud and aud.RECORDING:
                aud.switchRecording()

            # Stop streams.  Vidstream executes in parallel thread and it's
            # best to let this finish closing before attempting to do anything
            # else, so "join" the thread.
            if vid:
                vid.close()
                vid.join(timeout=5)
            if aud:
                aud.close()

            # Break main loop
            keepgoing = False

        # Other key press, switch recording status
        else:
            for s in streams:
                s.switchRecording()




def main():
    """
    Wrapper for the doRecord wrapper function!  Uses argparse to provide
    commandline access to doRecord.  This is a workaround for the bug
    where opencv crashes when attempting to re-acquire the uEye for a second
    time - by calling the script from the commandline we can re-initialise
    the entire python script every time (this is insanely inefficient but
    I haven't got any better ideas...)

    Command-line flags are provided for most of the videoRecord and audioRecord
    class options, prepended with 'vid-' and 'aud-' repsectively.  If flag not
    given, option will default to  whatever the default for the videoRecord /
    audioRecord class is.  If relevant ?-output flag not given for video or
    audio then that particular stream will not be run.

    """
    import argparse

    ### Set up parser
    parser = argparse.ArgumentParser(
            description=main.__doc__,
            formatter_class=argparse.RawTextHelpFormatter
            )

    # video flags
    parser.add_argument('--vid-output')
    parser.add_argument('--vid-backend')
    parser.add_argument('--vid-codec')
    parser.add_argument('--vid-wire', action='store_true')
    parser.add_argument('--vid-lr-flip-display', action='store_true')
    parser.add_argument('--vid-cam-num', type=int)
    parser.add_argument('--vid-fps', type=float)
    parser.add_argument('--vid-cam-res', nargs=2, type=int)
    parser.add_argument('--vid-colour-mode')

    # audio flags
    parser.add_argument('--aud-output')
    parser.add_argument('--aud-device', nargs='+', type=int)
    parser.add_argument('--aud-wire', action='store_true')
    parser.add_argument('--aud-samplerate', type=int)
    parser.add_argument('--aud-blocksize', type=int)
    parser.add_argument('--aud-dtype')
    parser.add_argument('--aud-channels', nargs='+', type=int)
    parser.add_argument('--aud-latency', nargs='+', type=float)

    ### Parse arguments
    # If no args given, print help and exit
    if not len(sys.argv) > 1:
        parser.print_help()
        sys.exit(0)

    # Parse, convert results to dict
    args = vars(parser.parse_args())

    # Pop filenames (pop them so the keys aren't still there for kwargs check)
    video_output = args.pop('vid_output')
    audio_output = args.pop('aud_output')

    # For remaining args, we allocate them to dicts for video and audio
    # kwargs if (and only if) values provided for them
    video_kwargs = {}
    audio_kwargs = {}
    for k in args.keys():
        if k.startswith('vid_') and args[k] is not None:
            # Special handling for cam_res, needs to be tuple
            if k == 'vid_cam_res':
                args[k] = tuple(args[k])
            video_kwargs[k.replace('vid_', '')] = args[k]

        if k.startswith('aud_') and args[k] is not None:
            # Some special handling for device, channels, latency args. May be
            # provided as single value or pair of values; if single value then
            # we extract that value from the list
            if k in ['aud_device', 'aud_channels', 'aud_latency'] \
            and len(args[k]) == 1:
                args[k] = args[k][0]
            audio_kwargs[k.replace('aud_', '')] = args[k]

    ### Run
    doRecord(video_output, audio_output, video_kwargs, audio_kwargs)


### Execute main function if script called from commandline
if __name__ == '__main__':
    main()
