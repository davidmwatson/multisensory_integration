#!/usr/bin/env python3

"""
Classes and functions for performing audio streaming
"""

from __future__ import division
import os, sys
import numpy as np
import sounddevice as sd
import soundfile as sf

# Determine platform - used for brevity in other platform specific code later
if any(sys.platform.startswith(x) for x in ['win','cygwin']):
    platform = 'windows'
elif sys.platform.startswith('linux'):
    platform = 'linux'
elif sys.platform.startswith('darwin'):
    platform = 'mac'
else:
    platform = None

# Platform specific imports for volume control.  All require non-standard
# modules, so allow for import errors
if platform == 'windows':
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        have_pycaw = True
    except ImportError:
        have_pycaw = False
elif platform == 'linux':
    try:
        import alsaaudio
        have_alsa = True
    except ImportError:
        have_alsa = False
elif platform == 'mac':
    try:
        import osax
        have_osax = True
    except ImportError:
        have_osax = False

# Default preferred audio API by platform
if platform == 'windows':
    default_api = 'WASAPI'
elif platform == 'linux':
    default_api = 'ALSA'
elif platform == 'mac':
    default_api = 'Core Audio'



##### Function definitions #####
def get_device4api(api=default_api):
    """
    Helpful function for returning default input and output devices from
    sounddevice list of available devices for specified api.  You can call
    sounddevice.query_hostapis() to list available APIs.  String does not have
    to be an exact match for names listed here, the string simply has to be
    contained within the name (case does not matter).  Default APIs are
    platform specific and are given by the default_api variable in this script.

    Returns an (input, output) tuple of device indices.
    """
    # Loop through all listed available apis
    for d in sd.query_hostapis():
        # Test for match - convert everything to lowercase to permit case
        # insensitive search
        if api.lower() in d['name'].lower():
            # Get input and output device indices, return
            input_idx = d['default_input_device']
            output_idx = d['default_output_device']
            return (input_idx, output_idx)

    # If we reach this point we failed to find a match, so error
    raise RuntimeError('Failed to find suitable API')




##### CLASS DEFINITIONS #####
class AudioStream(object):
    """
    Class provides methods for running audio stream, with options of playing
    back a live wire and / or writing audio stream to file.  It is basically
    a wrapper around sounddevice.Stream with a few extra bells and whistles
    tagged on.

    Arguments
    ---------
    api : str, optional
        Preferred API for input and output devices.  Will attempt to use
        default devices for specified API.  See get_device4api function
        for more details. Ignored if device keyword argument is provided.
    wire : bool, optional
        If True (default), will present a live wire of audio input over
        specified audio output device.  WARNING - if the audio output is over
        speakers then this may cause audio feedback!  If False, will present
        silence instead.
    ouput_file : str, optional
        Filepath to desired output audio file. Stream will be written to file.
        Recommend using .wav extension. Note that stream recording status
        defaults to Off when first initialised. Use the .switchRecording
        method to start recording.
    record_delay : bool, optional
        If True, and if stream blocksize > 0, and if recording to output file,
        then start of recording will be padded with <blocksize> zeros
        (i.e. silence). Idea is that if using a non-zero blocksize to
        deliberately delay the audio, then this will be reflected in the
        recording too. If not wanting a delay (e.g. have just increased the
        blocksize to ease up on the processor) then set this to False.
    file_kwargs : dict, optional
        Dictionary of keyword arguments to be passed to soundfile.SoundFile
        class. Ignored if output_file is None. See the class documention for
        full details, but a summary of the more important arguments is given
        here:
            * samplerate : int
                Sampling acquisition rate in Hz.  Defaults to samplerate of
                stream.
            * channels : int
                Number of channels.  Defaults to number of output channels in
                stream.
            * subtype : str
                Datatype for output.  Default depends on format of output file
                (specified directly, or determined from extension).  Likely to
                be PCM_16 (signed int16) for WAV files.
    overwrite_file : bool, optional
        If True, and the specified output file already exists, the file will
        be overwritten. If False (default), an error will be raised instead.
    **stream_kwargs
        Additional optional keyword arguments are passed to sounddevice.Stream
        (if wire is True) or sounddevice.InputStream (if wire is False).
        Defaults for most options can be found within the sounddevice.default
        object.  See the class documentation for full details, but a summary
        of the more important arguments is given here:
            * samplerate : int or (input, output) tuple of ints
                Sampling acquisition rate in Hz.  Likely to default to 44100.
            * blocksize : int
                Number of frames to pass to each call of the callback
                function.  Likely to default to 0, which is a special value
                that indicates to make the blocks as small as possible given
                the hardware limitations.  Could set this to a larger value
                if intending to introduce a delay into the audio stream.
            * device : int or (input, output) tuple of ints
                Indices for hardware devices.  If wire is False, may pass just
                input device index.  Class can guess values instead if
                preferred API is given.  Call sounddevice.query_devices() to
                list available devices.
            * channels : int or (input, output) tuple of ints
                Number of channels for input and / or output devices.
                Will default to maximum allowable for each device.
            * latency : 'low', 'high', float, or (input, output) tuple thereof
                Desired latency of audio stream in seconds.  Special values of
                'low' or 'high' select default latencies for given devices;
                call sounddevice.query_devices(device_idx) for exact values.
                Lower latencies are desirable, but may incur higher processing
                costs.  Likely to default to 'high'.  If intending to do a
                live wire of the stream, it would be recommended to set this
                to 'low'

    Methods
    -------
    .close
        Alias to stream's close method, but will also close output file if
        relevant.  Stops stream permanently.
    .start
        Alias to stream's start method.  Starts stream.
    .stop
        Alias to stream's stop method.  Temporarily stops / pauses stream.
    .switchRecording
        Only relevant if output_file is provided.  Switches recordng status
        On/Off.  Status defaults to Off when class is first initialised.

    Example usage
    -------------
    Create an instance of the AudioStream class.  We'll do a live wire
    and so will request the lowest possible latencies.  We'll also record the
    stream to an output file.  The recording status defaults to OFF when the
    class is initialised, so we'll also call .switchRecording to turn it ON.

    >>> audiostream = AudioStream(wire=True, latency='low',
    ...                           output_file='test.wav')
    >>> audiostream.switchRecording()

    Call the .start method to begin the stream.

    >>> audiostream.start()

    If you want to pause the stream again at any point, call the .stop method.
    Call the .start method again to resume the stream.

    >>> audiostream.stop()  # pause
    >>> ### Do some stuff ###
    >>> audiostream.start()  # resume

    Once you are done, call the .close method to stop the stream permanently.
    This will also close our output file.

    >>> audiostream.close()

    If you want to introduce a delay into the audio feed, you can increase the
    blocksize to equal the size of the delay.  For instance, if we want a 0.5s
    delay, then at 44100 samples/second we would use a blocksize of
    44100 * 0.5 = 22050.  If we want to include this delay in a recorded file,
    we'll also need to set record_delay to True.

    >>> samplerate = 44100
    >>> delay = 0.5  # in seconds
    >>> blocksize = int(samplerate * delay)
    >>> audiostream = AudioStream(wire=True, latency='low',
    ...                           output_file='test.wav', record_delay=True,
    ...                           samplerate=samplerate, blocksize=blocksize)

    """
    def __init__(self, api=default_api, wire=True, output_file=None,
                 record_delay=False, file_kwargs={}, overwrite_file=False,
                 **stream_kwargs):
        # Assign arguments into class
        self.api = api
        self.wire = wire
        self.output_file = output_file
        self.record_delay = record_delay
        self.overwrite_file = overwrite_file

        # Internal flags
        self.RECORDING = False

        # If no device provided but API is, get corresponding device
        if (not 'device' in stream_kwargs) and self.api:
            device = get_device4api(self.api)
            stream_kwargs['device'] = device
            print('\nUsing input audio device {0:d}: {1} with {2}' \
                  .format(device[0],
                          sd.query_devices(device[0])['name'],
                          api)
                  )
            if self.wire:
                print('Using output audio device {0:d}: {1} with {2}' \
                      .format(device[1],
                              sd.query_devices(device[1])['name'],
                              api)
                      )

        # Open stream - type depends on if we're doing wire or not
        if self.wire:  # input+output stream
            streamAPI = sd.Stream
        else:  # input-only stream
            streamAPI = sd.InputStream
        self.stream = streamAPI(callback=self._callback, **stream_kwargs)

        # Open output file if necessary
        if self.output_file:
            # Default to .wav extension
            if not os.path.splitext(self.output_file)[1]:
                self.output_file += '.wav'

            # Handle if file already exists
            if os.path.isfile(self.output_file) and not self.overwrite_file:
                raise IOError('Output audio file already exists')

            # Ensure file_kwargs contains necessary details
            if not 'samplerate' in file_kwargs:
                file_kwargs['samplerate'] = int(self.stream.samplerate)

            if not 'channels' in file_kwargs:
                if self.wire:  # input+output
                    n_input_channels = self.stream.channels[0]
                else:  # input-only
                    n_input_channels = self.stream.channels
                file_kwargs['channels'] = n_input_channels

            # Open file
            self.output_fd = sf.SoundFile(self.output_file, 'w', **file_kwargs)

            # Pad beginning with silence if told to
            if self.record_delay and self.stream.blocksize > 0:
                self.output_fd.write(np.zeros([int(self.stream.blocksize),
                                               file_kwargs['channels']]))

            print('Opened audio file ' + self.output_file)


    def _callback(self, *args):
        """
        Callback function for audio stream.  Passes input data to output
        stream (if wire==True) and writes data to output file (if recording).
        """
        # args depend on whether doing input+output or input-only stream
        if self.wire:  # input+output
            indata, outdata, frames, time, status = args
        else:  # input-only
            indata, frames, time, status = args

        # Status should be None, print something about it if it isn't
        if status:
            print(status)

        # If wire requested, pass indata to outdata
        if self.wire:
            outdata[:] = indata

        # If recording, write data to file
        if self.output_file and self.RECORDING:
            self.output_fd.write(indata)


    def close(self):
        """
        Close audio stream permanently.  If an output file is open, this will
        be closed too.
        """
        if self.output_file:
            if self.RECORDING:
                self.switchRecording()
            self.output_fd.close()
        self.stream.close()


    def start(self):
        """Start audio stream."""
        self.stream.start()

    def stop(self):
        """Stop audio stream temporarily (i.e. pause stream)."""
        self.stream.stop()


    def switchRecording(self, value=None):
        """
        Switches recording status ON or OFF. If provided value is a boolean,
        then will set status to this value. If provided value is None
        (default), then will switch to opposite of current status.
        """
        if self.output_file:
            if value is None:
                self.RECORDING = not self.RECORDING
            else:
                self.RECORDING = value
            if self.RECORDING:
                print('Audio recording')
            else:
                print('Audio not recording')
        else:
            return



## Following classes implement platform specific master volume controls
class _BaseVolumeControl(object):
    """
    Controller for setting system master volume. Windows, Mac OS, and Linux
    platforms are supported.

    Arguments
    ---------
    units : 'percent' (default) or 'raw'
        Whether volume units should be set as a percentage (min=0, max=100),
        or in the raw units of the tool used to set the volume.  Raw units
        are platform specific; the .volume_range attribute will list the
        possible range for your system if the units have been set to 'raw'.

    Further platform specific arguments may be available - see below.

    Methods
    -------
    .get_volume
        Get current volume level, in relevant units.
    .set_volume
        Set volume to specified level, in relevant units.
    .mute
        Toggle mute on / off.
    .set_units
        Update units between percent or raw.
    .reset
        Reset volume to original level.

    Example usage
    -------------
    Set volume to 50% of max.

    >>> volControl = VolumeControl()
    >>> volControl.set_volume(50)

    Reset to original volume when done.

    >>> volControl.reset()
    """

    """
    2nd docstring kept separate from main docstring so it won't be inherited
    by child classes.

    Base-class provides some generic functions that can be used by all the
    platform-specific child classes.  Child classes must implement methods
    .get_volume() and .set_volume(), and need to include an attribute
    ._raw_volume_range which is a 2-item tuple giving the (min,max) values
    of the raw units volume range.
    """
    def __init__(self, units):
        self.set_units(units)
        self.original_units = self.units
        self.current_volume = self.original_volume = self.get_volume()

    def reset(self):
        """Reset volume to level it was at when class was initialised."""
        current_units = self.units
        self.set_units(self.original_units)
        self.set_volume(self.original_volume)
        self.set_units(current_units)

    def set_units(self, units):
        """Set volume units to use: 'percent' or 'raw'."""
        self.units = units
        # Update other attributes dependent on units
        if units == 'percent':
            self.volume_range = (0,100)
        elif units == 'raw':
            self.volume_range = self._raw_volume_range
        else:
            raise ValueError('Units must be \'percent\' or \'raw\'')
        self.current_volume = self.get_volume()


class _WindowsVolumeControl(_BaseVolumeControl):
    """
    Dependencies
    -----------
    * pycaw - https://github.com/AndreMiras/pycaw
    """
    __doc__ = _BaseVolumeControl.__doc__ + __doc__

    def __init__(self, units='percent'):
        assert have_pycaw
        self.devices = AudioUtilities.GetSpeakers()
        self.interface = self.devices.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(self.interface, POINTER(IAudioEndpointVolume))
        self._raw_volume_range = self.volume.GetVolumeRange()
        super(_WindowsVolumeControl, self).__init__(units)

    def get_volume(self):
        """Get current volume."""
        if self.units == 'percent':
            return 100.0 * self.volume.GetMasterVolumeLevelScalar()
        else:
            return self.volume.GetMasterVolumeLevel()

    def set_volume(self, level):
        """Set current volume."""
        if self.units == 'percent':
            self.volume.SetMasterVolumeLevelScalar(level/100.0, None)
        else:
            self.volume.SetMasterVolumeLevel(level, None)
        self.current_volume = self.get_volume()

    def mute(self):
        """Toggle mute."""
        self.volume.SetMute(not self.volume.GetMute(), None)



class _LinuxVolumeControl(_BaseVolumeControl):
    """
    Platform-specific arguments
    ---------------------------
    mixer : str
        Mixer to use, default = 'Master'.  A list of available mixers can be
        provided by alsaaudio.mixers().

    Dependencies
    ------------
    * pyalsaudio : https://larsimmisch.github.io/pyalsaaudio/
        Python bindings to Alsa.  Provides analagous commands to "amixer"
        shell command. Is available in apt, try:
        sudo apt install python-alsaaudio
    """
    __doc__ = _BaseVolumeControl.__doc__ + __doc__

    def __init__(self, units='percent', mixer='Master'):
        assert have_alsa
        self.mixer = mixer
        self.mixer_control = alsaaudio.Mixer(mixer)
        self._raw_volume_range = self.mixer_control.getrange()
        super(_LinuxVolumeControl, self).__init__(units)

    def _raw2percent(self, v):
        """Hidden function - converts raw to percent volume units. """
        percent = 100.0 * v / (self.volume_range[1] - self.volume_range[0])
        return int(round(percent))

    def _percent2raw(self, v):
        """Hidden function - converts percent to raw volume units. """
        raw = (v / 100.0) * (self.volume_range[1] - self.volume_range[0])
        return int(round(raw))

    def get_volume(self):
        """Get current volume."""
        vol = map(int, self.mixer_control.getvolume())
        if self.units == 'raw':
            vol = map(self._percent2raw, vol)
        return vol

    def set_volume(self, level):
        """Set current volume."""
        if not hasattr(level, '__iter__'):
            level = [level]
        for channel, lev in enumerate(level):
            if self.units == 'raw':
                lev = self._raw2percent(lev)
            self.mixer_control.setvolume(lev, channel)
        self.current_volume = self.get_volume()

    def mute(self):
        """Toggle mute. """
        for channel, isMuted in enumerate(self.mixer_control.getmute()):
            self.mixer_control.setmute(not isMuted, channel)


class _MacOSVolumeControl(_BaseVolumeControl):
    """
    Dependencies
    ------------
    * appscript : https://pypi.python.org/pypi/appscript/
        Is available in pip, try: pip install --user appscript

    Notes
    -----
    Appscript can set raw volume units between 0 and 7, but doesn't seem to be
    able to retrieve them from the system.  Consequently, only percentage
    units are supported for now.
    """
    __doc__ = _BaseVolumeControl.__doc__ + __doc__

    def __init__(self, units='percent'):
        assert have_osax
        self.sa = osax.OSAX()
        self._raw_volume_range = (0,7)  # hard-code this one
        super(_MacOSVolumeControl, self).__init__(units)

    def get_volume(self):
        """Get current volume."""
        if self.units == 'raw':
            raise NotImplementedError('Only percent units supported')
        return self.sa.get_volume_settings()[osax.k.output_volume]

    def set_volume(self, level):
        """Set current volume."""
        if self.units == 'raw':
            self.sa.set_volume(level)
        else:
            self.sa.set_volume(output_volume=level)
        self.current_volume = self.get_volume()

    def mute(self):
        """Toggle mute."""
        isMuted = self.sa.get_volume_settings()[osax.k.output_muted]
        self.sa.set_volume(output_muted=not isMuted)


# Assign volume control class by platform
if platform == 'windows':
    VolumeControl = _WindowsVolumeControl
elif platform == 'linux':
    VolumeControl = _LinuxVolumeControl
elif platform == 'mac':
    VolumeControl = _MacOSVolumeControl



