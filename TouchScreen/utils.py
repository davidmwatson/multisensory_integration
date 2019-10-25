#!/usr/bin/env python3

import time, serial, warnings
import numpy as np
# Need full (not lazy) import of ElementArrayStim for sub-classing to work
from psychopy.visual.elementarray import ElementArrayStim



def findPortAddress(regexp='/dev/ttyACM.*', include_links=False):
    """
    Utility function for searching available serial ports for Display++

    Arguments
    ---------
    regexp - str
        Regular expression giving search pattern. Set to empty string to
        search everything.
    include_links - bool
        Whether to follow symlinks

    Returns
    -------
    address - str
        Port address. If no matches (or somehow multiple matches) are found
        an OSError will be raised instead.
    """
    from serial.tools import list_ports  # does most of the work for us

    # List all avaialable addresses
    addressList = [p.device for p in list_ports.grep(regexp, include_links)]
    if not addressList:
        raise OSError('No ports found matching search pattern')

    # Loop addresses
    matches = []
    for address in addressList:
        try:
            # Open port
            port = serial.Serial(port=address)
            time.sleep(0.1)
            assert port.writable()
            assert port.readable()

            # Tell Display++ to return its name, check for match in return
            port.read(port.in_waiting)  # flush buffer
            port.write(b'$ProductType\r\n')
            time.sleep(0.1)
            res = port.read(port.in_waiting).decode().rstrip().split(';')
            cmd, device = res[:2]
            port.close()
            if cmd == '$ProductType' and device == 'Display++':
                matches.append(address)
        except:
            try:
                port.close()
            except:
                pass

    # We should have 1 address - return if we do, otherwise error
    if len(matches) == 1:
        return matches[0]
    elif len(matches) == 0:
        raise OSError('Failed to find Display++ on any ports')
    else:  # len(matches) > 1
        raise OSError(f'Multiple matching ports: {matches}')



class TouchScreenReader(object):
    """
    Provides methods for connecting to and reading touch data from Display++

    Arguments
    ---------
    address : str or None, optional
        Serial port address. Default should be correct for Display++
        If None, will use findPortAddress function to search for port.
    resolution : array-like, opt
        [width, height] resolution of the display in pixels.
    origin : str {'C' | 'centre' | 'center' | 'L' | 'left'}
        Whether origin of co-ords should be in centre of screen, or at one
        of the left-hand corners (upper-left if <ydir> == 'down',
        or lower-left if <ydir> == 'up')
    ydir : str {'up' | 'down'}
        Set whether y-coordinates increase moving up or down the screen
    units : str {'pix' | 'norm'}
        Set whether co-ordinates are returned in units of raw pixels, or in a
        normalised range between -1:1 (or range 0:2 if <origin> == 'L')
    onError : str {'warn' | 'ignore' | 'error'}, optional
        What to do if an error is encountered during reading. If 'warn'
        (default), error is raised but as a warning. If 'ignore', do nothing.
        If 'error', raise the original error.
    *args, **kwargs : Further arguments passed to serial.Serial

    Methods
    -------
    .get_coords
        Return current co-ordinates from Display++
    .flush
        Flush input buffer.
    .close
        Close serial port - should call when finished.

    See also
    --------
    findPortAddress - Utility function for searching for Display++ across
        available serial ports.

    """

    def __init__(self, address='/dev/ttyACM0', resolution=[1920, 1080],
                 origin='centre', ydir='up', units='pix', onError='warn',
                 *args, **kwargs):

        # Error check
        if origin in ['C','centre','center']:
            origin = 'centre'
        elif origin in ['L','left']:
            origin = 'left'
        else:
            raise ValueError(f"Invalid value for origin: '{origin}'")

        if ydir not in ['up','down']:
            raise ValueError(f"Invalid value for ydir: '{ydir}'")

        if units not in ['pix','norm']:
            raise ValueError(f"Invalid value for units: '{units}'")

        if onError not in ['warn','ignore','error']:
            raise ValueError(f"Invalid value for onError: '{onError}'")

        # Search for address?
        if address is None:
            print('Searching for port address...')
            address = findPortAddress('', False)
            print('Found Display++ at ' + address)

        # Pack into class
        self.address = address
        self.resolution = np.asarray(resolution)
        self.half_resolution = self.resolution / 2
        self.origin = origin
        self.ydir = ydir
        self.units = units
        self.onError = onError

        # Open port
        self.port = serial.Serial(port=self.address, *args, **kwargs)
        if not self.port.is_open:
            raise serial.SerialException('Failed to open port')
        if not self.port.readable():
            raise serial.SerialException('Port not readable')

        # Port takes time to spool up - block a little bit to let it finish
        time.sleep(0.1)


    def get_coords(self, event_type='press'):
        """
        Get current co-ordinates and timestamp of touch input

        Arguments
        ---------
        event_type : str {'press' | 'release' | 'both'}
            Filter events for only presses, only releases, or both.

        Returns
        -------
        success : bool
            Boolean value indicating if data was read successfully (e.g. might
            be False if there is a transmission error or if the screen
            is not currently being touched)
        res : 2D (4 * n_events) array of flaots
            Array containing all events since last read. Along 1st dimension,
            values indicate (in order): timestamp (in seconds), x-coord,
            y-coord, and event type code (1 = press, 0 = release). If reading
            was unsuccessful, will return a 4*1 array of NaNs.
        """
        # Default return values - will not be modified if read unsuccessful
        success = False
        res = np.full([4,1], np.nan)

        # Use try block to handle possible transmission errors
        try:
            # Read from port
            if self.port.in_waiting > 0:
                lines = self.port.read(self.port.in_waiting).decode().rstrip().split()
                _res = []
                for line in lines:
                    parts = line.rstrip(';').split(';')
                    if parts and parts[0] == '$touch':
                        _res.append(parts[1:])

                # Cast to array and transpose
                _res = np.asarray(_res, dtype=np.float64).T

                # Filter events
                if event_type == 'press':
                    _res = _res[:, _res[3,:] == 1]
                elif event_type == 'release':
                    _res = _res[:, _res[3,:] == 0]
                elif event_type == 'both':
                    pass
                else:
                    raise ValueError(f"Invalid argument to event_type: '{event_type}'")

                # If array has data in (e.g. not all data filtered out)...
                if _res.size > 0:
                    # Adjust coords as necessary
                    if self.ydir == 'up':
                        _res[2,:] = self.resolution[1] - _res[2,:]

                    if self.origin == 'centre':
                        _res[1,:] -= self.half_resolution[0]
                        _res[2,:] -= self.half_resolution[1]

                    if self.units == 'norm':
                        _res[1,:] /= self.half_resolution[0]
                        _res[2,:] /= self.half_resolution[1]

                    # Update return values
                    res = _res
                    success = True

        # Transmission error, do required error action
        except Exception as e:
            if self.onError == 'ignore':
                pass
            elif self.onError == 'warn':
                with warnings.catch_warnings():
                    warnings.simplefilter('always')
                    warnings.warn(e)
            elif self.onError == 'error':
                raise e

        # Return
        return success, res


    def flush(self):
        """Flush input buffer"""
        self.get_coords('both')


    def close(self):
        """Close serial port connection"""
        self.port.close()



class MyElementArrayStim(ElementArrayStim):
    """
    Sub-classes ElementArrayStim to provide customised dot-stim object.

    Arguments
    ---------
    All as per ElementArrayStim, but some defaults are different.

    Methods
    -------
    .update_coords - Update dot co-ordinates (random or specific direction)
    """
    __doc__ += ElementArrayStim.__doc__
    def __init__(self, win, nElements, fieldShape=None, elementTex=None,
                 elementMask='circle', units='pix', *args, **kwargs):

        # Get win dims
        self.W, self.H = win.size
        self.W2 = self.W//2
        self.H2 = self.H//2

        # Initialise array co-ords - prepend with underscore as they need to
        # be kept separate from ElementArrayStim's .xys array
        self._xys = self._generate_coords(nElements)

        # Update kwargs and init parent class
        kwargs['nElements'] = nElements
        kwargs['fieldShape'] = fieldShape
        kwargs['elementTex'] = elementTex
        kwargs['elementMask'] = elementMask
        kwargs['units'] = units
        kwargs['xys'] = self._xys
        super().__init__(win, *args, **kwargs)

    def _generate_coords(self, N):
        """Hidden function, returns Nx2 array of random x,y co-ords"""
        xy = np.random.randint(low=(-self.W2, -self.H2),
                               high=(self.W2+1, self.H2+1),
                               size=(N, 2))
        return xy

    @staticmethod
    def _wrap(x, low, high):
        """Wraps vector x into interval low:high, operating in-place"""
        x[x > high] -=  (high-low)
        x[x < low] += (high-low)

    def update_dots(self, dxy=None, prop=1):
        """
        Update dot co-ordinates, either randomly or in a particular direction

        Arguments
        ---------
        dxy : None, 'rand', or [x,y] array-like
            If None or 'rand', just randomly update the dots. If an [x,y] array
            then move the dots in direction specified by this vector.
        prop : float in range 0:1
            Proportion of dots to update. Selection of which dots is random.
        """
        # Work out how many and which dots to update
        propN = int(round(self.nElements * prop))
        idcs = np.random.choice(self.nElements, size=propN, replace=False)

        # If random update, just randomly update coords
        if dxy is None or (isinstance(dxy, str) and dxy == 'rand'):
            self._xys[idcs, :] = self._generate_coords(propN)

        # If directional update, increment coords by specified amount, and
        # wrap any dots that fall outside of window limits
        elif isinstance(dxy, (list, tuple, np.ndarray)) and len(dxy) == 2:
            self._xys[idcs, :] += np.round(dxy).astype(int)
            self._wrap(self._xys[:,0], -self.W2, self.W2)
            self._wrap(self._xys[:,1], -self.H2, self.H2)

        else:
            raise ValueError(f'Invalid value for dxy: {dxy}')

        self.setXYs(self._xys)
