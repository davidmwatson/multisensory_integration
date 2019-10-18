#!/usr/bin/env python3

"""
Miscellaneous utility functions.
"""

import numpy as np
from psychopy.visual.shape import ShapeStim


class FixationStim(ShapeStim):
    """
    Sub-class of psychopy.visual.ShapeStim for drawing a fixation cross
    in the centre of the screen.

    Arguments
    ---------
    win - psychopy window instance, required
        Handle to open psychopy window.
    crossSize - int or float, or (L,W) tuple thereof, required
        Size of fixation cross in relevant units.  If a single value given
        then this will be used for both the length and width of the cross.
        Alternatively, can specify separate length and width values as an
        (L,W) tuple of values.
    **kwargs
        Further keyword arguments are passed to the ShapeStim base class.
    """
    def __init__(self, win, crossSize, **kwargs):
        # Parse size
        if not hasattr(crossSize, '__iter__'):
            crossL = crossW = crossSize
        else:
            crossL, crossW = crossSize
        hCrossL = crossL//2
        hCrossW = crossW//2

        # Assign args into base class.  Assign into class's __dict__ attribute
        # rather than class directly to avoid intefering with base class's
        # attribute setter methods.
        self.__dict__['crossSize'] = crossSize
        self.__dict__['crossL'] = crossL
        self.__dict__['crossW'] = crossW
        self.__dict__['hCrossL'] = hCrossL
        self.__dict__['hCrossW'] = hCrossW

        # Vertices defined by repeatedly doubling back between centre and
        # arm tips
        self.__dict__['vertices'] = np.asarray([
                (-hCrossW,0), (0,0),  # left arm
                (0,-hCrossL), (0,0), # bottom arm
                (hCrossW,0), (0,0),  # right arm
                (0,hCrossL), (0,0)  # top arm
                ])

        # Sort out kwargs
        kwargs['vertices'] = self.vertices
        kwargs['closeShape'] = False
        kwargs['fillColor'] = None

        # Instantiate
        super(FixationStim, self).__init__(win, **kwargs)



class EmulatedU3(object):
    """
    Dummy class to emulate psychopy.hardware.labjacks.U3. Allows passing a
    "fake" jack device for debugging.
    """
    def __init__(*args, **kwargs):
        pass

    def setData(*args, **kwargs):
        pass

    def close(*args, **kwargs):
        pass

