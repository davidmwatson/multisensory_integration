#!/usr/bin/env python3

"""
Script can be used to test playback on AudioFile device, controlled via
labjack device.  A specific track number may be passed as a positional
argument in the commandline after the script name.  If no track number given,
will default to track 1.  Specifying a track number of 0 will halt the current
playback (or whatever other index was indicated as the stop code in the xml
file).

Dependencies
------------
* psychopy
* Other psychopy.hardware.labjack dependencies as detailed at:
  http://www.psychopy.org/api/hardware/labjack.html

"""

import sys
from psychopy.hardware import labjacks


# Parse cmdline args
if not len(sys.argv) > 1:
    track = 1
else:
    if sys.argv[1] in ['-h','--help']:
        print(__doc__)
        sys.exit(0)
    else:
        track = int(sys.argv[1])


# Play track
jack = labjacks.U3()
jack.setData(track)
jack.setData(128)
jack.close()