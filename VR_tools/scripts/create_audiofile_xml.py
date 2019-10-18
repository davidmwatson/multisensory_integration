"""
Script may be used to create a playlist xml file for the AudioFile device.
Alternatively, make_files function can be imported from this script into your
own script if you would prefer to use it from within Python rather directly
at the commandline.

Arguments
---------
-i, --indir (required)
    Path to directory containing input audio files

-o, --outbase (required)
    Base path to desired output files, i.e. path to files but excluding any
    extensions

-f, --folder (required)
    Path of folder relative to AudioFile SD card root directory that
    audio files will be placed in.  This will be used as the Folder value
    in the output xml file.

-c, --codes (optional)
    Integer or range of integers to indicate codes to use for encoding each
    audio file in the input directory.  If a single integer, tracks will be
    coded sequentially starting from this number.  If a range of integers,
    the alphabetically sorted list of tracks will be coded against this range
    (in this case the length of the range must match the number of files).
    If not specified, defaults to a sequential range starting at 1.


-e, --ext (optional)
    Format extension (exlcuding leading . symbol) of input audio files,
    e.g. "mp4".  Only needed if format is not the "wav" default.

--use-D0 (optional)
    Specify to allow use of D0 input bit on the parallel port connector
    (pin 1 / Dout0).  If not specified, you may only use pins 2-8 (Dout1-7)
    on the parallel connector for setting which track in the playlist to play.
    This limits you to specifying 7 bit values, allowing only ranges between
    0 and 127.  If specified, you may use pins 1-8 (Dout0-7) thereby allowing
    8 bit values, in turn allowing ranges between 0 and 255.  Note that the
    number of allowable files in the playlist will always be at least 2 less
    than the allowable range (so 254 if using D0, or 126 otherwise) as 2 values
    must be reserved for stopping playback (0 by default, but can be changed
    using --stop-code flag) and for playing all tracks in sequence (max on all
    pins, so 127 or 255 depending on if D0 is used).  Note that this means
    you can never represent more than 254 files in a single playlist.

-v, --volume (optional)
    Output volume, must be in range 0 - 100

--stop-code (optional)
    Code used to indicate an immediate stop to playback from the device.
    Note that the stop code will always override any track code that it
    conflicts with, meaning you be unable to trigger playback of that track.
    Default code is 0, which cannot be used as a track code and hence will
    never conflict with one.

--swap-channels (optional)
    Specify to indicate left / right channels should be swapped in the output


Outputs
-------
Script outputs an xml and a csv file, named <outbase>.xml and <outbase>.csv
respectively.  The xml file may be placed in the Playlists directory on
the AudioFile SD card.  Note that although this directory may contain multiple
xml files, only the one named Playlist.xml will be used by the device.  If you
want to use this xml file then, you will have to (re)name it to this.

The csv file contains a list of the input filenames against the codes that
were used to denote them in the xml file.  This may be used as a lookup when
determining the correct code to send to the AudioFile device to trigger
playback of a specific track.


Example Usage
-------------
> python create_audiofile_xml.py -i ./audio_files -o ./audio_files/Playlist \\
    -f Tracks

"""

import os, sys, glob, argparse, csv, warnings

# Some default params to be shared by create_files func and argparser
defaults = {'codes':[1], 'ext':'wav', 'use_D0':False, 'volume':75,
            'stop_code':0, 'swap_channels':False}


### Custom functions and classes
class customFormatter(argparse.RawDescriptionHelpFormatter,
                      argparse.ArgumentDefaultsHelpFormatter):
    """
    Empty class - serves only to subclass argparse RawDescriptionHelpFormatter
    and ArgumentDefaultsHelpFormatter so that we can use both at same time
    """
    pass


def create_files(indir, outbase, folder, codes = defaults['codes'],
                 ext = defaults['ext'], use_D0 = defaults['use_D0'],
                 volume = defaults['volume'],
                 stop_code = defaults['stop_code'],
                 swap_channels = defaults['swap_channels']):
    """
    Main function for making xml and csv files.  Arguments correspond to
    equivalents in argparser, so see main script docs for more info.
    """
    # Error check and format args as necessary
    indir = os.path.realpath(indir) # ensure full filepath
    outbase = os.path.splitext(outbase)[0] # ensure extension not present
    n_codes = len(codes)
    if ext.startswith('.'):
        ext = ext.lstrip('.') # remove leading . character
    if not 0 <= volume <= 100:
        raise ValueError('Volume must be in range 0-100') # check correct range

    # Glob input files and error check
    input_files = sorted(glob.glob(os.path.join(indir, '*.' + ext)))
    n_files = len(input_files)
    if not input_files:
        raise IOError('Failed to find any input files')
    elif n_files > 126 and not use_D0:
        raise ValueError('Cannot represent more than 126 files without using '
                         'D0 input bit')
    elif n_files > 254:
        raise ValueError('Cannot represent more than 254 files')

    # If a code range for tracks given, check we have right number
    if n_codes > 1  and n_codes != n_files:
        raise ValueError('Number of track codes (%d) does not match number of '
                         'input files (%d)' % (n_codes, n_files))
    # If single code value for tracks given, generate range
    elif n_codes == 1:
        codes = range(codes[0], codes[0] + n_files)

    # Open output xml file for writing, write headers
    xml_f = open(outbase + '.xml', 'w')
    xml_f.write('<?xml version="1.0"?>\n')
    xml_f.write('<AUDIOFILE_PLAYLIST>\n')
    xml_f.write('\t<PLAYLIST1>\n')
    xml_f.write('\t\t<Entry Folder="%s"/>\n' % folder)

    # Open output csv file for writing, create writer object, write headers
    csv_f = open(outbase + '.csv', 'w')
    csv_writer = csv.DictWriter(csv_f, fieldnames = ['code','filename'],
                                delimiter = ',', lineterminator='\n')
    csv_writer.writeheader()

    # Begin looping through files
    for code, input_file in zip(codes, input_files):
        # Grab out basename
        input_name = os.path.basename(input_file)

        # Format code for output
        code = format(code, '03d')

        # If code equals stop code, print a warning
        if int(code) == stop_code:
            msg = 'Code %s for file %s matches stop code, you will be' \
                   'unable to play this file' % (code, input_name)
            warnings.warn(msg)

        # Write to xml
        xml_f.write('\t\t<Entry Code%s="%s"/>\n' % (code, input_name))

        # Write to csv
        csv_writer.writerow({'code':code, 'filename':input_name})

    # Write xml footers
    xml_f.write('\t</PLAYLIST1>\n')
    xml_f.write('\t<SYSTEM>\n')
    xml_f.write('\t\t<Entry UseDigitalInputD0="%s"/>\n' %str(use_D0).upper())
    xml_f.write('\t\t<Entry Volume="%d"/>\n' % volume)
    xml_f.write('\t\t<Entry StopCode="%d"/>\n' % stop_code)
    xml_f.write('\t\t<Entry SwapChannels="%s"/>\n' %str(swap_channels).upper())
    xml_f.write('\t\t<Entry SDRAMTest="FALSE"/>\n')
    xml_f.write('\t</SYSTEM>\n')
    xml_f.write('</AUDIOFILE_PLAYLIST>\n')

    # Finish up
    xml_f.close()
    csv_f.close()


def main():
    """
    Function to be called when script invoked from commandline'
    """
    # Parse cmdline args
    p = argparse.ArgumentParser(description = __doc__,
                                formatter_class = customFormatter)
    p.add_argument('-i', '--indir', required = True,
                   help = 'Directory containing input audio files')
    p.add_argument('-o', '--outbase', required = True,
                   help = 'Base path for output files, excluding extension')
    p.add_argument('-f', '--folder', required = True,
                   help = 'Name of playlist folder to be listed in xml file')
    p.add_argument('-c', '--codes', nargs = '+', type = int,
                   default = defaults['codes'],
                   help = 'Starting code or range of codes for audio files')
    p.add_argument('-e', '--ext', default = defaults['ext'],
                   help = 'Extension of input audio files')
    p.add_argument('--use-D0', action = 'store_true',
                   help = 'Allow use of Dout0 (pin 1)')
    p.add_argument('-v', '--volume', type = int, default = defaults['volume'],
                   help = 'Output volume level (range 0-100)')
    p.add_argument('-s', '--stop-code', type = int,
                   default = defaults['stop_code'],
                   help = 'Code to stop track playback immediately')
    p.add_argument('--swap-channels', action = 'store_true',
                   help = 'Swap left and right audio channels')

    if not len(sys.argv) > 1:
        p.print_help()
        sys.exit(0)

    # Run func
    create_files(**vars(p.parse_args()))
    print('\nDone\n')



### Run script when invoked from commandline
if __name__ == '__main__':
    main()


