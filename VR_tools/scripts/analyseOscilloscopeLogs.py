"""
Script provides function for analysing CSVs output by Syscomp Oscilloscope.
Function may be used directly, or script also provides option for commandline
access.

Takes data from CSV files in a given directory and concatenates them into
a single timeseries.  Video and audio channels are analysed to determine
onsets and offsets of the stimulus, and corresponding video-audio latencies
calculated:
* The video signal is easy enough to analyse.  The signal amplitude increases
  when the stimulus comes on, and decreases when it turns off.  We therefore
  just take the absolute values of the signal and then define the ON / OFF
  periods by the signal passing some threshold level.
* The audio signal is more complicated as this oscillates, so we can't just
  place a threshold through the raw input signal.  Instead, we calculate
  a spectrogram of the signal to analyse the oscillatory power within sliding
  time windows across the signals.  ON / OFF periods can then be determined
  by the oscillatory power passing some threshold level.

Both the video and audio signals can be subject to measurement noise, which
can produce false positives where the signal momentarily passes over or under
the threshold level, causing an onset or offset to be erroneously logged.
To reduce this, the video signal and audio oscillatory power signals are
passed through a low-pass Gaussian filter to smooth the signal out. The
bandwidth of this filter can be altered if needs be, but should be set high
enough so as not to attenuate oscillations related to the stimulus.

Script outputs a text file containing the estimated onset and offset times of
the video and audio stimuli.  It will also list the video-audio latencies
between onsets and offsets if possible.  If it is not possible to calculate
these (e.g. false positives have created a differing number of onsets/offsets
between channels) then NaNs will be output instead - in this case you will
need to take the raw onsets and offsets and remove the false positives by
hand, then calculate the latencies yourself.  In addition, a summary plot will
be created.

Commandline access to this script provides equivalent arguments for the
analyseOscilloscopeLogs function.  For more details on those arguments,
see the documentation for that function.

"""

from __future__ import division
import os, sys, argparse, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.signal import convolve, gaussian, spectrogram
from scipy.interpolate import interp1d


def fwhm2sigma(fwhm):
    """
    Converts a full-width-half-maximum value to a sigma value
    """
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def analyseOscillosocopeLogs(indir, outname=None,
                             channel_order=['video','audio'],
                             threshold=0.5, filter_fwhm=20):
    """
    Main analysis function.  Analyses onsets / offsets of video and audio
    stimuli, and attempts to calculate video-audio latencies.

    Arguments
    ---------
    indir - str (filepath), required
        Input directory, containing csv files to be analysed.  Function will
        simply glob all csv files in this directory, so make sure no other
        csv files are present in this directory.
    outname - str (filepath) or None, optional
        Path to desired output file.  The file extension should be omitted;
        a .txt extension will be appended for the text file output, and a .png
        extension will be appended for the plot.  If None (default), will
        default to '<indir>/analysis'.
    channel_order - 2 item list of str, optional
        Should contain the labels 'video' and 'audio', denoting the order
        of those columns in the csv file.  It is assumed that the first column
        of the csv file gives time indices.
    threshold - float in range 0-1, optional
        Proportion of peak-to-peak amplitude difference to use as the
        threshold for determingin onset / offset points.
    filter_fwhm - float, optional
        FWHM to use for Gaussian smoothing kernel, specified in Hz.  If set
        to 0, no smoothing will be applied.

    """
    # Handle outname
    if outname is None:
        outname = os.path.join(indir, 'analysis')
    else:
        outname = os.path.splitext(outname)[0]  # ensure extension stripped

    # Pre-allocate list for storing time, video, and audio points.  Also make
    # list for timepoints at splits between timeseries when we stitch
    # multiple csvs together
    tpoints = []
    tsplits = []
    vpoints = []
    apoints = []

    # Glob input files
    infiles = sorted(glob.glob(os.path.join(indir, '*.csv')))
    assert infiles

    ## Load data files
    for i, infile in enumerate(infiles):
        # Load csv
        thisData = np.loadtxt(infile, delimiter=',', skiprows=1, unpack=True)
        thisTime = thisData[0, :]
        if channel_order[0] == 'video':
            thisVideo, thisAudio = thisData[1:, :]
        else:
            thisAudio, thisVideo = thisData[1::-1, :]

        # For all but 1st file increment time values by final time value of
        # previous dataset, so that time appears continuous across datasets
        if i > 0:
            thisTime += tpoints[-1]

        # Normalise video and audio to 0 at start (base on first few samples
        # as very first sample is not guaranteed to be 0)
        thisVideo -= thisVideo[:20].mean()
        thisAudio -= thisAudio[:20].mean()

        # Append to lists
        tpoints.extend(thisTime)
        if i > 0:
            tsplits.append(thisTime[0])
        vpoints.extend(thisVideo)
        apoints.extend(thisAudio)

    # Cast lists to np arrays for easier indexing
    tpoints = np.array(tpoints)
    tsplits = np.array(tsplits)
    vpoints = np.array(vpoints)
    apoints = np.array(apoints)

    # Work out sampling freq
    samp_freq = len(tpoints) / (tpoints.max() - tpoints.min())

    # Create smoothing kernel
    if filter_fwhm:
        samp_fwhm = samp_freq / filter_fwhm
        samp_sigma = fwhm2sigma(samp_fwhm)
        M = int(round(samp_sigma * 6))  # extend over +/- 3 sigmas
        gauss_kernel = gaussian(M, samp_sigma)
        gauss_kernel /= gauss_kernel.sum()
    else:
        gauss_kernel = None


    ## Process video signal
    # Take absolute values
    abs_vpoints = np.abs(vpoints)

    # Smooth
    if gauss_kernel is not None:
        abs_vpoints = convolve(abs_vpoints, gauss_kernel, 'same')

    # Threshold
    vthr = abs_vpoints.min() \
           + (threshold * (abs_vpoints.max()-abs_vpoints.min()))

    # Mark on / off periods
    v_on = (abs_vpoints >= vthr).astype(float)

    # Determine onsets / offsets by difference scores (i.e. each point minus
    # the previous).  Should get a spike at +1 for flips from OFF (0) to
    # ON (1), and a spike at -1 for flips from ON (1) to OFF (0)
    v_on_diff = np.diff(v_on)
    v_onsets = tpoints[np.where(v_on_diff == 1)]
    v_offsets = tpoints[np.where(v_on_diff == -1)]


    ## Process audio signal
    # Calculate spectrogram
    spec_fq, spec_tpoints, audio_spec = spectrogram(apoints, samp_freq,
                                                    nperseg=100, noverlap=80)
    # TODO - we just set nperseg and noverlap arbitrarily here, maybe we
    # should tie this to the sampling freq somehow?

    # Get amplitude, average across freqs
    av_abs_audio_spec = np.abs(audio_spec).mean(axis=0)

    # Interpolate back to original sampling resolution
    interp_func = interp1d(spec_tpoints, av_abs_audio_spec, kind='linear',
                           bounds_error=False, fill_value='extrapolate')
    av_abs_audio_spec = interp_func(tpoints)

    # Smooth
    if gauss_kernel is not None:
        av_abs_audio_spec = convolve(av_abs_audio_spec, gauss_kernel, 'same')

    # Threshold
    athr = av_abs_audio_spec.min() \
           + ( threshold * (av_abs_audio_spec.max() \
                            - av_abs_audio_spec.min()) )

    # Mark on / off periods
    a_on = (av_abs_audio_spec >= athr).astype(float)

    # Determine onsets / offsets
    a_on_diff = np.diff(a_on)
    a_onsets = tpoints[np.where(a_on_diff == 1)]
    a_offsets = tpoints[np.where(a_on_diff == -1)]


    ## Get latencies
    if len(v_onsets) == len(a_onsets):
        onset_latencies = v_onsets - a_onsets
        mean_onset_latency = onset_latencies.mean()
        std_onset_latency = onset_latencies.std()
    else:
        print('Differing number of onsets')
        onset_latencies = np.array([np.nan])
        mean_onset_latency = std_onset_latency = np.nan

    if len(v_offsets) == len(a_offsets):
        offset_latencies = v_offsets - a_offsets
        mean_offset_latency = offset_latencies.mean()
        std_offset_latency = offset_latencies.std()
    else:
        print('Differing number of offsets')
        offset_latencies = np.array([np.nan])
        mean_offset_latency = std_offset_latency = np.nan


    ## Write output to file
    with open(outname + '.txt', 'w') as fd:
        # Onsets
        fd.write('ONSETS (seconds)\n')
        fd.write('Video:\t' + ', '.join(map(str, a_onsets)) + '\n')
        fd.write('Audio:\t' + ', '.join(map(str, a_onsets)) + '\n')
        fd.write('V-A latencies:\t' + ', '.join(map(str, onset_latencies)) \
                 + '\n')
        fd.write('Mean latency:\t{}\n'.format(mean_onset_latency))
        fd.write('Stdev latency:\t{}\n'.format(std_onset_latency))
        fd.write('\n')

        # Offsets
        fd.write('OFFSETS (seconds)\n')
        fd.write('Video:\t' + ', '.join(map(str, v_offsets)) + '\n')
        fd.write('Audio:\t' + ', '.join(map(str, a_offsets)) + '\n')
        fd.write('V-A latencies:\t' + ', '.join(map(str, offset_latencies)) \
                 + '\n')
        fd.write('Mean latency:\t{}\n'.format(mean_offset_latency))
        fd.write('Stdev latency:\t{}\n'.format(std_offset_latency))


    ## Plot - place video and audio on separate y-axes
    fig, ax1 = plt.subplots(figsize = (2*len(infiles), 6))
    ax2 = ax1.twinx()

    # Amplitudes
    ax1.plot(tpoints, abs_vpoints, 'r-', alpha=.75, lw=2, zorder=2)
    ax2.plot(tpoints, av_abs_audio_spec, 'b-', alpha=.75, lw=2, zorder=2)

    # Mark thresholds
    xlim = ax1.get_xlim()
    ax1.hlines(vthr, xlim[0], xlim[1], linestyles='--', colors='r',
               linewidths=2, alpha=.75, zorder=3)
    ax2.hlines(athr, xlim[0], xlim[1], linestyles=':', colors='b',
               linewidths=2, alpha=.75, zorder=3)
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)

    # ON periods + split markers
    ylim1 = ax1.get_ylim()
    ax1.plot(tpoints[v_on==1], v_on[v_on==1] * (ylim1[1]), 'r.', markersize=5)
    ax1.vlines(tsplits, ylim1[0], ylim1[1]*1.05, linestyles='--', colors='k',
               linewidths=2, zorder=1)
    ax1.set_ylim(ylim1[0], ylim1[1]*1.05)

    ylim2 = ax2.get_ylim()
    ax2.plot(tpoints[a_on==1], a_on[a_on==1] * (ylim2[1]*1.02), 'b.',
             markersize=5)
    ax2.set_ylim(ylim2[0], ylim2[1]*1.05)

    # Tweak axes
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    ax1.set_xlabel('Time (s)', fontsize=16)
    ax1.set_ylabel('Video Amplitude', color='r', fontsize=16)
    ax1.tick_params('y', colors='r')
    ax1.tick_params(labelsize=14)

    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    ax2.set_ylabel('Audio Oscillatory Amplitude', color='b', fontsize=16)
    ax2.tick_params('y', colors='b')
    ax2.tick_params(labelsize=14)

    # Save
    fig.savefig(outname + '.png', bbox_inches='tight', dpi=150)
    plt.close(fig)


### Commandline access
if __name__ == '__main__':
    # Set up parser
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter, usage=__doc__
            )
    parser.add_argument('-i', '--indir', required=True,
                        help='Directory containing input files (required)')
    parser.add_argument('-o', '--outname',
                        help='Name of output file excluding extension '
                             '(optional)')
    parser.add_argument('-c', '--channel-order', nargs=2,
                        help='Order of channels in columns of csv files '
                             '(optional)')
    parser.add_argument('-t', '--threshold', type=float,
                        help='Proportion of peak-to-peak amplitude '
                             '(optional)')
    parser.add_argument('--filter-fwhm', type=float,
                        help='FWHM of smoothing filter in Hz (optional)')

    # Parse input args
    if not len(sys.argv) > 1:
        parser.print_help()
        sys.exit(0)
    kwargs = vars(parser.parse_args())

    # Pop out null entries - will fall back on defaults in analyse func
    kwargs = dict([(k, kwargs[k]) for k in kwargs.keys() \
                   if kwargs[k] is not None])

    # Run function
    analyseOscillosocopeLogs(**kwargs)

    # Done
    print('\nDone\n')
