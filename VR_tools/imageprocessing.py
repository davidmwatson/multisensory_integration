#!/usr/bin/env python3

"""
Assorted functions / classes for image processing operations, e.g. filtering.
"""

from __future__ import division
import sys, cv2
import numpy as np
from numpy import pi
import scipy.signal
import scipy.ndimage

# Bit of error handling for pyfftw - module not in PsychoPy standalone
try:
    import pyfftw
    have_pyfftw = True
except ImportError:
    have_pyfftw = False

# Py2 <--> Py3 compatibility fixes
if sys.version_info.major == 3:
    import pickle
else:
    import cPickle as pickle



##### Class definitions ####
class pyfftw_FourierFilter(object):
    """
    Simple class for applying filtering to image via Fourier Domain. Fourier
    transforms are performed using the pyfftw library which allows very fast
    execution of the transforms, so that filtering can can potentially be done
    in real time.

    Arguments
    ---------
    imsize : (length, width) or (length, width, depth) tuple, required
        Length and width of image in pixels. If depth dimension also included,
        it is assumed that this refers to the image colour channels.
    filt : numpy array, required
        Desired filter, e.g. as produced by createFourierFilter function.
        Filters should be for an unshifted spectrum (DC in top left) and shaped
        for a "real" Fourier transfrom (i.e. length of filter should be length
        of image, whilst width of filter should be half of width of image then
        plus 1).  Filter datatype must also be something that can be converted
        to float32.
    apply_mask : bool, optional
        If True, a soft-window mask (as created by createSoftWindowMask
        function) will be applied as an alphamask to the image.  This
        "fades out" the edges of the image to help to hide some of the edge
        artifacts associated with the filtering process.
    mask_shape : str, optional
        Shape of the mask - 'rect' or 'ellipse'.  Ignored if apply_mask is
        False.
    mask_fwhm : float, optional
        FWHM of the mask, specified as a proportion of the image dimensions.
        Ignored if apply_mask is False.
    wisdom_filetpath : str or None, optional
        If not None, either path to existing pyfftw wisdom  pickle file, or
        desired path for such a file to be made.  If file exists, class will
        load in pre-existing wisdom so that the pyfftw planning stage may be
        skipped, allowing much faster initilisation of the class.  If the file
        doesn't exist, the class will caclulate new pyfftw wisdom and attempt
        to save it to this file.  If None, or if the string points a file but
        one which contains the wrong type of wisdom for the specified transform
        (e.g. wrong image dimensions) then new wisdom will be calculated but
        will not be saved out.
    planner_effort : str, optional
        One of the valid strings for the pyfftw planner effort levels. More
        extensive plans take longer to set up (unless a pre-existing wisdom
        file is loaded), but allow faster execution of Fourier transfoms once
        set up.  Default is the most extensive level of effort.
    output_dtype : valid numpy datatype, optional
        Datatype for output image.  Note that attempting to apply a mask will
        not work if this is anything other than uint8.
    output_range : (min,max) tuple or None, optional
        Range of luminance values to clip output image within. If None, no
        clipping is performed.


    Class methods
    -------------
    .filter
        Function applies filtering to a given image.


    Example usage
    -------------
    Get an example image.

    >>> import imageio
    >>> im = imageio.imread('imageio:coffee.png', as_gray=True)

    Create a high-pass filter with a FWHM of 50 cycles/image.  Note that the
    filter is designed for a "real" Fourier transform.

    >>> sigma = fwhm2sigma(50)
    >>> filt = createFourierFilter(
    ...     imsize=im.shape, mode='sf', filter_type='gaussian',
    ...     filter_kwargs={'mu':0, 'sigma':sigma}, invert=True, real=True
    ...     )

    Create an instance of the filtering class, passing our filter and also a
    pre-existing wisdom file.

    >>> from utils.imageprocessing import pyfftw_FourierFilter
    >>> filterer = pyfftw_FourierFilter(imsize=im.shape, filt=filt,
    ...                                 wisdom_filepath='./pyfftw_wisdom.pkl')

    Filter the image.

    >>> filtim = filterer.filter(im)

    Display

    >>> import matplotlib.pyplot as plt
    >>> fig, (ax1,ax2) = plt.subplots(nrows = 2)
    >>> plt.gray()
    >>> ax1.imshow(im, interpolation = 'nearest', vmin = 0, vmax = 255)
    >>> ax1.set_title('Original')
    >>> ax2.imshow(filtim, interpolation = 'nearest', vmin = 0, vmax = 255)
    >>> ax2.set_title('Filtered')
    >>> plt.show()

    """
    def __init__(self, imsize, filt, apply_mask=True, mask_shape='rect',
                 mask_fwhm=0.95, wisdom_filepath=None,
                 planner_effort='FFTW_EXHAUSTIVE', output_dtype=np.uint8,
                 output_range=(0,255)):
        self.imsize = imsize
        self.filt = filt
        self.apply_mask = apply_mask
        self.mask_shape = mask_shape
        self.mask_fwhm = mask_fwhm
        self.wisdom_filepath = wisdom_filepath
        self.planner_effort = planner_effort
        self.output_dtype = output_dtype
        self.output_range = output_range

        # Make sure image dimensions even
        if any(np.mod(self.imsize[:2], 2)):
            raise RuntimeError('Image dimensions must be even numbers')

        # Pad filter with trailing dim if colour image
        if len(self.imsize) == 3 and self.filt.ndim == 2:
            self.filt = self.filt[..., np.newaxis]

        # Work out size of spectrum
        spec_size = np.copy(self.imsize)
        spec_size[1] = spec_size[1]//2 + 1

        # Allocate byte aligned arrays for input / output images, filter,
        # and spectrum
        self.im_arr = pyfftw.empty_aligned(self.imsize, dtype=np.float32)
        self.filt_arr = pyfftw.empty_aligned(self.filt.shape, dtype=np.float32)
        self.F_arr = pyfftw.empty_aligned(spec_size, dtype=np.complex64)

        # Assign filter into btye aligned array
        self.filt_arr[:] = self.filt

        # Try to load in existing widsom file
        if self.wisdom_filepath is not None:
            try:
                with open(self.wisdom_filepath, 'rb') as fd:
                    pyfftw.import_wisdom(pickle.load(fd))
                self.existing_wisdom = True
                print('Loaded existing wisdom!')
            except IOError:
                self.existing_wisdom = False
                print('Could not load existing wisdom, will generate now\n'
                      'Be warned - this could take a while!')

        # Create pyfftw builder objects
        self.fft_obj = pyfftw.builders.rfft2(
                self.im_arr, axes=(0,1), overwrite_input=True,
                planner_effort=self.planner_effort
                )
        self.ifft_obj = pyfftw.builders.irfft2(
                self.F_arr, axes=(0,1),  planner_effort=self.planner_effort
                )

        # Save wisdom back out if we failed to load it earlier
        if self.wisdom_filepath is not None and not self.existing_wisdom:
            try:
                with open(self.wisdom_filepath, 'wb') as fd:
                    pickle.dump(pyfftw.export_wisdom(), fd)
            except Exception as e:
                print(e)
                print('Failed to save new wisdom, continuing anyway')

        # Create softwindow mask for image.  Function returns a "true" mask (in
        # the sense it ranges from 0-1 and can be used to weight image by), but
        # we're going to use it as the image alpha channel instead as this is
        # faster than weighting, so scale it up to 0-255 and cast to uint8
        if self.apply_mask:
            _mask = createSoftWindowMask(
                    self.imsize[:2], maskshape=mask_shape, fwhm=mask_fwhm
                    )
            self.mask = (_mask * 255).astype(np.uint8)

    def filter(self, im):
        """
        Apply filter to given image and return filtered array.  Can be passed
        to postproc argument of utils.videostreaming.VideoStream class.

        Arguments
        ---------
        im : numpy.ndarray
            Input image as a numpy array.

        Returns
        -------
        filtim : numpy.ndarray
            Filterted image as a numpy array with uint8 datatype.
            If apply_mask is True, image will be in RGBA space where the alpha
            channel is used to encode the soft-window mask, otherwise the image
            will be in its original space.
        """
        # Allocate image to array
        self.im_arr[:] = im

        # Apply forward transform
        self.fft_obj(self.im_arr, self.F_arr)

        # Weight by filter
        self.F_arr *= self.filt_arr

        # Apply inverse transform
        self.ifft_obj(self.F_arr, self.im_arr)

        # Grab image back out of array, clip and cast to specified dtype
        filtim = self.im_arr[:]
        if self.output_range is not None:
            filtim = filtim.clip(*self.output_range)
        filtim = filtim.astype(self.output_dtype)

        # If masking, convert to RGBA image using mask as alpha channel
        if self.apply_mask:
            if filtim.ndim == 2:  # mono -> RGBA
                filtim = cv2.merge(3*[filtim] +  [self.mask])
            else:  # RGB -> RGBA
                filtim = cv2.merge([filtim, self.mask])

        # Return
        return filtim

    def setFilter(self, filt):
        """
        Set filter array.
        """
        if self.filt_arr.ndim == 3 and filt.ndim == 2:
            filt = filt[..., np.newaxis]
        self.filt = filt
        self.filt_arr[:] = filt



### Function definitions

def fwhm2sigma(fwhm):
    """
    Converts a full-width-half-maximum value to a sigma value
    """
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def sigma2fwhm(sigma):
    """
    Converts a sigma value to a full-width-half-maximum value.
    """
    return sigma * (2 * np.sqrt(2 * np.log(2)))


def butterworth(X, cutoff, order, mu=0, cutin=None):
    """
    Returns Butterworth filter.

    Arguments
    ---------
    X : numpy.ndarray
        Range of values to plot over.
    cutoff : float
        Cut-off value for function.
    order : int
        Order of function - higher values give steeper descent after cutoff.
    mu : float
        Value to centre function about.
    cutin : float
        If not None, provides a cut-in value for the function.  This allows
        construction of a bandpass filter.  The value should therefore be less
        than the value of the cut-off.

    """
    # Sub-func for butterworth transfer function
    def _butter(X, cutoff, order, mu):
        return 1 / (1 + ( (X-mu) / cutoff)**(2*order))

    # Create cutoff filter
    B = _butter(X, cutoff, order, mu)

    # If cutin, create 2nd filter and subtract from 1st one
    if cutin is not None:
        B2 = _butter(X, cutin, order, mu)
        B -= B2

    # Return
    return B


def gaussian(X, sigma, mu=0):
    """
    Returns Gaussian filter.

    Arguments
    ---------
    X : numpy.ndarray
        Range of values to plot over.
    sigma : float
        Standard deviation of Gaussian.
    mu : float
        Value to centre Gaussian on.
    """
    return np.exp(-0.5 * ((X-mu)/sigma)**2)


def createFourierFilter(imsize, mode, filter_type, filter_kwargs={},
                        invert=False, real=True):
    """
    Makes filter of spatial frequency or orientation, to be applied in Fourier
    domain.

    Arguments
    ---------
    imsize : tuple, required
        (length, width) tuple indicating size of spectrum.  Any trailing values
        beyond the first 2 are ignored.
    mode : 'sf' or 'ori', required
        Use strings 'sf' or 'ori' to indicate to make a spatial frequency or
        an orientation filter respectively.
    filter_type : str or callable, required
        Type of filter to use.  Available options are:
         * 'gaussian' to use a Gaussian filter
         * 'butterworth' to use a Butterworth filter
        Alternatively, may specify a custom callable function.  This function
        should take a numpy arrary mapping the values (either spatial frequency
        or orientation) of the Fourier spectrum as its first argument
        (see also createFourierMaps function).  Spatial frequency values should
        be given in units of cycles/image, whilst orientation values should be
        given in radians and should be in the interval 0:pi.  Further keyword
        arguments to the function may be passed using the filter_kwargs
        argument of this function.
    filter_kwargs : dict or list of dicts, optional
        Dictionary of additional keyword arguments to be passed to filter
        function. Keys should indicate argument names, values should indicate
        argument values.  If a list of dicts is provided, a separate filter
        will be created for each set of paramemters, and all of them summed
        together to create a composite filter - this is useful to create a
        filter with multiple components, such as a cardinal or an oblique
        orientation filter.  In general, units should be given in cycles/image
        for a spatial frequency filter, and in units of radians in the interval
        0:pi for an orientation filter.  If using a named filter
        (e.g. 'gaussian'), see help information of the relevant base
        function for details of which keyword arguments are required.
    invert : bool, optional
        Set invert = True to invert filter, e.g. to make a high-pass filter
        (default = False).
    real : bool, optional
        Whether filter should be for real fft (rfft2) or full fft (fft2).

    Returns
    -------
    filt : numpy.ndarray
        Requested filter as numpy array.

    Example usage
    -------------

    High-pass spatial frequency 5th order Butterworth filter with cut-off at
    30 cycles/image for a 512x512 pixel image.

    >>> filt = createFourierFilter(
    ...     imsize=(512,512), mode='sf', filter_type='butterworth',
    ...     filter_kwargs={'cutoff':30, 'order':5}, invert=True
    ...     )

    Cardinal-pass orientation Gaussian filter with 30 degree FWHM for a
    512x512 pixel image.

    >>> from utils.imageprocessing import fwhm2sigma
    >>> fwhm = np.radians(30)
    >>> sigma = fwhm2sigma(fwhm)
    >>> filt = createFourierFilter(
    ...     imsize=(512,512), mode='ori', filter_type='gaussian',
    ...     filter_kwargs=[{'mu':np.radians(0), 'sigma':sigma},
    ...                    {'mu':np.radians(90), 'sigma':sigma}]
    ...     )

    """
    # Ensure appropriate mode
    if mode not in ['sf', 'ori']:
        raise ValueError('Mode must be \'sf\' or \'ori\'')

    # Assign appropriate filter function
    if isinstance(filter_type, str):
        filter_type = filter_type.lower() # ensure case insensitive
        if filter_type == 'gaussian':
            filter_func = gaussian
        elif filter_type == 'butterworth':
            filter_func = butterworth
        else:
            raise ValueError('Unrecognised filter type')
    elif callable(filter_type):
        filter_func = filter_type
    else:
        raise TypeError('filter_type must be allowable string or callable')

    # If filter_kwargs is dict, only one has been passed so coerce to list
    if isinstance(filter_kwargs, dict):
        filter_kwargs = [filter_kwargs]

    # Create spatial frequency or orientations map
    X = createFourierMaps(imsize, real)[mode]

    # Pre-allocate filter, including trailing dimension for each sub-filter
    filt = np.empty( X.shape + (len(filter_kwargs),) )

    # Loop through filter_kwargs
    for i, this_filter_kwargs in enumerate(filter_kwargs):
        # If doing spatial frequency, just make filter and allocate to array
        if mode == 'sf':
            filt[..., i] = filter_func(X, **this_filter_kwargs)
        # If doing orientation, make 3 filters to include +/- pi rad, then sum
        else:
            tmp = [filter_func(X - offset, **this_filter_kwargs) for offset \
                   in [-pi, 0, pi]]
            filt[..., i] = np.sum(tmp, axis = 0)

    # Sum filters along last dimension to create composite
    filt = filt.sum(axis = -1)

    # Scale into range 0-1
    filt /= filt.max() # scale into range 0-1

    # Invert if requested
    if invert:
        filt = 1 - filt

    # Add in DC
    filt[0,0] = 1

    # Return
    return filt


def createFourierMaps(imsize, real=True):
    """
    Returns spatial frequency and orienation maps for a Fourier spectrum.

    Arguments
    ---------
    imsize : tuple, required
        (length, width) tuple indicating size of spectrum.  And trailing values
        beyond the first 2 are ignored.
    real : bool, optional
        If True (default), maps are made for a half spectrum, e.g. for use
        with numpy.fft.rffft2.  If False, returns full maps.

    Returns
    -------
    maps - dict
        Dictionary with keys 'sf' and 'ori', containing the spatial frequency
        and orientation maps respectively.

    """
    # Get length and width
    L,W = imsize[:2]

    # Make sure dims are even numbers
    if any(np.mod([L,W], 2)):
        raise RuntimeError('Image dimensions must be even numbers')

    # Make meshgrid
    Wrng = np.fft.ifftshift(np.arange(-W//2, W//2))
    Lrng = np.fft.ifftshift(np.arange(-L//2, L//2))
    if real:
        Wrng = Wrng[:1+W//2]
    [fx,fy] = np.meshgrid(Wrng, Lrng)

    # Create maps, return
    return {'sf':np.sqrt(fx**2 + fy**2), 'ori':np.arctan2(fx,fy) % pi}


def createSoftWindowMask(imsize, maskshape, fwhm=0.95):
    """
    Create soft-window mask.  Can be used to window images with a soft edge.

    Arguments
    ---------
    imsize : (L, W) tuple
        Size of image.  Any further values beyond L and W are ignored.
    maskshape : {'ellipse', 'rect'}
        Desired shape of mask (elliptical or rectangular)
    fwhm : float, optional
        Value in range 0-1; dimensions of the mask as a proportion of the
        dimensions of the image (default = 0.9)

    Returns
    -------
    mask : ndarray
        Mask as numpy array with datatype float.

    """
    # Extract image length and width for brevity
    L, W = imsize[:2]

    # Create ideal mask
    if maskshape == 'ellipse':
        x_radius = W * fwhm / 2
        y_radius = L * fwhm / 2
        [fx,fy] = np.meshgrid(range(W),range(L))
        mask = ( ((fx - W/2)/x_radius)**2 + ((fy-L/2)/y_radius)**2 ) < 1
        mask = mask.astype(float)
    elif maskshape == 'rect':
        mask = np.zeros([L,W])
        x_bordW = int(W * (1 - fwhm) // 2)
        y_bordW = int(L * (1 - fwhm) // 2)
        mask[y_bordW:-(y_bordW), x_bordW:-(x_bordW)] = 1
    else:
        raise ValueError('Unrecognised argument to \'maskshape\'')

    # Convolve mask with cosine kernels to give blurred edge
    for axis, size in enumerate([L, W]):
        blur = int(round(size * (1 - fwhm)))
        kernel = np.cos(np.linspace(-pi/2, pi/2, blur))
        kernel /= kernel.sum()
        mask = scipy.ndimage.convolve1d(mask, kernel, axis)

    # Return
    return mask
