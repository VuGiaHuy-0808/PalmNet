
# GAUSSFILT -  Small wrapper function for convenient Gaussian filtering
#
# Usage:  smim = gaussfilt(im, sigma)
#
# Arguments:   im - Image to be smoothed.
#           sigma - Standard deviation of Gaussian filter.
#
# Returns:   smim - Smoothed image.
#
# If called with sigma = 0 the function immediately returns with im assigned
# to smim

from scipy import signal
import numpy as np
import math

def matlab_style_gauss2D(shape ,sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def gaussfilt(im, sigma):

    smim = np.zeros(shape= im.shape)
    if sigma < np.finfo(float).eps:
        smim = im

    # If needed convert im to double
    if type(im[0][0]) != np.float64:
        im = im.astype(np.float64)

    sze = max(math.ceil(6 * sigma), 1)

    if (sze % 2) == 0:
        sze += 1

    h = matlab_style_gauss2D((sze, sze), sigma);
    # Apply filter to all image channels
    smim = signal.convolve2d(im, np.rot90(h, 2), mode='same')

    return smim