import math
from derivative5 import derivative5
from gaussfilt import gaussfilt
import numpy as np
from scipy import signal

# RidgeOrient - Estimates the local orientation of riges in palmprint
# Usage: orientim, reliability, coherence = ridgeorientation[im, gradientsigma, blocksigma, orientsmoothsigma]
#
# Args:       im                    - A normalized input image
#             gradientsigma         - Sigma of the derivative of Gaussian used to compute image gradients
#             blocksigma            - Sigma of the Gaussian weighting used to sum the gradient moments
#             orientsmoothsigma     - Sigma of the Gaussian used to smooth the final orientation vector field
#                                     Default = 0
#
# Returns:    orientim              - The orientation image in radians
#             reliability           - Measure of the reliability of the orientation measure. This is a value between 0
#                                     and 1. A value above 0.5 can be considered 'reliable'
#             coherence             - A measure of the degree to which the local area is oriented

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

def ridgeorient(im, gradientsigma, blocksigma, orientsmoothsigma):

    rows = im.shape[0]
    cols = im.shape[1]

    # Calculate image gradients
    varargin = ['x','y']
    gradi = derivative5(gaussfilt(im, gradientsigma), varargin)
    Gx = gradi[0]
    Gy = gradi[1]

    # Estimate the local ridge orientation at each point by finding the
    # principal axis of variation in the image gradients.

    # Covariance data for the image gradients
    Gxx = np.power(Gx, 2)
    Gxy = Gx * Gy
    Gyy = np.power(Gy, 2)

    # Now smooth the covariance data to perform a weighted summation of the
    # data.
    sze = np.fix(6 * blocksigma)
    if sze % 2 == 0:
        sze += 1
    f = matlab_style_gauss2D((sze, sze), blocksigma)
    Gxx = signal.convolve2d(Gxx, np.rot90(f, 2), mode='same')
    Gxy = 2 * (signal.convolve2d(Gxy, np.rot90(f, 2), mode='same'))
    Gyy = signal.convolve2d(Gyy, np.rot90(f, 2), mode='same')

    # Analytic solution of principal direction
    denom = np.sqrt(Gxy * Gxy + (Gxx - Gyy) * (Gxx - Gyy)) + np.finfo(float).eps
    # Sine and cosine of doubled angles
    sin2theta = Gxy/denom
    cos2theta = (Gxx - Gyy)/denom

    if orientsmoothsigma:
        sze = np.fix(6 * orientsmoothsigma)
        if sze % 2 == 0:
            sze += 1
        f = matlab_style_gauss2D((sze, sze), orientsmoothsigma)
        # Smoothed sine and cosine of doubled angles
        cos2theta = signal.convolve2d(cos2theta, np.rot90(f, 2), mode='same')
        sin2theta = signal.convolve2d(sin2theta, np.rot90(f, 2), mode='same')

    orientim = math.pi / 2 + np.arctan2(sin2theta, cos2theta) / 2

    return orientim