
# DERIVATIVE5 - 5-Tap 1st and 2nd discrete derivatives
#
# This function computes 1st and 2nd derivatives of an image using the 5-tap
# coefficients given by Farid and Simoncelli.  The results are significantly
# more accurate than MATLAB's GRADIENT function on edges that are at angles
# other than vertical or horizontal. This in turn improves gradient orientation
# estimation enormously.  If you are after extreme accuracy try using DERIVATIVE7.
#
# Usage:  [gx, gy, gxx, gyy, gxy] = derivative5(im, derivative specifiers)
#
# Arguments:
#                       im - Image to compute derivatives from.
#    derivative specifiers - A comma separated list of character strings
#                            that can be any of 'x', 'y', 'xx', 'yy' or 'xy'
#                            These can be in any order, the order of the
#                            computed output arguments will match the order
#                            of the derivative specifier strings.
# Returns:
#  Function returns requested derivatives which can be:
#     gx, gy   - 1st derivative in x and y
#     gxx, gyy - 2nd derivative in x and y
#     gxy      - 1st derivative in y of 1st derivative in x

import numpy as np

def conv2(p, d, im, mode='same'):
    """
    Two-dimensional convolution of matrix im by vectors p and d

    First convolves each column of 'im' with the vector 'p'
    and then it convolves each row of the result with the vector 'd'.

    """
    tmp = np.apply_along_axis(np.convolve, 0, im, p, mode)
    return np.apply_along_axis(np.convolve, 1, tmp, d, mode)

def derivative5(im, varargin):

    varargout = []

    # Check if we are just computing 1st derivatives.  If so use the
    # interpolant and derivative filters optimized for 1st derivatives, else
    # use 2nd derivative filters and interpolant coefficients.
    # Detection is done by seeing if any of the derivative specifier
    # arguments is longer than 1 char, this implies 2nd derivative needed.

    secondDeriv = bool(False)
    for n in range(len(varargin)):
        if len(varargin[n]) > 1:
            secondDeriv = bool(True)

    if ~secondDeriv:
        # 5 tap 1st derivative cofficients.  These are optimal if you are just
        # seeking the 1st deriavtives
        p = [0.037659, 0.249153, 0.426375, 0.249153, 0.037659]
        d1 = [0.109604, 0.276691, 0.000000, -0.276691, -0.109604]
    else:
        # 5-tap 2nd derivative coefficients. The associated 1st derivative
        # coefficients are not quite as optimal as the ones above but are
        # consistent with the 2nd derivative interpolator p and thus are
        # appropriate to use if you are after both 1st and 2nd derivatives.
        p = [0.030320, 0.249724, 0.439911, 0.249724, 0.030320]
        d1 = [0.104550, 0.292315, 0.000000, -0.292315, -0.104550]
        d2 = [0.232905, 0.002668, -0.471147, 0.002668, 0.232905]

    gx = bool(False)

    for n in range(len(varargin)):
        if varargin[n] == 'x':
            varargout.append(conv2(p, d1, im, 'same'))
            gx = bool(True) # Record that gx is available for gxy if needed
            gxn = n
        elif varargin[n] == 'y' :
            varargout.append(conv2(d1, p, im, 'same'))
        elif varargin[n] == 'xx':
            varargout.append(conv2(p, d2, im, 'same'))
        elif varargin[n] == 'yy':
            varargout.append(conv2(d2, p, im, 'same'))
        elif varargin[n] == 'xy' or varargin[n] == 'yx':
            if gx:
                varargout.append(conv2(d1, p, varargout[gxn], 'same'))
            else:
                gx = conv2(p, d1, im, 'same')
                varargout.append(conv2(d1, p, gx, 'same'))

    return varargout


