import numpy as np
import math
from computeFilterSizeFromScale import computeFilterSizeFromScale

class GaborBank:
    even = []
    odd = []
    scale = []
    theta = []

def gaborArrayFromScales(imageSize, thetaVector, param):

    # Scaling factor
    a0 = 2 ** (1 / param.N)

    # The number of scale (in 6 octaves)
    m = np.ceil(math.log2(imageSize[0] / 2)) * param.N

    # kai
    kai = math.sqrt(2 * math.log(2)) * (2 ** param.phai + 1) / (2 ** param.phai - 1)

    # Init
    # Gabor filter counter
    gaborFilter = GaborBank

    for ii in range(int(m)):

        # Compute filterSize from scale
        filterSize = computeFilterSizeFromScale(ii)

        # Coefficient dependent on scale
        ctr = (4 + 2 ** (-ii)) / 2

        # Create meshgrid to compute filter
        tx = np.linspace(1, filterSize, filterSize)
        ty = np.linspace(1, filterSize, filterSize)
        x, y = np.meshgrid(tx,ty)

        # Multiply meshgrid with coefficient
        xx = a0 ** (-ii) * x - ctr * param.b0
        yy = a0 ** (-ii) * y - ctr * param.b0

        for ll in range(len(thetaVector)):

            # Angle
            theta = thetaVector[ll]

            # Rotate meshgrid with angle theta
            rotX = xx * math.cos(theta) + yy * math.sin(theta)
            rotY = -xx * math.sin(theta) + yy * math.cos(theta)

            # Compute filters
            # Even
            gb_even = np.real(a0 ** (-ii) * (1 / math.sqrt(2) * np.exp(-1 / (2 * (param.aspectRatio ** 2)) *
                    ((param.aspectRatio ** 2) * np.power(rotX, 2) + np.power(rotY, 2)))
                    * (np.exp(1j * kai * rotX) - np.exp(-(kai ** 2)/2))))
            # Odd
            gb_odd = np.imag(a0 ** (-ii) * (1 / math.sqrt(2) * np.exp(-1 / (2 * (param.aspectRatio ** 2)) *
                    ((param.aspectRatio ** 2) * np.power(rotX, 2) + np.power(rotY, 2)))
                    * np.exp(1j * kai * rotX)))

            # Save filter and related info
            gaborFilter.even.append(gb_even)
            gaborFilter.odd.append(gb_odd)
            gaborFilter.scale.append(ii)
            gaborFilter.theta.append(theta)

    return gaborFilter