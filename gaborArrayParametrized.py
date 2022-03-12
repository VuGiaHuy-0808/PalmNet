import math
import numpy as np

def gaborArrayParametrized(sigma, wavelength, orient_chosen):

    # Bounding box
    halfLength = 17

    xmax = halfLength
    xmin = -halfLength
    ymax = halfLength
    ymin = -halfLength

    h, w = (35,35)
    xw = np.linspace(xmin, xmax, w)
    yh = np.linspace(ymin, ymax, h)
    x,y = np.meshgrid(xw, yh)

    # Init
    gabor = []

    for oriIndex in range(len(orient_chosen)):

        theta = orient_chosen[oriIndex]
        x_theta = x
        y_theta = y
        gb = (1 / (2 * math.pi * (math.pow(sigma, 2)))) * np.exp(-(np.power(x_theta, 2) + np.power(y_theta, 2)) /
            (2 * math.pow(sigma, 2))) * np.exp(2 * math.pi * 1j * (wavelength * x_theta * math.cos(theta) + 
            wavelength * y_theta * math.sin(theta)))
        gabor.append(gb)

    return gabor