import numpy as np
from gaborArrayFromScales import gaborArrayFromScales
from gaborArrayParametrized import gaborArrayParametrized
from referenceGaborFilter import referenceGaborFilter
from computePowerMaps import computePowerMaps
from sortFilterResponse import sortFilterResponse
from getMostUsedWavelets import getMostUsedWavelets

class GaborBank():
    even = []
    odd = []
    scale = []
    theta = []

def findBestWaveletsTesting(imagesCell, orient_default, orient_best, numImagesToUse, imageSize, param):

    # Init
    bestWaveletAll = []

    #------------------------------
    # Compute parametrized wavelets
    # Create fixed-scale Gabor filter
    parametrizedWavelets = gaborArrayParametrized(param.sigma, param.wavelength, np.deg2rad(orient_default))

    #------------------------------
    # Compute complete multi-scale Gabor filter
    gaborBank = gaborArrayFromScales(imageSize, np.deg2rad(np.unique([orient_default, orient_best])), param)

    # Init counter for how many times each wavelet is chosen
    o_counterAll = np.zeros((len(gaborBank.even), 1))

    for j in range(1):

        im = imagesCell[j]

        #-----------------------------
        # Perform reference Gabor filtering
        resF = referenceGaborFilter(gaborBank, im)

        #-----------------------------
        # Create powerMap for all filters
        powerMapF = computePowerMaps(resF, gaborBank)

        #-----------------------------
        # Sort filter response information
        sortRes, sizeRes = sortFilterResponse(gaborBank, powerMapF)

        #-----------------------------
        # Compute most used wavelets
        o_counter = getMostUsedWavelets(sortRes, gaborBank, powerMapF, param, sizeRes)

        # Increment
        o_counterAll += o_counter

    #------------------------------
    # Sort most used filters counter
    o_counterAll = np.transpose(np.array(o_counterAll))
    ind_o_counter_All_sort = np.argsort(o_counterAll[0])[::-1]

    # Consider only best wavelets
    ind_o_counter_All_sort = ind_o_counter_All_sort[ : param.numBestWavelets]
    bestWavelets = GaborBank
    for i in range(param.numBestWavelets):
        bestWavelets.even.append(gaborBank.even[ind_o_counter_All_sort[i]])
        bestWavelets.odd.append(gaborBank.odd[ind_o_counter_All_sort[i]])
        bestWavelets.scale.append(gaborBank.scale[ind_o_counter_All_sort[i]])
        bestWavelets.theta.append(gaborBank.theta[ind_o_counter_All_sort[i]])

    # -----------------------------
    # assign to global structure
    countFilter = 1

    # parametrized
    for o in range(len(parametrizedWavelets)):
        bestWaveletAll.append(parametrizedWavelets[o])
        countFilter += 1

    # dynamic
    for o in range(len(bestWavelets.even)):
        bestWaveletAll.append(bestWavelets.even[o] + 1j * bestWavelets.odd[o])
        countFilter += 1

    return bestWaveletAll