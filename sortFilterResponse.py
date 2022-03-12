import numpy as np

def sortFilterResponse(gaborBank, powerMapF):

    # Put powermap info in 1d vector
    # Get number of pixels in powermap
    sizeTmp = 0
    for g in range(len(gaborBank.even)):
        s = powerMapF[g].shape
        sizeTmp += s[0] * s[1]

    tmp = np.zeros((sizeTmp, 6))

    # Init counter
    ind = 0

    for o in range(len(powerMapF)):

        # get size of current powermap
        currentSize = powerMapF[o].shape

        for rows in range(currentSize[0]):
            for cols in range(currentSize[1]):
                tmp[ind, 0] = powerMapF[o][rows, cols]
                tmp[ind, 1] = gaborBank.scale[o]
                tmp[ind, 2] = gaborBank.theta[o]
                tmp[ind, 3] = cols
                tmp[ind, 4] = rows
                tmp[ind, 5] = o
                ind += 1

    # sort powerMap in descending order
    tempPowerMap = tmp[:,0]
    indSort = np.argsort(tempPowerMap)[::-1]
    B = tempPowerMap[indSort]
    sortRes = np.zeros(tmp.shape)
    sortRes[:, 0] = B
    sortRes[:, 1] = tmp[indSort, 1]
    sortRes[:, 2] = tmp[indSort, 2]
    sortRes[:, 3] = tmp[indSort, 3]
    sortRes[:, 4] = tmp[indSort, 4]
    sortRes[:, 5] = tmp[indSort, 5]
    sizeRes = sortRes.shape[0]

    return sortRes, sizeRes
