import numpy as np

def myConv2(h, tmpX, step):

    # preparation. X consists of original image surounded by zeros
    tmpImageSize = tmpX.shape
    filterSize = h.shape

    numFilter = np.ceil(((tmpImageSize[0] - filterSize[0]) / 2 + 1) / step) * 2 + 1

    imageSize = int(step * (numFilter - 1) + filterSize[0])
    X = np.zeros((imageSize, imageSize))
    start = int(imageSize / 2 - int(tmpImageSize[0])/2)
    stop = int(imageSize / 2 + int(tmpImageSize[0])/2)
    X[start:stop, start:stop] = tmpX

    startingPoints = []
    startingPoints.append(np.linspace(0, imageSize - filterSize[0],int((imageSize - filterSize[0] + 1)/step)))
    startingPoints.append(np.linspace(0, imageSize - filterSize[1],int((imageSize - filterSize[1] + 1)/step)))

    res = np.zeros((len(startingPoints[0]), len(startingPoints[0])))

    for ii in range(len(startingPoints[0])):
        for jj in range(len(startingPoints[0])):
            offset1 = startingPoints[0][ii]
            offset2 = startingPoints[1][jj]

            start = 0
            end = filterSize[0]

            XIndStart = int(offset1 + start)
            XIndEnd = int(offset1 + end)
            YIndStart = int(offset2 + start)
            YIndEnd = int(offset2 + end)

            tmpX = X[XIndStart : XIndEnd, YIndStart : YIndEnd]
            dotProd = np.sum(np.multiply(tmpX, h))

            res[ii,jj] = dotProd

    return res