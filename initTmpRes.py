import numpy as np


def initTmpRes(gaborBank, powerMapF):

    tmpRes = []
    for i in range(len(gaborBank.even)):
        # get size of current powerMap
        currentSize = powerMapF[i].shape
        tmpRes.append(np.zeros(currentSize))

    return tmpRes