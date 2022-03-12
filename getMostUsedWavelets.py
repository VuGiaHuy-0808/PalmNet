from initTmpRes import initTmpRes
import numpy as np

def getMostUsedWavelets(sortRes, gaborBank, powerMapF, param, sizeRes):

    # Init matrix indicating if for each position a wavelet with orient and scale already used
    tmpRes = initTmpRes(gaborBank, powerMapF)

    # Init counter for how many times each wavelet is chosen
    o_counter = np.zeros((len(gaborBank.even), 1))

    # Init counters
    countW = 1 # Wavelet counter
    ind = 0 # Sorted response counter

    while (countW <= param.numWavelets and ind <= sizeRes):

        # Get information of corresponding wavelet
        currX = int(sortRes[ind, 3])
        currY = int(sortRes[ind, 4])
        currO = int(sortRes[ind, 5])

        # Check if current wavelet at current position already
        if (tmpRes[currO][currY, currX] == 1):
            # increment counter
            ind += 1
            continue

        # increment by 1 the corresponding wavelet index counter
        o_counter[currO] += 1
        tmpRes[currO][currY, currX] = 1

        # increment counter
        countW += 1
        ind += 1

    return o_counter