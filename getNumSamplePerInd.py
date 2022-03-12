import numpy as np
import statistics

def getNumSamplePerInd(files):

    # Loop on files to compute number of samples for each individual
    numSamplePerInd = np.zeros(shape=(len(files), 1), dtype=int)
    for i in range(len(files)):
        fileName = files[i]
        ind = int(fileName[:4])
        numSamplePerInd[ind] += 1

#     remove '0' element in numSamplePerInd
    indZero = np.where(numSamplePerInd == 0)
    numSamplePerInd = np.delete(numSamplePerInd, indZero)
    numSampleMean = round(statistics.mean(numSamplePerInd))

    return numSampleMean
