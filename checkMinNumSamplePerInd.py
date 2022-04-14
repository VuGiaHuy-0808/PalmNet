import numpy as np

def checkMinNumSamplePerInd (files):

    indexrem = []
    numSamplePerInd = np.zeros(shape=(len(files), 1), dtype= int)

#    loop on files to compute num of samples for each ind
    for i in range(len(files)):
        fileName = files[i]
        ind = int(fileName[:4])
        numSamplePerInd[ind] += 1

#     remove '0' element in numSamplePerInd
    indZero = np.where(numSamplePerInd == 0)
    numSamplePerInd = np.delete(numSamplePerInd, indZero)

#     loop again to remove samples without minimum number
    for i in range(len(files)):
        fileName = files[i]
        ind = int(fileName[:4])
        if numSamplePerInd[ind - 1] == 1:
            indexrem.append(i)

    for i in range(len(indexrem)):
        files.pop(indexrem[i])