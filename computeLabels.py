import numpy as np

def computeLabels(files):

    numImageAll = len(files)
    labels = np.zeros(shape=(len(files), 1), dtype=int)
    for i in range(numImageAll):
        fileName = files[i]
        labels[i] = int(fileName[:4])

    return labels, numImageAll