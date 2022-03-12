import random
import numpy as np

def personFold (numSamples, labels):

    foldPerPerson = np.random.randint(1, 3, (max(labels)[0], 1))

    # init
    Indexes = np.zeros(shape=(numSamples, 1), dtype= int)

    for i in range(numSamples):
        fold = foldPerPerson[labels[i] - 1]
        Indexes[i] = fold

    return Indexes