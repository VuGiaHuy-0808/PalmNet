from personFold import personFold
import numpy as np

def computeIndexesPersonFold(numImagesAll, labels):

    indexesFold = personFold(numImagesAll, labels)
    indImagesTrain = np.zeros(indexesFold.shape, dtype= int)
    indImagesTest = np.zeros(indexesFold.shape, dtype= int)
    for i in range(len(indexesFold)):
        if indexesFold[i] == 1:
            indImagesTrain[i] = 1
        else: indImagesTest[i] = 1
    allIndexes = np.linspace(0, numImagesAll - 1, numImagesAll)
    numImagesTrain = np.count_nonzero(indImagesTrain)
    numImagesTest = np.count_nonzero(indImagesTest)

    return indexesFold, allIndexes, indImagesTrain, indImagesTest, numImagesTrain, numImagesTest