
import glob
import time
import numpy as np
from loadImages import loadImages
from checkMinNumSamplePerInd import checkMinNumSamplePerInd
from computeLabels import computeLabels
from getNumSamplePerInd import getNumSamplePerInd
from computeIndexesPersonFold import computeIndexesPersonFold
from adjustFormat import adjustFormat
from searchGaborOrientation import searchGaborOrientation
from findBestWaveletsTesting import findBestWaveletsTesting
from featExtrGaborAdapt import featExtrGaborAdapt
import paramsPalmNet as param

# --------------------------------
# General parameters
numCoresKnn = 2
stepPrint = 100

# --------------------------------
# Directory of DataBase
ext = 'bmp'
dbName = 'Tongji_Contactless_Palmprint_Dataset'
dirDB = 'D:\PalmNet\PalmNet-master\images\Tongji_Contactless_Palmprint_Dataset'
path = dirDB + '/*.' + ext

# --------------------------------
# DB processing
# Extract samples
files = []
for name in glob.glob(path):
    cutName = name.split('Dataset')
    files.append(cutName[1][1:])

# Check that there is at least one sample for each individual
checkMinNumSamplePerInd(files)
# Compute labels
labels, numImageAll = computeLabels(files)
# Compute number of individuals
numInd = len(np.unique(labels))
# Compute number of samples per individual
numSampleMean = getNumSamplePerInd(files)

# ------------------------------------
# Display
print('\n')
print('Extracting samples...')
print(str(numInd) + ' individuals')
print(str(numSampleMean) + ' samples per individual, on average')
print(str(numImageAll) + ' images in total')
print('\n')

# Loop on iterations
for r in range(param.numIterations):

    # -----------------------------
    # Display
    print('Iteration : ' + str(r) + '\n')

    # -----------------------------
    # Compute random person-fold indexes
    indexesFold, allIndexes, indImagesTrain, indImagesTest, numImageTrain, numImageTest = computeIndexesPersonFold(numImageAll, labels)
    # Corresponding labels
    TrnLabels = []
    TestLabels = []
    for i in range(len(indImagesTrain)):
        if indImagesTrain[i] == 1:
            TrnLabels.append(labels[i][0])
    for i in range(len(indImagesTest)):
        if indImagesTest[i] == 1:
            TestLabels.append(labels[i][0])

    # -------------------------------
    # Display
    print(str(numImageTrain) + ' images are chosen for training')
    print(str(len(np.unique(TrnLabels))) + ' individuals for training')
    print(str(numImageTest) + ' images are chosen for testing')
    print(str(len(np.unique(TestLabels))) + ' individuals for testing')
    print('\n')

    # --------------------------------
    ############   Training  #############

    # --------------------------------
    print('\tTraining...')

    # --------------------------------
    # Load images for training
    print('\t\tLoading images for training...')
    imagesCellTrain, filenameTrain = loadImages(files, dirDB, allIndexes, indImagesTrain, numImageTrain)

    # --------------------------------
    # Adjusting format
    print('\t\tAdjusting format...')
    imagesCellTrain, imageSize = adjustFormat(imagesCellTrain)

    # --------------------------------
    # Search for orientations
    print('\t\tSearching for best orientations...')
    # default orientations by sampling
    orient_default = np.linspace(0, 180 - int((180 / param.divTheta)), param.divTheta, dtype= int)
    # adaptive orientations by gradient
    orient_best = searchGaborOrientation(imagesCellTrain, param)


    # --------------------------------
    # Gabor analysis
    print('\t\tGabor analysis...')
    start = time.time()
    bestWaveletsAll = findBestWaveletsTesting(imagesCellTrain, orient_default, orient_best, numImageTrain, imageSize, param)
    end = time.time()
    print('\t\t\tGabor analysis time : ' + str(((end - start) / 60)) + ' minutes')
    print('\t\t\tTotal number of Gabor Wavelets : ' + str(param.PalmNet_numFilters[0]))


    # ---------------------------------
    ############  Testing  #############

    # ---------------------------------
    print('\n\tTesting...')

    # ---------------------------------
    # Load images for testing
    print('\tLoading images for testing...')
    imagesCellTest, filenameTest = loadImages(files, dirDB, allIndexes, indImagesTest, numImageTest)

    # ---------------------------------
    # Adjusting format
    print('\tAdjusting format...')
    imagesCellTest, imageSize = adjustFormat(imagesCellTest)


    # ---------------------------------
    # Feature extraction
    print('\tFeature extraction...')
    fTest_all, numFeatures = featExtrGaborAdapt(imagesCellTest, param, bestWaveletsAll, numImageTest)
    sizeTest = fTest_all.shape[1]
# print(indexesFold)
