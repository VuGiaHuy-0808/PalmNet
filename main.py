import glob
import time
import numpy as np
import cv2
from loadImages import loadImages
from loadImages import im2double_matlab
from checkMinNumSamplePerInd import checkMinNumSamplePerInd
from computeLabels import computeLabels
from getNumSamplePerInd import getNumSamplePerInd
from computeIndexesPersonFold import computeIndexesPersonFold
from adjustFormat import adjustFormat
from searchGaborOrientation import searchGaborOrientation
from findBestWaveletsTesting import findBestWaveletsTesting
from featExtrGaborAdapt import featExtrGaborAdapt
from fastEuclideanDistance import fastEuclideanDistance
import paramsPalmNet as param
import math
from Gabor_FeaExt import Gabor_FeaExt
from scipy.io import savemat, loadmat
from scipy.sparse import save_npz, load_npz, csr_matrix

# --------------------------------
# General parameters
numCoresKnn = 2
stepPrint = 100
# model = loadmat('model.mat')
# --------------------------------
# Directory of DataBase
ext = 'bmp'
dbName = 'Tongji_Contactless_Palmprint_Dataset'
dirDB = 'C:\PalmNet_Matlab\PalmNet\images\Tongji_Contactless_Palmprint_Dataset'
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

# ----------------------------------
#   KNN Test
#
# Load image
# im = cv2.imread(dirDB + "\\" + "0001_0001.bmp")
# if im.shape[2] == 3:
#     im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     im2double = im2double_matlab(im)
#
# # adjust format
# closPow2 = int(math.pow(2, math.floor(math.log2(im2double.shape[0]))))
# imageSize = (closPow2, closPow2)
# # Back to rgb
# im2double = (im2double * 255).round().astype(np.uint8)
# # Cast
# im2double = im2double.astype(np.float64)
# # Resize image, must be power of 2
# im2double = cv2.resize(im2double, imageSize, interpolation=cv2.INTER_CUBIC)
# # Subtract mean
# im2double = im2double - cv2.mean(im2double)[0]
#
# # load model
# model = loadmat('model.mat')
# bestWaveletsAll = []
# for i in range(model['bestWaveletsAll'][0].shape[0]):
#     bestWaveletsAll.append(model['bestWaveletsAll'][0][i])
#
# # load feature map
# featureMap = load_npz('mapFeature.npz')
#
# # feature extraction
# fTest, _ = Gabor_FeaExt(im, param, bestWaveletsAll)
# fTest = csr_matrix(fTest).transpose()
#
# # compute distMatrix
# distMat = fastEuclideanDistance(featureMap, fTest).flatten()
# ind = np.argsort(distMat)
#

#__________________ SUCCESS _____________________________________
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
    indexesFold, allIndexes, indImagesTrain, indImagesTest, numImageTrain, numImageTest = computeIndexesPersonFold(
        numImageAll, labels)
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
    orient_default = np.linspace(0, 180 - int((180 / param.divTheta)), param.divTheta, dtype=int)
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
    savemat('model.mat', {'bestWaveletsAll': bestWaveletsAll})
    model = loadmat('model.mat')

    # ---------------------------------
    ############  Testing  #############

    # ---------------------------------
    print('\n\tTesting...')

    # ---------------------------------
    # Load images for testing
    print('\tLoading images for testing...')
    imagesCellTest, filenameTest = loadImages(files, dirDB, allIndexes, indImagesTest, numImageTest)
    with open('filenameTest.txt', 'w') as f:
        for item in filenameTest:
            f.write("%s\n" % item)

    # ---------------------------------
    # Adjusting format
    print('\tAdjusting format...')
    imagesCellTest, imageSize = adjustFormat(imagesCellTest)

    # ---------------------------------
    # Feature extraction
    print('\tFeature extraction...')
    fTest_all, numFeatures = featExtrGaborAdapt(imagesCellTest, param, bestWaveletsAll, numImageTest)
    fTest_all = fTest_all.tocsr(True)
    save_npz('mapFeature.npz', fTest_all)
    featureMap = load_npz('mapFeature.npz')
    sizeTest = fTest_all.shape[1]
# print(indexesFold)
