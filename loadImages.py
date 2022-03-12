import numpy as np
import cv2
import glob

def im2double_matlab(im):
    info = np.iinfo(im.dtype) 
    return im.astype(np.float) / info.max 

def loadImages(files, dirDB, allIndexes, indImagesTrain, numImagesTrain):

    # Init
    vectorIndexTrain = []
    for i in range(len(indImagesTrain)):
        if indImagesTrain[i] == 1:
            vectorIndexTrain.append(allIndexes[i])
    filenameTrain = []
    imagesCellTrain = []

    # Loop
    for i in range(numImagesTrain):
        filenameTrain.append(files[int(vectorIndexTrain[i])])
        im = cv2.imread(dirDB + "\\" + filenameTrain[i])
        if im.shape[2] == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im2double = im2double_matlab(im)
        imagesCellTrain.append(im2double)

    return imagesCellTrain, filenameTrain