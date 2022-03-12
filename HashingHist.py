import numpy as np
from img2col import img2col
from scipy.sparse import csr_matrix

def time_along_column(feaMap, mulMatix):
    for cols in range(feaMap.shape[1]):
        feaMap[:,cols] = feaMap[:,cols] * mulMatix[0,cols]
    return feaMap

def histc(X, bins):
    map_to_bins = np.digitize(X,bins)
    binMatrix = np.zeros((len(bins), X.shape[1]), dtype= int)
    for cols in range(binMatrix.shape[1]):
        temp = map_to_bins[:,cols]
        for i in temp:
            binMatrix[i-1,cols] += 1
    return binMatrix

def HashingHist(param, imgIdx, outImg):

    # output layer of PCANet (Hashing plus local histogram)
    # ------------ Output ----------------
    # f         PCANet features (each column corresponds to feature of each image)
    # BlkIdx    index of local block from which the histogram is computed

    numImg = 1
    f = []
    BHist = []
    # weights for binary to decimal conversion
    map_weights = np.power(2, np.linspace(param.PalmNet_numFilters[1] - 1, 0, param.PalmNet_numFilters[1]))

    for Idx in range(numImg):
        Idx_span = np.where(imgIdx == Idx)
        NumOs = len(Idx_span[0]) / param.PalmNet_numFilters[1] # the number of '0's

        for i in range(int(NumOs)):

            T = 0
            for j in range(param.PalmNet_numFilters[1]):
                # weighted combination, hashing codes to decimal number conversion
                T += map_weights[j] * np.heaviside(outImg[Idx_span[0][param.PalmNet_numFilters[1] * i + j]], 0)

            # img2col    : hashing features of block image into column
            # histc      : count the occurrences of value in img2col and save them by indexes
            # csr_matrix : reduce memory
            rangeCount = np.linspace(0, np.power(2, param.PalmNet_numFilters[1]) - 1, np.power(2, param.PalmNet_numFilters[1]))
            blkwise_fea = csr_matrix(histc(img2col(T, param.PalmNet_HistBlockSize), rangeCount), dtype=float)
            Halinh = np.power(2, param.PalmNet_numFilters[1]) / np.sum(blkwise_fea, axis=0)
            blkwise_fea = time_along_column(blkwise_fea, Halinh)
            BHist.append(blkwise_fea.toarray().flatten())

        # save features into f
        f = np.concatenate(BHist)
    
    # init index
    idxRange = (np.linspace(0, Halinh.shape[1] - 1 , Halinh.shape[1], dtype=int)).reshape(-1,1)
    blkIdx = np.kron(np.ones(shape=(int(NumOs), 1)), np.kron(idxRange, np.ones(shape=(blkwise_fea.shape[0],1))))

    return f, blkIdx