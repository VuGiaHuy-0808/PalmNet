import math
import numpy as np
from Gabor_output import Gabor_output
from HashingHist import HashingHist

def Gabor_FeaExt(inImg, param, bestWavelets):

    # Output:
    # f:         PCANet features (each column corresponds to feature of each image)
    # BlkInd:    index of local block from which the histogram is computed

    numImg = 1
    outImg = inImg
    imgIdx = np.transpose(np.linspace(0, numImg - 1, numImg))

    for stage in range(param.PalmNet_numStages):
        if stage == 0:
            # gabor output
            outImg = Gabor_output(outImg, bestWavelets, param.PalmNet_numFilters[stage])
            outImgIdx = np.kron(imgIdx, np.ones((param.PalmNet_numFilters[stage],1)))
        else:
            # gabor output
            outImg = Gabor_output(outImg, bestWavelets, param.PalmNet_numFilters[stage])
            outImgIdx = np.kron(outImgIdx, np.ones((param.PalmNet_numFilters[stage],1)))

    f, blkIdx = HashingHist(param, outImgIdx, outImg)

    return f, blkIdx