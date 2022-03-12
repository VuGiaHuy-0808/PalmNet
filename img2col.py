import numpy as np

def img2col(im, histBlockSize):

    heightStep = im.shape[0] // histBlockSize[0]
    widthStep = im.shape[1] // histBlockSize[1]
    rearrangeMatrix = []

    for cols in range(widthStep):
        for rows in range(heightStep):

            # separate img input to (heightStep * widthStep) blocks
            # size of each block is histBlockSize
            # rearrange it into 1D array
            tempImg = im[histBlockSize[0] * rows : (histBlockSize[0] * (rows + 1)),
                      histBlockSize[1] * cols : (histBlockSize[1] * (cols + 1))]

            rearrangeMatrix.append(tempImg.flatten('F'))

    return np.transpose(np.array(rearrangeMatrix))