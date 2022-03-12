import math
import numpy as np
import cv2
import scipy

def adjustFormat(imagesCellTrain):

    # Compute image size
    # Image size must be power of 2
    im = imagesCellTrain[0]
    closPow2 = int(math.pow(2, math.floor(math.log2(im.shape[0]))))
    imageSize = (closPow2, closPow2)
    
    for i in range(len(imagesCellTrain)):
        
        im = imagesCellTrain[i]
        # Back to rgb
        im = (im * 255).round().astype(np.uint8)
        # Cast
        im = im.astype(np.float64)
        # Resize image, must be power of 2    
        im = cv2.resize(im, imageSize, interpolation= cv2.INTER_CUBIC)
        # Subtract mean
        im = im - cv2.mean(im)[0]

        # Assign
        imagesCellTrain[i] = im

    return imagesCellTrain, imageSize