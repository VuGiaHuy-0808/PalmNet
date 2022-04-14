from scipy.sparse import lil_matrix
from Gabor_FeaExt import Gabor_FeaExt

def featExtrGaborAdapt(imagesCellTest, param, bestWavelets, numImagesTest):

    # One iteration to get size and init
    fTest, _ = Gabor_FeaExt(imagesCellTest[0], param, bestWavelets)
    numFeatures = fTest.shape[0]
    # init features matrix
    fTest_all = lil_matrix((numFeatures, numImagesTest), dtype=float)
    fTest_all[:,0] = fTest
    count = 0
    # Extract features
    for j in range(numImagesTest):
        # get image
        im = imagesCellTest[j]

        # PCANet output
        ftest, _ = Gabor_FeaExt(im,param, bestWavelets)
        print(count)
        count += 1
        # save data
        fTest_all[:,j] = ftest

    return fTest_all, numFeatures