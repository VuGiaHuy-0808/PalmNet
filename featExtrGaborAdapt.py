from scipy.sparse import csr_matrix
from Gabor_FeaExt import Gabor_FeaExt

def featExtrGaborAdapt(imagesCellTest, param, bestWavelets, numImagesTest):

    # One iteration to get size and init
    fTest, _ = Gabor_FeaExt(imagesCellTest[0], param, bestWavelets)
    numFeatures = fTest.shape[0]
    # init features matrix
    fTest_all = csr_matrix(shape=(numFeatures, numImagesTest), dtype=float)
    fTest_all[:,0] = fTest

    # Extract features
    for j in range(1, numImagesTest):
        # get image
        im = imagesCellTest[j]

        # PCANet output
        ftest, _ = Gabor_FeaExt(im,param, bestWavelets)

        # save data
        fTest_all[:,j] = ftest

    return fTest_all, numFeatures