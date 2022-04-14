from scipy.sparse import csr_matrix
import numpy as np

def fastEuclideanDistance(a, b):

    aPow = a.power(2, dtype=float)
    bPow = b.power(2, dtype=float)
    aa = np.asarray(aPow.sum(0, dtype=float))
    bb = np.asarray(bPow.sum(0, dtype=float))
    aT = a.transpose()
    ab = (aT * b).toarray()
    repmatA = np.tile(aa.transpose(), (1, bb.shape[1]))
    repmatB = np.tile(bb, (aa.shape[1], 1))
    d = np.sqrt(np.abs(repmatA + repmatB - 2 * ab))

    return d