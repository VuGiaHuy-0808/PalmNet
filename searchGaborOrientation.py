import numpy as np
from ridgeorient import ridgeorient

def searchGaborOrientation(imagesCellTrain, param):

    # Output :
    # Orient_chosen : vector of angles (in degrees)

    # Init
    histAll = np.zeros(shape=(1,param.numBins))
    edges = []

    for im_index in range(len(imagesCellTrain)):

        img = imagesCellTrain[im_index]

        # Orientation map
        orientim = ridgeorient(img, 0.1, 1.5, 1.5)

        # Convert to deg
        orientim_deg = np.rad2deg(orientim)

        N, edges = np.histogram(orientim_deg, bins= np.arange(0, 184, 4))

        # Sum for all images
        histAll += N

    # Average hist
    histAll = histAll / len(imagesCellTrain)

    # Compute centers from edges
    centers = (edges + np.roll(edges, -1)) / 2
    centers = np.delete(centers, -1)

    # Remove last element (0 = 180)
    histAll = np.delete(histAll, -1)
    centers = np.delete(centers, -1)

    # Convert to anti-cw
    centers = 180 - centers

    # Sorting descend
    histAll_sort_ind = np.argsort(histAll)[::-1]
    histAll_sort = histAll[histAll_sort_ind]

    # Get orient sorted
    orient_sort = centers[histAll_sort_ind]

    # Choose orient
    orient_chosen = orient_sort[:param.maxOrient]

    return orient_chosen