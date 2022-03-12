import numpy as np

def computePowerMaps(resF, gaborBank):

    powerMapF = []
    for g in range(len(gaborBank.even)):
        powerMapF.append(np.power(resF.even[g], 2) + np.power(resF.odd[g],2))

    return powerMapF