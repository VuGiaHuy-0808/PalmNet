from computeStepFromScale import computeStepFromScale
from myConv2 import myConv2

class ResFilter:
    even = []
    odd = []

def referenceGaborFilter(gaborBank, im):

    # init
    resF = ResFilter()
    for g in range(len(gaborBank.even)):

        # Filtering arcording to paper
        step = computeStepFromScale(gaborBank.scale[g])
        resF.even.append(myConv2(gaborBank.even[g], im, step))
        resF.odd.append(myConv2(gaborBank.odd[g], im, step))

    return resF