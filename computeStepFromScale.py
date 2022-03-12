
def computeStepFromScale(scale):

    if scale == 0:
        step = 2 ** scale
    else:
        step = (2 ** scale) * 3/2

    return step