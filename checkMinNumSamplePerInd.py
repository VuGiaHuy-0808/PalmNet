import numpy as np

def checkMinNumSamplePerInd (files):

    indexrem = []

    for i in range(len(files)):
        if i != (len(files) - 1):
            if files[i - 1][:4] != files[i][:4] and files[i + 1][:4] != files[i][:4]:
                indexrem.append(i)
        else:
            if files[i - 1][:4] != files[i][:4] and files[i][:4] != files[0][:4]:
                indexrem.append(i)

    for i in range(len(indexrem)):
        files.pop(indexrem[i])