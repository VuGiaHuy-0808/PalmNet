
# -------------------------
# general parameters
numIterations = 5
kfold = 2 # k-1 partition used for training, 1 partition for testing
# KNN parameters
knnNeighbors = 1

# --------------------------
# Parameters orientation search
numBins = 45
divTheta = 10
maxOrient = 10

# ------------------------------
# Parameters Gabor from scales
N = 1 # Sampling points per octave
b0 = 1 # the unit spatial interval
phai = 1.5 # band width of gabor (octave)
aspectRatio = 1 # aspect ratio of gabor filter

# -------------------------------
# Parameters Gabor paremetrized
divThetaParametrized = divTheta
sigma = 5.6179 # used for generating Gabor filter
wavelength = 0.11 # used for generating Gabor filter

# -------------------------------
# Parameters wavelet add
numWavelets = 10000
minSE = 0.005
minCountW = 2000
numBestWavelets = 5

# -------------------------------
# PCANet parameters
PalmNet_numStages = 2
PalmNet_patchSize = [15, 15]
PalmNet_numFilters = [divThetaParametrized + numBestWavelets, divThetaParametrized + numBestWavelets]
PalmNet_HistBlockSize = [23, 23]
PalmNet_BlkOverLapRatio = 0
PalmNet_pyramid = []