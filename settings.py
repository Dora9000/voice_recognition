#FEATURES_FROM_FILE = False
#LOAD_SIGNAL = True
#REMOVE_PAUSE = True
#UBM_FROM_FILE = False

SR = 8000  # sample rate
N_MFCC = 13  # number of MFCC to extract
N_FFT = 0.020  # length of the FFT window in seconds
HOP_LENGTH = 0.010  # number of samples between successive frames in seconds
N_COMPONENTS = 16  # number of gaussians
COVARINACE_TYPE = 'full'  # cov type for GMM

DURATION_TO_REMEMBER = 20
DURATION_TO_RECOGNIZE = 5