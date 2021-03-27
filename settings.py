FEATURES_FROM_FILE = False
LOAD_SIGNAL = True
REMOVE_PAUSE = True
UBM_FROM_FILE = False

SR = 8000 # sample rate
N_MFCC = 13 # number of MFCC to extract
N_FFT = 0.020  # length of the FFT window in seconds
HOP_LENGTH = 0.010 # number of samples between successive frames in seconds
N_COMPONENTS = 16 # number of gaussians
COVARINACE_TYPE = 'full' # cov type for GMM           # try covariance_type='diag',


FILE_NUMBER = 1

abs_path = 'C:/Users/super/Desktop/UIR3'
data_name = abs_path + '/data/6319-64726-0005.wav'
ubm_file_name = abs_path + '/features/ubm/ubm_{0}_{1}_{2}mfcc.pkl'.format(N_COMPONENTS, COVARINACE_TYPE, N_MFCC)
feature_name = abs_path + '/features/{0}-mfcc{1}.pkl'.format(FILE_NUMBER, N_MFCC)