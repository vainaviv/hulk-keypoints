class ExperimentTypes:
    CLASSIFY_OVER_UNDER = 'classify'
    OPPOSITE_ENDPOINT_PREDICTION = 'opposite_endpoint_prediction'

ALLOWED_EXPT_TYPES = [ExperimentTypes.CLASSIFY_OVER_UNDER,
                      ExperimentTypes.OPPOSITE_ENDPOINT_PREDICTION]

NUM_KEYPOINTS = 1
IMG_HEIGHT  = 200
IMG_WIDTH   = 200
GAUSS_SIGMA = 5
epochs = 50 #500
batch_size = 4
