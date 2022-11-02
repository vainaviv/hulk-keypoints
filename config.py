class ExperimentTypes:
    CLASSIFY_OVER_UNDER = 'cou'
    OPPOSITE_ENDPOINT_PREDICTION = 'oep'

ALLOWED_EXPT_TYPES = [ExperimentTypes.CLASSIFY_OVER_UNDER,
                      ExperimentTypes.OPPOSITE_ENDPOINT_PREDICTION]

NUM_KEYPOINTS = 1
IMG_HEIGHT  = 200
IMG_WIDTH   = 200
GAUSS_SIGMA = 5
epochs = 50 #500
batch_size = 4

dataset_dir ='/home/kaushiks/hulk-keypoints/processed_sim_data/under_over_crossings_dataset'
