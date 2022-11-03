class ExperimentTypes:
    CLASSIFY_OVER_UNDER = 'cou'
    OPPOSITE_ENDPOINT_PREDICTION = 'oep'
    TRACE_PREDICTION = 'trp'

ALLOWED_EXPT_TYPES = [ExperimentTypes.CLASSIFY_OVER_UNDER,
                      ExperimentTypes.OPPOSITE_ENDPOINT_PREDICTION,
                      ExperimentTypes.TRACE_PREDICTION]

NUM_KEYPOINTS = 1
IMG_HEIGHT  = 200
IMG_WIDTH   = 200
GAUSS_SIGMA = 5
epochs = 50
batch_size = 4

def get_dataset_dir(expt_type):
    if expt_type == ExperimentTypes.TRACE_PREDICTION:
        return '/home/kaushiks/rope-rendering/processed_sim_data/trace_dataset/train/'
    return '/home/kaushiks/hulk-keypoints/processed_sim_data/under_over_crossings_dataset'