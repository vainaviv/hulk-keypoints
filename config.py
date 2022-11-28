class ExperimentTypes:
    CLASSIFY_OVER_UNDER = 'cou'
    OPPOSITE_ENDPOINT_PREDICTION = 'oep'
    TRACE_PREDICTION = 'trp'

# TODO Jainil: add Experiment type for cage pinch predictions

ALLOWED_EXPT_TYPES = [ExperimentTypes.CLASSIFY_OVER_UNDER,
                      ExperimentTypes.OPPOSITE_ENDPOINT_PREDICTION,
                      ExperimentTypes.TRACE_PREDICTION]

PARAMS_CONDITION_ON_POINTS = {'COND_POINT_DIST_PX': 25,
                                'CONDITION_LEN':  4,
                                'CROP_WIDTH': 100,
                                'PRED_LEN': 1}

PARAMS_CONDITION_ON_SWEEP = {'COND_POINT_DIST_PX': 8,
                                'CONDITION_LEN':  6,
                                'CROP_WIDTH': 50,
                                'PRED_LEN': 1}

NUM_KEYPOINTS = 1
IMG_HEIGHT  = lambda expt_type: 200 if is_crop_task(expt_type) else 100 
IMG_WIDTH   = lambda expt_type: 200 if is_crop_task(expt_type) else 100
GAUSS_SIGMA = 5
epochs = 50
batch_size = 4
COND_POINT_DIST_PX = 8
CONDITION_LEN = 6
CROP_WIDTH = 50
PRED_LEN = 1

def get_dataset_dir(expt_type):
    if expt_type == ExperimentTypes.TRACE_PREDICTION:
        return '/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_complex'
    return '/home/kaushiks/hulk-keypoints/processed_sim_data/under_over_crossings_dataset'

def is_crop_task(expt_type):
    return expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER or expt_type == ExperimentTypes.OPPOSITE_ENDPOINT_PREDICTION

# TODO Jainil: add cage pinch as a point pred type
def is_point_pred(expt_type):
    return expt_type == ExperimentTypes.OPPOSITE_ENDPOINT_PREDICTION or expt_type == ExperimentTypes.TRACE_PREDICTION