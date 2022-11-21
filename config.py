import json
import os

class ExperimentTypes:
    CLASSIFY_OVER_UNDER = 'cou'
    OPPOSITE_ENDPOINT_PREDICTION = 'oep'
    TRACE_PREDICTION = 'trp'
    CAGE_PREDICTION = 'cap'

# TODO Jainil: add Experiment type for cage pinch predictions
ALLOWED_EXPT_TYPES = [ExperimentTypes.CLASSIFY_OVER_UNDER,
                      ExperimentTypes.OPPOSITE_ENDPOINT_PREDICTION,
                      ExperimentTypes.TRACE_PREDICTION,
                      ExperimentTypes.CAGE_PREDICTION]

NUM_KEYPOINTS = 1
IMG_HEIGHT  = lambda expt_type: 200 if is_crop_task(expt_type) else 100 
IMG_WIDTH   = lambda expt_type: 200 if is_crop_task(expt_type) else 100
GAUSS_SIGMA = 2
epochs = 150
batch_size = 4
COND_POINT_DIST_PX = 20
CONDITION_LEN = 5
CROP_WIDTH = 120
PRED_LEN = 2
EVAL_CHECKPT_FREQ = 1
MIN_CHECKPOINT_FREQ = 10
MODEL_IMG_SIZE = 100
RESNET_TYPE = '50'
PRETRAINED = False

# TODO Jainil: add link to dataset
def get_dataset_dir(expt_type):
    if expt_type == ExperimentTypes.TRACE_PREDICTION:
        return '/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_complex'
    return '/home/kaushiks/hulk-keypoints/processed_sim_data/under_over_crossings_dataset'

def is_crop_task(expt_type):
    return expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER or expt_type == ExperimentTypes.OPPOSITE_ENDPOINT_PREDICTION

# TODO Jainil: add cage pinch as a point pred type
def is_point_pred(expt_type):
    return expt_type == ExperimentTypes.OPPOSITE_ENDPOINT_PREDICTION or expt_type == ExperimentTypes.TRACE_PREDICTION or expt_type == ExperimentTypes.CAGE_PREDICTION

def save_config_params(path, expt_type):
    params_dict = {
        'expt_type': expt_type,
        'num_keypoints': NUM_KEYPOINTS,
        'img_height': IMG_HEIGHT(expt_type),
        'img_width': IMG_WIDTH(expt_type),
        'gauss_sigma': GAUSS_SIGMA,
        'epochs': epochs,
        'batch_size': batch_size,
        'cond_point_dist_px': COND_POINT_DIST_PX,
        'condition_len': CONDITION_LEN,
        'crop_width': CROP_WIDTH,
        'pred_len': PRED_LEN
    }

    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump(params_dict, f)
        f.close()