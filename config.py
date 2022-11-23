import json
import os
import dataclasses
from dataclasses import dataclass
# from dataclasses_json import dataclass_json

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

def save_config_params(path, expt_config):
    with open(os.path.join(path, 'config.json'), 'w') as f:
        dct = {}
        for k in dir(expt_config):
            if k[0] != '_':
                dct[k] = getattr(expt_config, k)
        json.dump(dct, f, indent=4)
        f.close()
    with open(os.path.join(path, 'expt_class.txt'), 'w') as f:
        f.write(str(expt_config.__class__.__name__))
        f.close()

def load_config_class(path):
    with open(os.path.join(path, 'config.json'), 'r') as f:
        dct = json.load(f)
        f.close()
    return BaseTraceExperimentConfig(**dct)

@dataclass
class BaseTraceExperimentConfig:
    expt_type = ExperimentTypes.TRACE_PREDICTION
    img_height = 80
    img_width = 80
    crop_width = 80
    num_keypoints = 1
    gauss_sigma = 2
    epochs = 150
    batch_size = 4
    cond_point_dist_px = 20
    condition_len = 5
    pred_len = 1
    eval_checkpoint_freq = 1
    min_checkpoint_freq = 10
    resnet_type = '50'
    pretrained = False

@dataclass
class TRCR80(BaseTraceExperimentConfig):
    crop_width = 80

@dataclass
class TRCR100(BaseTraceExperimentConfig):
    crop_width = 100

@dataclass
class TRCR120(BaseTraceExperimentConfig):
    crop_width = 120

@dataclass
class CAP800(BaseTraceExperimentConfig):
    expt_type = ExperimentTypes.CAGE_PREDICTION
    img_height = 800
    img_width = 800
    gauss_sigma = 4
    condition_len = 4

def get_class_name(cls):
    return cls.__name__

ALL_EXPERIMENTS_LIST = [BaseTraceExperimentConfig, TRCR80, TRCR100, TRCR120, CAP800]
ALL_EXPERIMENTS_CONFIG = {get_class_name(expt): expt for expt in ALL_EXPERIMENTS_LIST}
