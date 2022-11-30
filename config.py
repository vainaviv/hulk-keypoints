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
    expt_type: str = ExperimentTypes.TRACE_PREDICTION
    dataset_dir: str = get_dataset_dir(ExperimentTypes.TRACE_PREDICTION)
    img_height: int = 100
    img_width: int = 100
    crop_width: int = 80
    num_keypoints: int = 1
    gauss_sigma: int = 2
    epochs: int = 150
    batch_size: int = 4
    cond_point_dist_px: int = 20
    condition_len: int = 5
    pred_len: int = 1
    eval_checkpoint_freq: int = 1
    min_checkpoint_freq: int = 10
    resnet_type: str = '50'
    pretrained: bool = False

@dataclass
class TRCR80(BaseTraceExperimentConfig):
    crop_width: int = 80

@dataclass
class TRCR100(BaseTraceExperimentConfig):
    crop_width: int = 100

@dataclass
class TRCR120(BaseTraceExperimentConfig):
    crop_width: int = 120

@dataclass
class CL5_20_PL1(BaseTraceExperimentConfig):
    crop_width: int = 100
    cond_point_dist_px: int = 20
    condition_len: int = 5
    pred_len: int = 1

@dataclass
class TRCR140_CL4_25_PL1(BaseTraceExperimentConfig):
    crop_width: int = 140
    cond_point_dist_px: int = 25
    condition_len: int = 4
    pred_len: int = 1

@dataclass
class CL3_10_PL2(BaseTraceExperimentConfig):
    crop_width: int = 100
    cond_point_dist_px: int = 10
    condition_len: int = 3
    pred_len: int = 2
    epochs: int = 75

@dataclass
class CL10_10_PL2(BaseTraceExperimentConfig):
    crop_width: int = 100
    cond_point_dist_px: int = 10
    condition_len: int = 10
    pred_len: int = 2
    epochs: int = 75

@dataclass
class CL3_10_PL1(BaseTraceExperimentConfig):
    crop_width: int = 100
    cond_point_dist_px: int = 10
    condition_len: int = 3
    pred_len: int = 1
    epochs: int = 75

@dataclass
class CL10_10_PL1(BaseTraceExperimentConfig):
    crop_width: int = 100
    cond_point_dist_px: int = 10
    condition_len: int = 10
    pred_len: int = 1
    epochs: int = 75


@dataclass
class CAP800(BaseTraceExperimentConfig):
    expt_type: str = ExperimentTypes.CAGE_PREDICTION
    img_height: int = 800
    img_width: int = 800
    gauss_sigma: int = 4
    condition_len: int = 4

@dataclass
class TRCR140_CL4_25_PL1_RN34(BaseTraceExperimentConfig):
    crop_width: int = 140
    cond_point_dist_px: int = 25
    condition_len: int = 4
    pred_len: int = 1
    resnet_type: str = '34'
    dataset_dir: str = '/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset'


@dataclass
class TRCR140_CL4_25_PL1_RN34_MED(BaseTraceExperimentConfig):
    crop_width: int = 140
    cond_point_dist_px: int = 25
    condition_len: int = 4
    pred_len: int = 1
    resnet_type: str = '34'
    dataset_dir: str = '/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_medium_2'

@dataclass
class TRCR80_CL4_25_PL1_RN34_MED(BaseTraceExperimentConfig):
    crop_width: int = 80
    cond_point_dist_px: int = 34
    condition_len: int = 4
    pred_len: int = 1
    resnet_type: str = '34'
    dataset_dir: str = '/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_medium_2'


def get_class_name(cls):
    return cls.__name__

ALL_EXPERIMENTS_LIST = [BaseTraceExperimentConfig, TRCR80, TRCR100, TRCR120, CL5_20_PL1, CL3_10_PL2, CL10_10_PL2, CL3_10_PL1, CL10_10_PL1, CAP800, TRCR140_CL4_25_PL1, TRCR140_CL4_25_PL1_RN34, TRCR140_CL4_25_PL1_RN34_MED, TRCR80_CL4_25_PL1_RN34_MED]
ALL_EXPERIMENTS_CONFIG = {get_class_name(expt): expt for expt in ALL_EXPERIMENTS_LIST}
