import json
import os
import dataclasses
from dataclasses import dataclass, field
from typing import List
# from dataclasses_json import dataclass_json

class ExperimentTypes:
    CLASSIFY_OVER_UNDER = 'cou'
    OPPOSITE_ENDPOINT_PREDICTION = 'oep'
    TRACE_PREDICTION = 'trp'
    CAGE_PREDICTION = 'cap'
    CLASSIFY_OVER_UNDER_NONE = 'coun'

ALLOWED_EXPT_TYPES = [ExperimentTypes.CLASSIFY_OVER_UNDER,
                      ExperimentTypes.OPPOSITE_ENDPOINT_PREDICTION,
                      ExperimentTypes.TRACE_PREDICTION,
                      ExperimentTypes.CAGE_PREDICTION,
                      ExperimentTypes.CLASSIFY_OVER_UNDER_NONE]

def get_dataset_dir(expt_type):
    if expt_type == ExperimentTypes.TRACE_PREDICTION:
        return '/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_complex'
    elif expt_type == ExperimentTypes.CAGE_PREDICTION:
        return '/home/mkparu/rope-rendering/data_processing/post_processed_sim_data/crop_cage_pinch_dataset'
    elif expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER:
        return '/home/vainavi/hulk-keypoints/processed_sim_data/under_over_crossing_set2'
    elif expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER_NONE:
        return '/home/vainavi/hulk-keypoints/processed_sim_data/under_over_none2'

def is_crop_task(expt_type):
    return expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER or expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER_NONE or expt_type == ExperimentTypes.OPPOSITE_ENDPOINT_PREDICTION

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
    return BaseConfig(**dct)

@dataclass
class BaseConfig:
    expt_type: str = ExperimentTypes.TRACE_PREDICTION
    dataset_dir: List[str] = field(default_factory=lambda: [get_dataset_dir(ExperimentTypes.TRACE_PREDICTION)])
    dataset_weights: List[float] = field(default_factory=lambda: [1.0])
    dataset_real: List[bool] = field(default_factory=lambda: [False])
    img_height: int = 100
    img_width: int = 100
    crop_width: int = 80
    num_keypoints: int = 1
    gauss_sigma: int = 2
    classes: int = 1
    epochs: int = 150
    batch_size: int = 4
    cond_point_dist_px: int = 20
    condition_len: int = 5
    pred_len: int = 1
    eval_checkpoint_freq: int = 1
    min_checkpoint_freq: int = 10
    resnet_type: str = '50'
    pretrained: bool = False
    oversample: bool = False
    oversample_rate: float = 0.8
    oversample_factor: float = 1.0
    rot_cond: bool = False
    sharpen: bool = False

@dataclass
class TRCR80(BaseConfig):
    crop_width: int = 80

@dataclass
class TRCR100(BaseConfig):
    crop_width: int = 100

@dataclass
class TRCR120(BaseConfig):
    crop_width: int = 120

@dataclass
class CL5_20_PL1(BaseConfig):
    crop_width: int = 100
    cond_point_dist_px: int = 20
    condition_len: int = 5
    pred_len: int = 1

@dataclass
class TRCR140_CL4_25_PL1(BaseConfig):
    crop_width: int = 140
    cond_point_dist_px: int = 25
    condition_len: int = 4
    pred_len: int = 1

@dataclass
class CL3_10_PL2(BaseConfig):
    crop_width: int = 100
    cond_point_dist_px: int = 10
    condition_len: int = 3
    pred_len: int = 2
    epochs: int = 75

@dataclass
class CL10_10_PL2(BaseConfig):
    crop_width: int = 100
    cond_point_dist_px: int = 10
    condition_len: int = 10
    pred_len: int = 2
    epochs: int = 75

@dataclass
class CL3_10_PL1(BaseConfig):
    crop_width: int = 100
    cond_point_dist_px: int = 10
    condition_len: int = 3
    pred_len: int = 1
    epochs: int = 75

@dataclass
class CL10_10_PL1(BaseConfig):
    crop_width: int = 100
    cond_point_dist_px: int = 10
    condition_len: int = 10
    pred_len: int = 1
    epochs: int = 75


@dataclass
class CAP600(BaseConfig):
    expt_type: str = ExperimentTypes.CAGE_PREDICTION
    img_height: int = 200
    img_width: int = 200
    gauss_sigma: int = 4
    condition_len: int = 4
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/mkparu/rope-rendering/data_processing/post_processed_sim_data/crop_cage_pinch_dataset'])

@dataclass
class TRCR140_CL4_25_PL1_RN34(BaseConfig):
    crop_width: int = 140
    cond_point_dist_px: int = 25
    condition_len: int = 4
    pred_len: int = 1
    resnet_type: str = '34'
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset'])


@dataclass
class TRCR140_CL4_25_PL1_RN34_MED(BaseConfig):
    crop_width: int = 140
    cond_point_dist_px: int = 25
    condition_len: int = 4
    pred_len: int = 1
    resnet_type: str = '34'
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_medium_2'])

@dataclass
class TRCR80_CL4_25_PL1_RN34_MED(BaseConfig):
    crop_width: int = 80
    cond_point_dist_px: int = 34
    condition_len: int = 4
    pred_len: int = 1
    resnet_type: str = '34'
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_medium_2'])

@dataclass
class TRCR80_CL4_25_PL1_RN34_MED3(BaseConfig):
    crop_width: int = 80
    cond_point_dist_px: int = 34
    condition_len: int = 4
    pred_len: int = 1
    resnet_type: str = '34'
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_medium_3'])

@dataclass
class TRCR80_CL4_25_PL1_RN50_MED3(BaseConfig):
    crop_width: int = 80
    cond_point_dist_px: int = 34
    condition_len: int = 4
    pred_len: int = 1
    resnet_type: str = '50'
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_medium_3'])

@dataclass
class TRCR60_CL4_25_PL1_RN34_MED3_V2(BaseConfig):
    crop_width: int = 60
    cond_point_dist_px: int = 20
    condition_len: int = 3
    pred_len: int = 1
    resnet_type: str = '34'
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_medium_3'])

@dataclass
class TRCR60_CL4_25_PL1_RN50_MED3_V2(BaseConfig):
    crop_width: int = 60
    cond_point_dist_px: int = 20
    condition_len: int = 3
    pred_len: int = 1
    resnet_type: str = '50'
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_medium_3'])

@dataclass
class TRCR60_CL4_25_PL1_RN50_MED3_B32_V2(BaseConfig):
    crop_width: int = 60
    cond_point_dist_px: int = 20
    condition_len: int = 3
    pred_len: int = 1
    resnet_type: str = '50'
    batch_size: int = 32
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_medium_3'])

@dataclass
class TRCR60_CL4_25_PL1_RN34_MED3_B32_V2(BaseConfig):
    crop_width: int = 60
    cond_point_dist_px: int = 20
    condition_len: int = 3
    pred_len: int = 1
    resnet_type: str = '34'
    batch_size: int = 32
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_medium_3'])


@dataclass
class TRCR60_CL3_20_PL1_RN34_MED3_RN34_B64_OS(BaseConfig):
    crop_width: int = 60
    cond_point_dist_px: int = 20
    condition_len: int = 3
    pred_len: int = 1
    resnet_type: str = '34'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_1'])
    oversample: bool = True


@dataclass
class TRCR60_CL3_20_PL1_MED3_RN50_B64_OS(BaseConfig):
    crop_width: int = 60
    cond_point_dist_px: int = 20
    condition_len: int = 3
    pred_len: int = 1
    resnet_type: str = '50'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_1'])
    oversample: bool = True

@dataclass
class TRCR60_CL3_20_PL1_MED3_UNet34_B64_OS(BaseConfig):
    crop_width: int = 60
    cond_point_dist_px: int = 20
    condition_len: int = 3
    pred_len: int = 1
    img_height: int = 96
    img_width: int = 96
    resnet_type: str = 'UNet34'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_1'])
    oversample: bool = True

@dataclass
class TRCR60_CL3_20_PL1_MED3_UNet34_B64_OS_RotCond(BaseConfig):
    crop_width: int = 60
    cond_point_dist_px: int = 20
    condition_len: int = 3
    pred_len: int = 1
    img_height: int = 96
    img_width: int = 96
    resnet_type: str = 'UNet34'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_1'])
    oversample: bool = True
    rot_cond: bool = True
    epochs: int = 250

@dataclass
class TRCR50_CL3_15_PL1_MED3_UNet34_B64_OS_RotCond(BaseConfig):
    crop_width: int = 50
    cond_point_dist_px: int = 15
    condition_len: int = 3
    pred_len: int = 1
    img_height: int = 96
    img_width: int = 96
    resnet_type: str = 'UNet34'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_1'])
    oversample: bool = True
    rot_cond: bool = True
    epochs: int = 250

@dataclass
class TRCR50_CL3_15_PL1_MED3_UNet18_B64_OS_RotCond(BaseConfig):
    crop_width: int = 50
    cond_point_dist_px: int = 15
    condition_len: int = 3
    pred_len: int = 1
    img_height: int = 96
    img_width: int = 96
    resnet_type: str = 'UNet18'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_1'])
    oversample: bool = True
    rot_cond: bool = True
    epochs: int = 250

@dataclass
class TRCR50_CL3_15_PL1_MED3_UNet50_B64_OS_RotCond(BaseConfig):
    crop_width: int = 50
    cond_point_dist_px: int = 15
    condition_len: int = 3
    pred_len: int = 1
    img_height: int = 96
    img_width: int = 96
    resnet_type: str = 'UNet50'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_1'])
    oversample: bool = True
    rot_cond: bool = True
    epochs: int = 250

@dataclass
class TRCR50_CL3_15_PL1_MED3_UNet101_B64_OS_RotCond(BaseConfig):
    crop_width: int = 50
    cond_point_dist_px: int = 15
    condition_len: int = 3
    pred_len: int = 1
    img_height: int = 96
    img_width: int = 96
    resnet_type: str = 'UNet101'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_1'])
    oversample: bool = True
    rot_cond: bool = True
    epochs: int = 250

@dataclass
class TRCR40_CL3_15_PL1_MED3_UNet34_B64_OS_RotCond(BaseConfig):
    crop_width: int = 40
    cond_point_dist_px: int = 15
    condition_len: int = 3
    pred_len: int = 1
    img_height: int = 96
    img_width: int = 96
    resnet_type: str = 'UNet34'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_1'])
    oversample: bool = True
    rot_cond: bool = True
    epochs: int = 250

@dataclass
class TRCR36_CL3_14_PL1_MED3_UNet34_B64_OS_RotCond(BaseConfig):
    crop_width: int = 36
    cond_point_dist_px: int = 14
    condition_len: int = 3
    pred_len: int = 1
    img_height: int = 96
    img_width: int = 96
    resnet_type: str = 'UNet34'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_1'])
    oversample: bool = True
    rot_cond: bool = True
    epochs: int = 250

@dataclass
class TRCR32_CL3_12_PL1_MED3_UNet34_B64_OS_RotCond(BaseConfig):
    crop_width: int = 32
    cond_point_dist_px: int = 12
    condition_len: int = 3
    pred_len: int = 1
    img_height: int = 96
    img_width: int = 96
    resnet_type: str = 'UNet34'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_1'])
    oversample: bool = True
    rot_cond: bool = True
    epochs: int = 250

@dataclass
class TRCR40_CL3_15_PL1_MED3_UNet34_B64_OS_RotCond_Hard2(BaseConfig):
    crop_width: int = 40
    cond_point_dist_px: int = 15
    condition_len: int = 3
    pred_len: int = 1
    img_height: int = 96
    img_width: int = 96
    resnet_type: str = 'UNet34'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_2'])
    oversample: bool = True
    rot_cond: bool = True
    epochs: int = 125

@dataclass
class TRCR36_CL3_14_PL1_MED3_UNet34_B64_OS_RotCond_Hard2(BaseConfig):
    crop_width: int = 36
    cond_point_dist_px: int = 14
    condition_len: int = 3
    pred_len: int = 1
    img_height: int = 96
    img_width: int = 96
    resnet_type: str = 'UNet34'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_2'])
    oversample: bool = True
    rot_cond: bool = True
    epochs: int = 125

@dataclass
class TRCR32_CL3_12_PL1_MED3_UNet34_B64_OS_RotCond_Hard2(BaseConfig):
    crop_width: int = 32
    cond_point_dist_px: int = 12
    condition_len: int = 3
    pred_len: int = 1
    img_height: int = 96
    img_width: int = 96
    resnet_type: str = 'UNet34'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_2'])
    oversample: bool = True
    rot_cond: bool = True
    epochs: int = 125

@dataclass
class TRCR28_CL3_12_PL1_MED3_UNet34_B64_OS_RotCond_Hard2(BaseConfig):
    crop_width: int = 28
    cond_point_dist_px: int = 12
    condition_len: int = 3
    pred_len: int = 1
    img_height: int = 96
    img_width: int = 96
    resnet_type: str = 'UNet34'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_2'])
    oversample: bool = True
    rot_cond: bool = True
    epochs: int = 125

@dataclass
class TRCR24_CL3_12_PL1_MED3_UNet34_B64_OS_RotCond_Hard2(BaseConfig):
    crop_width: int = 24
    cond_point_dist_px: int = 12
    condition_len: int = 3
    pred_len: int = 1
    img_height: int = 96
    img_width: int = 96
    resnet_type: str = 'UNet34'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_2'])
    oversample: bool = True
    rot_cond: bool = True
    epochs: int = 125

@dataclass
class TRCR40_CL3_15_PL1_MED3_UNet34_B64_OS_RotCond_Adj1(BaseConfig):
    crop_width: int = 40
    cond_point_dist_px: int = 15
    condition_len: int = 3
    pred_len: int = 1
    img_height: int = 96
    img_width: int = 96
    resnet_type: str = 'UNet34'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_adjacent_1'])
    oversample: bool = True
    rot_cond: bool = True
    epochs: int = 125

@dataclass
class TRCR36_CL3_14_PL1_MED3_UNet34_B64_OS_RotCond_Adj1(BaseConfig):
    crop_width: int = 36
    cond_point_dist_px: int = 14
    condition_len: int = 3
    pred_len: int = 1
    img_height: int = 96
    img_width: int = 96
    resnet_type: str = 'UNet34'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_adjacent_1'])
    oversample: bool = True
    rot_cond: bool = True
    epochs: int = 125

@dataclass
class TRCR32_CL3_12_PL1_MED3_UNet34_B64_OS_RotCond_Adj1(BaseConfig):
    crop_width: int = 32
    cond_point_dist_px: int = 12
    condition_len: int = 3
    pred_len: int = 1
    img_height: int = 96
    img_width: int = 96
    resnet_type: str = 'UNet34'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_adjacent_1'])
    oversample: bool = True
    rot_cond: bool = True
    epochs: int = 125

@dataclass
class TRCR32_CL3_12_PL1_MED3_UNet34_B64_OS_RotCond_Hard2_WReal(BaseConfig):
    crop_width: int = 32
    cond_point_dist_px: int = 12
    condition_len: int = 3
    pred_len: int = 1
    img_height: int = 96
    img_width: int = 96
    resnet_type: str = 'UNet34'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_2', '/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_real_1/real_data_for_tracer'])
    dataset_weights: List[float] = field(default_factory=lambda: [0.8, 0.2])
    dataset_real: List[bool] = field(default_factory=lambda: [False, True])
    # real_dataset_dir: List[str] = []
    oversample: bool = True
    rot_cond: bool = True
    epochs: int = 125

@dataclass
class TRC_HW128(BaseConfig):
    crop_width: int = 32
    cond_point_dist_px: int = 12
    condition_len: int = 3
    pred_len: int = 1
    img_height: int = 128
    img_width: int = 128
    oversample_rate: float = 0.8
    resnet_type: str = 'UNet34'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_2', '/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_real_1/real_data_for_tracer']    )
    dataset_weights: List[float] = field(default_factory=lambda: [0.8, 0.2])
    dataset_real: List[bool] = field(default_factory=lambda: [False, True])
    oversample: bool = True
    rot_cond: bool = True
    epochs: int = 125
    real_sample_rate: float = 0.2

@dataclass
class TRCR32_CL3_12_PL1_MED3_UNet34_B64_OS_RotCond_Hard2_Medley(BaseConfig):
    crop_width: int = 32
    cond_point_dist_px: int = 12
    condition_len: int = 3
    pred_len: int = 1
    img_height: int = 96
    img_width: int = 96
    resnet_type: str = 'UNet34'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_2', '/home/kaushiks/hulk-keypoints/processed_sim_data/annotations_hard_knots_3', '/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_adjacent_1', '/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_real_1/real_data_for_tracer'])
    dataset_weights: List[float] = field(default_factory=lambda: [0.3, 0.2, 0.35, 0.15])
    dataset_real: List[bool] = field(default_factory=lambda: [False, False, False, True])
    oversample: bool = True
    oversample_rate: float = 0.95
    rot_cond: bool = True
    epochs: int = 125

@dataclass
class TRCR32_CL3_12_PL1_MED3_UNet34_B64_OS_RotCond_Hard2_Medley_Sharp(BaseConfig):
    crop_width: int = 32
    cond_point_dist_px: int = 12
    condition_len: int = 3
    pred_len: int = 1
    img_height: int = 96
    img_width: int = 96
    resnet_type: str = 'UNet34'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_2', '/home/kaushiks/hulk-keypoints/processed_sim_data/annotations_hard_knots_3', '/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_adjacent_1', '/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_real_1/real_data_for_tracer'])
    dataset_weights: List[float] = field(default_factory=lambda: [0.3, 0.2, 0.35, 0.15])
    dataset_real: List[bool] = field(default_factory=lambda: [False, False, False, True])
    oversample: bool = True
    oversample_rate: float = 0.95
    rot_cond: bool = True
    epochs: int = 125
    sharpen: bool = True

@dataclass
class TRCR32_CL3_12_PL1_MED3_UNet34_B64_OS_RotCond_Hard2_Medley_MoreReal_Sharp(BaseConfig):
    crop_width: int = 32
    cond_point_dist_px: int = 12
    condition_len: int = 3
    pred_len: int = 1
    img_height: int = 96
    img_width: int = 96
    resnet_type: str = 'UNet34'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_2', '/home/kaushiks/hulk-keypoints/processed_sim_data/annotations_hard_knots_3', '/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_adjacent_1', '/home/vainavi/hulk-keypoints/real_data/real_data_for_tracer'])
    dataset_weights: List[float] = field(default_factory=lambda: [0.15, 0.15, 0.35, 0.35])
    dataset_real: List[bool] = field(default_factory=lambda: [False, False, False, True])
    oversample: bool = True
    oversample_rate: float = 0.95
    rot_cond: bool = True
    epochs: int = 125
    sharpen: bool = True

@dataclass
class TRCR32_CL3_12_PL1_MED3_UNet50_B64_OS_RotCond_Hard2_Medley_MoreReal_Sharp(BaseConfig):
    crop_width: int = 32
    cond_point_dist_px: int = 12
    condition_len: int = 3
    pred_len: int = 1
    img_height: int = 96
    img_width: int = 96
    resnet_type: str = 'UNet50'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_2', '/home/kaushiks/hulk-keypoints/processed_sim_data/annotations_hard_knots_3', '/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_adjacent_1', '/home/vainavi/hulk-keypoints/real_data/real_data_for_tracer'])
    dataset_weights: List[float] = field(default_factory=lambda: [0.15, 0.15, 0.35, 0.35])
    dataset_real: List[bool] = field(default_factory=lambda: [False, False, False, True])
    oversample: bool = True
    oversample_rate: float = 0.95
    rot_cond: bool = True
    epochs: int = 125
    sharpen: bool = True

@dataclass
class TRCR32_CL3_12_PL1_MED3_UNet34_B64_OS_RotCond_RealOnly_Sharp(BaseConfig):
    crop_width: int = 32
    cond_point_dist_px: int = 12
    condition_len: int = 3
    pred_len: int = 1
    img_height: int = 96
    img_width: int = 96
    resnet_type: str = 'UNet34'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/vainavi/hulk-keypoints/real_data/real_data_for_tracer'])
    dataset_weights: List[float] = field(default_factory=lambda: [1.0])
    dataset_real: List[bool] = field(default_factory=lambda: [True])
    oversample: bool = True
    oversample_rate: float = 0.95
    rot_cond: bool = True
    epochs: int = 1000
    sharpen: bool = True

@dataclass
class TRCR32_CL3_12_PL1_MED3_UNet34_B64_OS_RotCond_Hard2_Medley_Sharp_HW64(BaseConfig):
    crop_width: int = 32
    cond_point_dist_px: int = 12
    condition_len: int = 3
    pred_len: int = 1
    img_height: int = 64
    img_width: int = 64
    resnet_type: str = 'UNet34'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_2', '/home/kaushiks/hulk-keypoints/processed_sim_data/annotations_hard_knots_3', '/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_adjacent_1', '/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_real_1/real_data_for_tracer'])
    dataset_weights: List[float] = field(default_factory=lambda: [0.3, 0.2, 0.35, 0.15])
    dataset_real: List[bool] = field(default_factory=lambda: [False, False, False, True])
    oversample: bool = True
    oversample_rate: float = 0.95
    rot_cond: bool = True
    epochs: int = 125
    sharpen: bool = True

@dataclass
class UNDER_OVER(BaseConfig):
    expt_type: str = ExperimentTypes.CLASSIFY_OVER_UNDER
    dataset_dir: str = field(default_factory=lambda: [get_dataset_dir(ExperimentTypes.CLASSIFY_OVER_UNDER)])
    classes: int = 1
    img_height: int = 20
    img_width: int = 20
    crop_width: int = 10
    num_keypoints: int = 1
    gauss_sigma: int = 2
    epochs: int = 50
    batch_size: int = 4
    cond_point_dist_px: int = 20
    condition_len: int = 5
    pred_len: int = 0
    eval_checkpoint_freq: int = 1
    min_checkpoint_freq: int = 10
    resnet_type: str = '50'
    pretrained: bool = False
    rot_cond: bool = True

@dataclass
class UNDER_OVER_NONE(BaseConfig):
    expt_type: str = ExperimentTypes.CLASSIFY_OVER_UNDER_NONE
    dataset_dir: str = field(default_factory=lambda: [get_dataset_dir(ExperimentTypes.CLASSIFY_OVER_UNDER_NONE)])
    classes: int = 3
    img_height: int = 20
    img_width: int = 20
    crop_width: int = 10
    num_keypoints: int = 1
    gauss_sigma: int = 2
    epochs: int = 50
    batch_size: int = 4
    cond_point_dist_px: int = 20
    condition_len: int = 5
    pred_len: int = 0
    eval_checkpoint_freq: int = 1
    min_checkpoint_freq: int = 10
    resnet_type: str = '50'
    pretrained: bool = False
    rot_cond: bool = True

def get_class_name(cls):
    return cls.__name__

ALL_EXPERIMENTS_LIST = [BaseConfig, TRCR80, TRCR100, TRCR120, 
CL5_20_PL1, CL3_10_PL2, CL10_10_PL2, CL3_10_PL1, CL10_10_PL1, TRCR140_CL4_25_PL1, 
TRCR140_CL4_25_PL1_RN34, TRCR140_CL4_25_PL1_RN34_MED, TRCR80_CL4_25_PL1_RN34_MED, 
TRCR80_CL4_25_PL1_RN34_MED3, TRCR80_CL4_25_PL1_RN50_MED3, TRCR60_CL4_25_PL1_RN34_MED3_V2, 
TRCR60_CL4_25_PL1_RN50_MED3_V2, TRCR60_CL4_25_PL1_RN34_MED3_B32_V2, 
TRCR60_CL4_25_PL1_RN50_MED3_B32_V2, TRCR60_CL3_20_PL1_RN34_MED3_RN34_B64_OS, 
TRCR60_CL3_20_PL1_MED3_RN50_B64_OS, TRCR60_CL3_20_PL1_MED3_UNet34_B64_OS, 
TRCR60_CL3_20_PL1_MED3_UNet34_B64_OS_RotCond, 
TRCR50_CL3_15_PL1_MED3_UNet34_B64_OS_RotCond, 
TRCR50_CL3_15_PL1_MED3_UNet18_B64_OS_RotCond, 
TRCR50_CL3_15_PL1_MED3_UNet50_B64_OS_RotCond, 
TRCR50_CL3_15_PL1_MED3_UNet101_B64_OS_RotCond, 
TRCR40_CL3_15_PL1_MED3_UNet34_B64_OS_RotCond, 
TRCR36_CL3_14_PL1_MED3_UNet34_B64_OS_RotCond, 
TRCR32_CL3_12_PL1_MED3_UNet34_B64_OS_RotCond, 
TRCR40_CL3_15_PL1_MED3_UNet34_B64_OS_RotCond_Hard2, 
TRCR36_CL3_14_PL1_MED3_UNet34_B64_OS_RotCond_Hard2, 
TRCR32_CL3_12_PL1_MED3_UNet34_B64_OS_RotCond_Hard2, 
TRCR40_CL3_15_PL1_MED3_UNet34_B64_OS_RotCond_Adj1, 
TRCR36_CL3_14_PL1_MED3_UNet34_B64_OS_RotCond_Adj1, 
TRCR32_CL3_12_PL1_MED3_UNet34_B64_OS_RotCond_Adj1,
TRCR28_CL3_12_PL1_MED3_UNet34_B64_OS_RotCond_Hard2, 
TRCR24_CL3_12_PL1_MED3_UNet34_B64_OS_RotCond_Hard2,
TRCR32_CL3_12_PL1_MED3_UNet34_B64_OS_RotCond_Hard2_WReal, 
TRC_HW128,
TRCR32_CL3_12_PL1_MED3_UNet34_B64_OS_RotCond_Hard2_Medley,
TRCR32_CL3_12_PL1_MED3_UNet34_B64_OS_RotCond_Hard2_Medley_Sharp,
TRCR32_CL3_12_PL1_MED3_UNet34_B64_OS_RotCond_Hard2_Medley_Sharp_HW64,
TRCR32_CL3_12_PL1_MED3_UNet34_B64_OS_RotCond_Hard2_Medley_MoreReal_Sharp,
TRCR32_CL3_12_PL1_MED3_UNet34_B64_OS_RotCond_RealOnly_Sharp,
TRCR32_CL3_12_PL1_MED3_UNet50_B64_OS_RotCond_Hard2_Medley_MoreReal_Sharp,
UNDER_OVER,
UNDER_OVER_NONE]

ALL_EXPERIMENTS_CONFIG = {get_class_name(expt): expt for expt in ALL_EXPERIMENTS_LIST}
