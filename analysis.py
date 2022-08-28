import pickle
import cv2
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from config import *
from src.model import KeypointsGauss
from src.dataset import KeypointsDataset, transform, gauss_2d_batch, bimodal_gauss, get_gauss
from src.prediction import Prediction
from datetime import datetime, time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

GAUSS_SIGMA = 8

os.environ["CUDA_VISIBLE_DEVICES"]="1"
# torch.cuda.set_device(1)

# WITHOUT COND
model_ckpts = ["hulkL_cond_aug21_icra_w_anal_NOMASK_1/model_2_1_499_0.5222878034335012.pth",
               "hulkL_cond_aug21_icra_w_anal_NOMASK_2/model_2_1_499_0.5374186725368331.pth",
               "hulkL_cond_aug21_icra_w_anal_NOMASK_3/model_2_1_499_0.5009409086017771.pth"]

folder_name = 'new_ensemble'
output_folder_name = f'preds_{folder_name}'
if not os.path.exists(output_folder_name):
    os.mkdir(output_folder_name)

# cuda
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(0)
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

# model
keypoints_models = []
for model_ckpt in model_ckpts:
    keypoints = KeypointsGauss(NUM_KEYPOINTS, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, channels=3).cuda()
    keypoints.load_state_dict(torch.load('checkpoints/%s'%model_ckpt))
    keypoints_models.append(keypoints)

if use_cuda:
    for keypoints in keypoints_models:
        keypoints = keypoints.cuda()

predictions = []

for keypoints in keypoints_models:
    prediction = Prediction(keypoints, NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, use_cuda)
    predictions.append(prediction)

transform = transform = transforms.Compose([
    transforms.ToTensor()
])

test_dataset = KeypointsDataset('hulkL_seg/test/', IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA, augment=False, only_full=True, condition=True, sim=False)
# test_data = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
custom_image = None

for i, f in enumerate(test_dataset):
    print(i)
    img_t = f[0]
    if (len(img_t.shape) < 4):
        img_t = img_t.unsqueeze(0)

    # display image and user will click on two points
    plt.clf()
    plt.imshow(img_t[0].squeeze().detach().cpu().numpy().transpose(1,2,0))
    # plot one heatmap for each model with matplotlib
    plt.figure()

    heatmaps = []
    # create len(predictions) subplots
    for j, prediction in enumerate(predictions):
        heatmap = prediction.predict(img_t[0])
        heatmap = heatmap.detach().cpu().numpy()
        print(img_t.shape)
        heatmap = heatmap * (img_t.detach().cpu().numpy()[None, 0, 1:2, ...] > 100/255) # * (img_t.detach().cpu().numpy()[None, 0, 1:2, ...] > 100.0/255.0)
        heatmaps.append(heatmap)
        horiz_concat = None
        for layer in range(NUM_KEYPOINTS):
            overlay = prediction.plot(img_t.detach().cpu().numpy(), heatmap, image_id=i, write_image=False, heatmap_id=layer)
            if (horiz_concat is None):
                horiz_concat = overlay
            else:
                horiz_concat = np.hstack((horiz_concat, overlay))

        plt.subplot(1, len(predictions), j+1)
        plt.imshow(horiz_concat)
        plt.title("Model %d"%(j+1))
    plt.savefig(f'{output_folder_name}/test_heatmaps_{i}.png')

    # create the min heatmap
    min_heatmap = np.min(heatmaps, axis=0)
    horiz_concat = None
    for layer in range(NUM_KEYPOINTS):
        plt.clf()
        # print(min_heatmap.shape)
        overlay = prediction.plot(img_t.detach().cpu().numpy(), min_heatmap, image_id=i, write_image=False, heatmap_id=layer)
        if (horiz_concat is None):
            horiz_concat = overlay
        else:
            horiz_concat = np.hstack((horiz_concat, overlay))

    plt.imshow(horiz_concat)
    plt.title(f"Min Model {j+1} Cage and Pinch, Max Values {min_heatmap[0].max()}, {min_heatmap[1].max()}")
    plt.savefig(f'{output_folder_name}/test_min_heatmaps_{i}_{layer}.png')