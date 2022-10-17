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
from scipy.signal import convolve2d

GAUSS_SIGMA = 8

def get_density_map(img, kernel=150):
    img = cv2.dilate((img).astype(np.uint8), np.ones((6, 6)), iterations=1)
    img = img.squeeze()
    print(img.shape)
    # padded convolution with kernel size 
    kernel = np.ones((kernel, kernel), np.uint8)
    # every pixel in the kernel within radius of kernel/2 is 1, else 0
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            if (i - kernel.shape[0]/2)**2 + (j - kernel.shape[1]/2)**2 > kernel.shape[0]/2**2:
                kernel[i, j] = 0

    img = convolve2d(img, kernel, mode='same')
    return img


os.environ["CUDA_VISIBLE_DEVICES"]="1"
# torch.cuda.set_device(1)

# model_ckpts = ["hulkL_cond_aug21_icra_w_anal_NOMASK_1/model_2_1_499_0.5222878034335012.pth",
#                "hulkL_cond_aug21_icra_w_anal_NOMASK_2/model_2_1_499_0.5374186725368331.pth",
#                "hulkL_cond_aug21_icra_w_anal_NOMASK_3/model_2_1_499_0.5009409086017771.pth"]
# model_ckpts = ['hulkL_cond_aug21_icra_w_anal_NOMASK_lr2e-5_sigma_wpreRSS_2_3/model_2_1_299_0.16817463159559926.pth',
#                'hulkL_cond_aug21_icra_w_anal_NOMASK_lr2e-5_sigma_wpreRSS_2_3/model_2_1_299_0.18750610115741131.pth',
#                'hulkL_cond_aug21_icra_w_anal_NOMASK_lr2e-5_sigma_wpreRSS_2_2/model_2_1_299_0.15622884791929934.pth',
#                'hulkL_cond_aug21_icra_w_anal_NOMASK_lr2e-5_sigma_wpreRSS_2_1/model_2_1_299_0.14870232070779832.pth',]
# model_ckpts = ['hulkL_cond_rnet34_wnotrace3_always_sep5/model_2_1_99_0.08647923276907996.pth'] 
# model_ckpts = ['hulkL_cond_rnet34_wtrace3_always_sep5/model_2_1_99_0.056361758567128994.pth']
# model_ckpts = ['hulkL_cond_rnet34_wnotrace4_always_sep6/model_2_1_199_0.057098522507624754.pth',
#                'hulkL_cond_rnet34_wnotrace4_always_sep6_2/model_2_1_199_0.054455696754075314.pth',
#                'hulkL_cond_rnet34_wnotrace4_always_sep6_3/model_2_1_199_0.056172625463381906.pth']
# model_ckpts = ['hulkL_cond_rnet34_wtrace4_always_sep6/model_2_1_199_0.057852378280007.pth']
model_ckpts = ['hulkL_cond_rnet34_wnotrace5/model_2_1_199_0.11578577275285896.pth', 
'hulkL_cond_rnet34_wnotrace5-2/model_2_1_199_0.08690919075492234.pth',
'hulkL_cond_rnet34_wnotrace5-3/model_2_1_199_0.07517632795884831.pth']

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

test_dataset = KeypointsDataset('hulkL_trace_4/test/', IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA, augment=False, only_full=True, condition=True, sim=False, trace_imgs=True)
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

    img_masked = img_t.detach().cpu().numpy()[0, 2:3, ...] > 100/255
    # density_map = get_density_map(img_masked)
    # plt.clf()
    # plt.imshow(np.hstack((img_masked.squeeze() * density_map,)))
    # plt.colorbar()
    # plt.savefig(f'{output_folder_name}/density_map_{i}.png')

    input_img_np = img_t.detach().cpu().numpy()[0, 0:3, ...]
    plt.clf()
    plt.imshow(input_img_np.transpose(1,2,0))
    plt.savefig(f'{output_folder_name}/input_img_{i}.png')

    heatmaps = []
    # create len(predictions) subplots
    for j, prediction in enumerate(predictions):
        heatmap = prediction.predict(img_t[0])

        heatmap = heatmap.detach().cpu().numpy()
        heatmap = heatmap * img_masked[None, :]
        heatmaps.append(heatmap)
        horiz_concat = None
        for layer in range(NUM_KEYPOINTS):
            overlay = prediction.plot(img_t.detach().cpu().numpy(), heatmap, image_id=i, write_image=False, heatmap_id=layer)
            if (horiz_concat is None):
                horiz_concat = overlay
            else:
                horiz_concat = np.hstack((horiz_concat, overlay))
        plt.clf()
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
    plt.title(f"Min Model {j+1} Cage and Pinch, Max Values {min_heatmap[0, 0].max():.3f}, {min_heatmap[0, 1].max():.3f}")
    plt.savefig(f'{output_folder_name}/test_min_heatmaps_{i}_{layer}.png')