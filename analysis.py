import pickle
import cv2
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from config import *
from src.model import KeypointsGauss, ClassificationModel
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
model_ckpts = ['over_under_model_withtrace/model_2_1_49_26.02977597541978.pth']

folder_name = 'over_under_withtrace'
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
    keypoints = ClassificationModel(NUM_KEYPOINTS, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, channels=3).cuda()
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

test_dataset = KeypointsDataset('/home/kaushiks/hulk-keypoints/processed_sim_data/under_over_crossings_dataset/test', IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA, augment=False, only_full=True, condition=True, sim=False, trace_imgs=True)

preds = []
gts = []
for i, f in enumerate(test_dataset):
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

    heatmaps = []
    # create len(predictions) subplots
    for j, prediction in enumerate(predictions):
        output = prediction.predict(img_t[0])

    print(output, f[1])
    preds.append(output.detach().cpu().numpy().item())
    gts.append(f[1].detach().cpu().numpy().item())

    plt.title(f'Pred: {preds[-1]}, GT: {gts[-1]}')
    plt.savefig(f'{output_folder_name}/input_img_{i}.png')


# calculate auc score
import sklearn.metrics as metrics
fpr, tpr, thresholds = metrics.roc_curve(gts, preds, pos_label=1)
auc = metrics.auc(fpr, tpr)
print(auc)