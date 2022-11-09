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
import argparse

GAUSS_SIGMA = 8

# parse command line flags
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default='default')
parser.add_argument('--expt_type', type=str, default='trp')

flags = parser.parse_args()

experiment_time = time.strftime("%Y%m%d-%H%M%S")
checkpoint_path = flags.checkpoint_path
expt_type = flags.expt_type

model_ckpt = flags.checkpoint_path

def get_density_map(img, kernel=150):
    img = cv2.dilate((img).astype(np.uint8), np.ones((6, 6)), iterations=1)
    img = img.squeeze()
    # padded convolution with kernel size 
    kernel = np.ones((kernel, kernel), np.uint8)
    # every pixel in the kernel within radius of kernel/2 is 1, else 0
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            if (i - kernel.shape[0]/2)**2 + (j - kernel.shape[1]/2)**2 > kernel.shape[0]/2**2:
                kernel[i, j] = 0

    img = convolve2d(img, kernel, mode='same')
    return img

# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# torch.cuda.set_device(1)

expt_name = os.path.normpath(checkpoint_path).split(os.sep)[-2]
output_folder_name = f'preds/preds_{expt_name}'
if not os.path.exists(output_folder_name):
    os.mkdir(output_folder_name)

# cuda
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(0)
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

# model
keypoints_models = []
# for model_ckpt in model_ckpts:
if expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER:
    keypoints = ClassificationModel(NUM_KEYPOINTS, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, channels=3).cuda()
elif is_point_pred(expt_type):
    keypoints = KeypointsGauss(1, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, channels=3).cuda()
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

test_dataset = KeypointsDataset(os.path.join(get_dataset_dir(expt_type), 'test'), 
                                IMG_HEIGHT, 
                                IMG_WIDTH, 
                                transform,
                                gauss_sigma=GAUSS_SIGMA, 
                                augment=False, 
                                expt_type=expt_type, 
                                condition_len=CONDITION_LEN, 
                                crop_width=CROP_WIDTH, 
                                spacing=COND_POINT_DIST_PX)

preds = []
gts = []
hits = 0
total = 0
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

    input_img_np = img_t.detach().cpu().numpy()[0, 0:3, ...]
    # plt.clf()
    # plt.imshow(input_img_np.transpose(1,2,0))

    # plt.savefig(f'{output_folder_name}/input_img_{i}.png')

    heatmaps = []
    # create len(predictions) subplots
    for j, prediction in enumerate(predictions):
        output = prediction.predict(img_t[0])

    if expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER:
        preds.append(output.detach().cpu().numpy().item())
        gts.append(f[1].detach().cpu().numpy().item())
        plt.title(f'Pred: {preds[-1]}, GT: {gts[-1]}')
    elif is_point_pred(expt_type):
        argmax_yx = np.unravel_index(np.argmax(output.detach().cpu().numpy()[0, 0, ...]), output.detach().cpu().numpy()[0, 0, ...].shape)
        output_yx = np.unravel_index(np.argmax(f[1].detach().cpu().numpy()[0, 0, ...]), f[1].detach().cpu().numpy()[0, 0, ...].shape)
        if np.linalg.norm((np.array(argmax_yx) - np.array(output_yx)), 2) < 4:
            hits += 1
        output_heatmap = output.detach().cpu().numpy()[0, 0, ...]
        output_image = f[0][0:3, ...].detach().cpu().numpy().transpose(1,2,0)
        output_image[:, :, 2] = output_heatmap
        output_image = output_image.copy()
        output_image = (output_image * 255.0).astype(np.uint8)
        overlay = output_image # cv2.circle(output_image, (argmax_yx[1], argmax_yx[0]), 2, (255, 255, 255), -1)
        plt.imshow(overlay)
        plt.savefig(f'{output_folder_name}/output_img_{i}.png')

    # check if the gt at argmax is 1
    total += 1

if expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER:
    # calculate auc score
    import sklearn.metrics as metrics
    fpr, tpr, thresholds = metrics.roc_curve(gts, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print("Classification AUC:", auc)
elif is_point_pred(expt_type):
    print("Mean within threshold accuracy:", hits/total)