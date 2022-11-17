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
parser.add_argument('--trace_if_trp', action='store_true', default=False)

flags = parser.parse_args()

experiment_time = time.strftime("%Y%m%d-%H%M%S")
checkpoint_path = flags.checkpoint_path
expt_type = flags.expt_type
trace_if_trp = flags.trace_if_trp

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

def trace(image, start_point_1, start_point_2, stop_when_crossing=False, resume_from_edge=False, timeout=30,
          bboxes=[], termination_map=None, viz=True, exact_path_len=None, viz_iter=None, filter_bad=False, x_min=None, x_max=None, y_min=None, y_max=None, start_points=None, model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])

    num_condition_points = 4
    crop_size = 100

    if start_points is None or len(start_points) < num_condition_points:
        raise ValueError(f"Need at least {len(start_points)} start points")
    path = [start_point for start_point in start_points]
    disp_img = image.copy()

    for iter in range(exact_path_len):
        condition_pixels = [p[::-1] for p in path[-num_condition_points:]]
        print(condition_pixels)
        crop_center = condition_pixels[-1]
        xmin, xmax, ymin, ymax = max(0, crop_center[0] - crop_size), min(image.shape[1], crop_center[0] + crop_size), max(0, crop_center[1] - crop_size), min(image.shape[0], crop_center[1] + crop_size)
        crop = image.copy()[ymin:ymax, xmin:xmax]
        cond_pixels_in_crop = condition_pixels - np.array([xmin, ymin])
        spline = test_dataset.draw_spline(crop, cond_pixels_in_crop[:, 1], cond_pixels_in_crop[:, 0], label=False)
        crop[:, :, 0] = spline
        # crop[:, :, 1] = 1 - spline
        crop_eroded = cv2.erode((crop > 0.1).astype(np.uint8), np.ones((8, 8)), iterations=1)

        cv2.imshow('model input', crop)
        cv2.waitKey(1000)
        if viz:
            plt.scatter(cond_pixels_in_crop[:, 0], cond_pixels_in_crop[:, 1], c='r')
            plt.imshow(crop)
            plt.show()

            plt.imshow(crop_eroded * 255)
            plt.show()
        
        model_input = transform(crop.copy()).unsqueeze(0).to(device)
        start_time = time.time()
        model_output = model(model_input).detach().cpu().numpy().squeeze()
        if viz or iter > 11:
            plt.imshow(model_output.squeeze())
            plt.show()

        model_output *= crop_eroded[:, :, 1]
        print("Time taken for prediction: ", time.time() - start_time)

        if viz or iter > 11:
            plt.imshow(model_output.squeeze())
            plt.show()

        argmax_yx = np.unravel_index(model_output.argmax(), model_output.shape)
        global_yx = np.array([argmax_yx[0] + ymin, argmax_yx[1] + xmin])

        path.append(global_yx)

        if viz:
            plt.scatter(argmax_yx[1], argmax_yx[0], c='r')
            plt.imshow(crop)
            plt.show()

            plt.scatter(global_yx[1], global_yx[0], c='r')
            plt.imshow(image)
            plt.show()
        
        disp_img = cv2.circle(disp_img, (global_yx[1], global_yx[0]), 1, (0, 0, 255), 2)
        # add line from previous to current point
        if len(path) > 1:
            disp_img = cv2.line(disp_img, (path[-2][1], path[-2][0]), (global_yx[1], global_yx[0]), (0, 0, 255), 2)
        cv2.imshow('image to display', disp_img)
        cv2.waitKey(1)

# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# torch.cuda.set_device(1)

expt_name = os.path.normpath(checkpoint_path).split(os.sep)[-2]
output_folder_name = f'preds/preds_{expt_name}'
if not os.path.exists(output_folder_name):
    os.mkdir(output_folder_name)
success_folder_name = os.path.join(output_folder_name, 'success')
if not os.path.exists(success_folder_name):
    os.mkdir(success_folder_name)
failure_folder_name = os.path.join(output_folder_name, 'failure')
if not os.path.exists(failure_folder_name):
    os.mkdir(failure_folder_name)

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
keypoints.load_state_dict(torch.load(model_ckpt))
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

if expt_type == ExperimentTypes.TRACE_PREDICTION and trace_if_trp:
    image_folder = '/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset/test'
    images = os.listdir(image_folder)
    for image in images:
        # print(os.path.join(image_folder, image))
        loaded_img = np.load(os.path.join(image_folder, image), allow_pickle=True).item()
        img = loaded_img['img'][:, :, :3]
        img_size = img.shape[0]

        # now get starting points
        pixels = loaded_img['pixels']
        for i in range(len(pixels)):
            cur_pixel = pixels[i][0]
            if cur_pixel[0] >= 0 and cur_pixel[1] >= 0 and cur_pixel[0] < img_size and cur_pixel[1] < img_size:
                start_idx = i
                break

        # print("HERE HERE HERE")
        starting_points = test_dataset._get_evenly_spaced_points(pixels, CONDITION_LEN + 1, start_idx, COND_POINT_DIST_PX, backward=False)
        spline = trace(img, None, None, exact_path_len=20, start_points=starting_points, model=keypoints_models[0])

        plt.imshow(img)
        for pt in spline:
            plt.scatter(pt[1], pt[0], c='r')
        plt.show()

else:
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
            output_yx = np.unravel_index(np.argmax(f[1][0].detach().cpu().numpy()), f[1][0].detach().cpu().numpy().shape)
            # print(argmax_yx, output_yx)
            output_heatmap = output.detach().cpu().numpy()[0, 0, ...]
            output_image = f[0][0:3, ...].detach().cpu().numpy().transpose(1,2,0)
            output_image[:, :, 2] = output_heatmap
            output_image = output_image.copy()
            output_image = (output_image * 255.0).astype(np.uint8)
            overlay = output_image # cv2.circle(output_image, (argmax_yx[1], argmax_yx[0]), 2, (255, 255, 255), -1)
            plt.imshow(overlay)
            save_path = os.path.join(failure_folder_name, f'output_img_{i}.png')
            if np.linalg.norm((np.array(argmax_yx) - np.array(output_yx)), 2) < 15:
                hits += 1
                save_path = os.path.join(success_folder_name, f'output_img_{i}.png')
            plt.savefig(save_path)

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