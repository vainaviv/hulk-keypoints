import pickle
import cv2
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from src.model import KeypointsGauss, ClassificationModel
from src.dataset import KeypointsDataset, transform, gauss_2d_batch, bimodal_gauss, get_gauss
from src.prediction import Prediction
from datetime import datetime, time
from PIL import Image
import imgaug.augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import convolve2d
import argparse
from annot_real_img import REAL_WORLD_DICT 
from config import load_config_class, is_point_pred, get_dataset_dir, ExperimentTypes

def visualize_heatmap_on_image(img, heatmap):
    argmax = list(np.unravel_index(np.argmax(heatmap), heatmap.shape))[::-1]
    # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    # plot argmax location as white dot
    cv2.circle(cam, argmax, 1, (1,1,1), -1)
    return cam

def center_pixels_on_cable(image, pixels):
    # for each pixel, find closest pixel on cable
    image_mask = image[:, :, 0] > 100
    # erode white pixels
    kernel = np.ones((2,2),np.uint8)
    image_mask = cv2.erode(image_mask.astype(np.uint8), kernel, iterations=1)
    white_pixels = np.argwhere(image_mask)

    processed_pixels = []
    for pixel in pixels:
        # find closest pixel on cable
        distances = np.linalg.norm(white_pixels - pixel[::-1], axis=1)
        closest_pixel = white_pixels[np.argmin(distances)]
        processed_pixels.append([closest_pixel])

    # plot pixels on cable
    # plt.imshow(image_mask)
    # for pixel in processed_pixels:
    #     plt.scatter(pixel[0][1], pixel[0][0], c='r')
    # plt.show()

    return np.array(processed_pixels)[:, ::-1]

# parse command line flags
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--checkpoint_file_name', type=str, default='')
parser.add_argument('--trace_if_trp', action='store_true', default=False)
parser.add_argument('--real_world_trace', action='store_true', default=False)

flags = parser.parse_args()

experiment_time = time.strftime("%Y%m%d-%H%M%S")
checkpoint_path = flags.checkpoint_path
checkpoint_file_name = flags.checkpoint_file_name
trace_if_trp = flags.trace_if_trp
real_world_trace = flags.real_world_trace

if checkpoint_path == '':
    raise ValueError("--checkpoint_path must be specified")

min_loss, min_checkpoint = 100000, None
if checkpoint_file_name == '':
    # choose the one with the lowest loss
    for file in os.listdir(checkpoint_path):
        if file.endswith(".pth"):
            # file is structured as "..._loss.pth", extract loss
            loss = float(file.split('_')[-1].split('.')[-2])
            if loss < min_loss:
                min_loss = loss
                min_checkpoint = os.path.join(checkpoint_path, file)
    checkpoint_file_name = min_checkpoint
else:
    checkpoint_file_name = os.path.join(checkpoint_path, checkpoint_file_name)

# laod up all the parameters from the checkpoint
config = load_config_class(checkpoint_path)
expt_type = config.expt_type

print("Using checkpoint: ", checkpoint_file_name)
print("Loaded config: ", config)

def trace(image, start_point_1, start_point_2, stop_when_crossing=False, resume_from_edge=False, timeout=30,
          bboxes=[], termination_map=None, viz=True, exact_path_len=None, viz_iter=None, filter_bad=False, x_min=None, x_max=None, y_min=None, y_max=None, start_points=None, model=None):    
    num_condition_points = config.condition_len
    if start_points is None or len(start_points) < num_condition_points:
        raise ValueError(f"Need at least {num_condition_points} start points")
    path = [start_point for start_point in start_points]
    disp_img = (image.copy() * 255.0).astype(np.uint8)

    for iter in range(exact_path_len):
        print(iter)
        # tm = time.time()
        condition_pixels = [p for p in path[-num_condition_points:]]
        # print(condition_pixels)
        
        # crop_center = condition_pixels[-1]
        # xmin, xmax, ymin, ymax = max(0, crop_center[0] - crop_size), min(image.shape[1], crop_center[0] + crop_size), max(0, crop_center[1] - crop_size), min(image.shape[0], crop_center[1] + crop_size)
        # crop = image.copy()[ymin:ymax, xmin:xmax]
        # cond_pixels_in_crop = condition_pixels - np.array([xmin, ymin])
        
        crop, cond_pixels_in_crop, top_left = test_dataset.get_crop_and_cond_pixels(image, condition_pixels, center_around_last=True)
        ymin, xmin = np.array(top_left) - test_dataset.crop_width

        model_input, _, cable_mask, angle = test_dataset.get_trp_model_input(crop, cond_pixels_in_crop, test_dataset.img_transform, center_around_last=True)

        crop_eroded = cv2.erode((cable_mask).astype(np.uint8), np.ones((2, 2)), iterations=1)
        # print("Model input prep time: ", time.time() - tm)

        if viz:
            cv2.imshow('model input', model_input.detach().cpu().numpy().transpose(1, 2, 0))
            cv2.waitKey(1)

        model_output = model(model_input.unsqueeze(0)).detach().cpu().numpy().squeeze()
        model_output *= crop_eroded.squeeze()
        model_output = cv2.resize(model_output, (crop.shape[1], crop.shape[0]))

        # undo rotation if done in preprocessing
        M = cv2.getRotationMatrix2D((model_output.shape[1]/2, model_output.shape[0]/2), -angle*180/np.pi, 1)
        model_output = cv2.warpAffine(model_output, M, (model_output.shape[1], model_output.shape[0]))

        # mask model output by disc of radius COND_POINT_DIST_PX around the last condition pixel
        tolerance = 2
        last_condition_pixel = cond_pixels_in_crop[-1]
        # now create circle of radius COND_POINT_DIST_PX + tolerance
        # X, Y = np.meshgrid(np.arange(model_output.shape[1]), np.arange(model_output.shape[0]))
        # outer_circle = (X - last_condition_pixel[0])**2 + (Y - last_condition_pixel[1])**2 < (config.cond_point_dist_px + tolerance)**2
        # inner_circle = (X - last_condition_pixel[0])**2 + (Y - last_condition_pixel[1])**2 < (config.cond_point_dist_px - tolerance)**2
        # disc = (outer_circle & ~inner_circle)

        # model_output *= disc

        # if viz:
        #     plt.imshow(disc.squeeze())
        #     plt.show()

        # print(model_output.shape, np.unravel_index(model_output.argmax(), model_output.shape))
        # print(crop.shape, np.unravel_index(model_output.argmax(), model_output.shape), np.array([crop.shape[0] / config.img_height, crop.shape[1] / config.img_width]))
        # print(ymin, xmin)
        argmax_yx = np.unravel_index(model_output.argmax(), model_output.shape)# * np.array([crop.shape[0] / config.img_height, crop.shape[1] / config.img_width])

        # get angle of argmax yx
        global_yx = np.array([argmax_yx[0] + ymin, argmax_yx[1] + xmin]).astype(int)

        path.append(global_yx)
        # print("Model output post-processing time: ", time.time() - tm)

        if viz:
            # plt.scatter(argmax_yx[1], argmax_yx[0], c='r')
            # plt.imshow(crop)
            # plt.show()

            cv2.imshow('heatmap on crop', visualize_heatmap_on_image(crop, model_output))
            cv2.waitKey(1)
            # plt.scatter(global_yx[1], global_yx[0], c='r')
            # plt.imshow(image)
            # plt.show()

        disp_img = cv2.circle(disp_img, (global_yx[1], global_yx[0]), 1, (0, 0, 255), 2)
        # add line from previous to current point
        if len(path) > 1:
            disp_img = cv2.line(disp_img, (path[-2][1], path[-2][0]), (global_yx[1], global_yx[0]), (0, 0, 255), 2)
        plt.imsave(f'preds/disp_img_{i}.png', disp_img)
        cv2.imshow("disp_img", disp_img)
        cv2.waitKey(1)

expt_name = os.path.normpath(checkpoint_path).split(os.sep)[-1]
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
    keypoints = ClassificationModel(config.num_keypoints, img_height=config.img_height, img_width=config.img_width, channels=3).cuda()
elif is_point_pred(expt_type):
    keypoints = KeypointsGauss(1, img_height=config.img_height, img_width=config.img_width, channels=3, resnet_type=config.resnet_type, pretrained=config.pretrained).cuda()
keypoints.load_state_dict(torch.load(checkpoint_file_name))
keypoints_models.append(keypoints)

if use_cuda:
    for keypoints in keypoints_models:
        keypoints = keypoints.cuda()

predictions = []

for keypoints in keypoints_models:
    prediction = Prediction(keypoints, config.num_keypoints, config.img_height, config.img_width, use_cuda)
    predictions.append(prediction)

transform = transform = transforms.Compose([
    transforms.ToTensor()
])

test_dataset = KeypointsDataset(os.path.join(config.dataset_dir, 'test'), 
                                    config.img_height, 
                                    config.img_width, 
                                    transform,
                                    gauss_sigma=config.gauss_sigma, 
                                    augment=False, 
                                    condition_len=config.condition_len, 
                                    crop_width=config.crop_width,
                                    spacing=config.cond_point_dist_px,
                                    expt_type=config.expt_type,
                                    pred_len=config.pred_len,
                                    real_world=real_world_trace,
                                    oversample=True,
                                    config=config)

if expt_type == ExperimentTypes.TRACE_PREDICTION and trace_if_trp:
    if not real_world_trace:
        image_folder = '/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_medium_2/test'
        images = os.listdir(image_folder)
    else:
        image_folder = ''
        images = list(REAL_WORLD_DICT.keys())

    for i, image in enumerate(images):
        print(images[i])
        # if i < :
        #     continue
        # print(os.path.join(image_folder, image))
        if not real_world_trace:
            loaded_img = np.load(os.path.join(image_folder, image), allow_pickle=True).item()
            img = loaded_img['img'][:, :, :3]
            pixels = loaded_img['pixels']
        else:
            img = np.load(os.path.join(image_folder, image), allow_pickle=True)
            pixels = center_pixels_on_cable(img, REAL_WORLD_DICT[image])[..., ::-1]

        # now get starting points
        # print(pixels, img.shape)
        for j in range(len(pixels)):
            cur_pixel = pixels[j][0]
            if cur_pixel[0] >= 0 and cur_pixel[1] >= 0 and cur_pixel[1] < img.shape[0] and cur_pixel[0] < img.shape[1]:
                start_idx = j
                break

        # print(start_idx, pixels)
        starting_points = test_dataset._get_evenly_spaced_points(pixels, config.condition_len, start_idx + 1, config.cond_point_dist_px, img.shape, backward=False, randomize_spacing=False)
        # print(len(starting_points), config.condition_len)
        if len(starting_points) < config.condition_len:
            continue

        # plt.imshow(img)
        # for pt in starting_points:
        #     plt.scatter(pt[1], pt[0], c='r')
        # plt.show()

        # normalize image from 0 to 1
        if img.max() > 1:
            img = (img / 255.0).astype(np.float32)

        spline = trace(img, None, None, exact_path_len=80, start_points=starting_points, model=keypoints_models[0])

        # plt.imshow(img)
        # for pt in spline:
        #     plt.scatter(pt[1], pt[0], c='r')
        # plt.show()

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

        plt.clf()
        plt.imshow(img_t[0].squeeze().detach().cpu().numpy().transpose(1,2,0))
        plt.savefig(f'{output_folder_name}/input_img_{i}.png'.format(i=i))

        # plot one heatmap for each model with matplotlib
        plt.figure()

        # input_img_np = img_t.detach().cpu().numpy()[0, 0:3, ...]
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
            print("argmax_yx", argmax_yx)
            output_heatmap = output.detach().cpu().numpy()[0, 0, ...]
            output_image = f[0][0:3, ...].detach().cpu().numpy().transpose(1,2,0)
            # output_image[:, :, 2] = output_heatmap * 255
            output_image[:, :, 2] = output_heatmap 
            output_image = output_image.copy()

            vis_image = visualize_heatmap_on_image(img_t[0].squeeze().detach().cpu().numpy().transpose(1,2,0), output_heatmap)
            vis_image = cv2.circle(vis_image, (output_yx[1], output_yx[0]), 1, (0, 255, 255), -1)
            
            # output_image = (output_image * 255.0).astype(np.uint8)
            overlay = output_image
            #adding white circle for argmax of cage point prediction because gaussian heatmap is too uncertain
            if(expt_type == ExperimentTypes.CAGE_PREDICTION):
                cv2.circle(output_image, (argmax_yx[1], argmax_yx[0]), 2, (255, 255, 255), -1)
            plt.imshow(overlay)

            save_path = os.path.join(failure_folder_name, f'output_img_{i}.png')
            if np.linalg.norm((np.array(argmax_yx) - np.array(output_yx)), 2) < 5:
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
