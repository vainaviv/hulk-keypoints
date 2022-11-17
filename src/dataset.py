import torch
import random
import cv2
import time
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import pickle
import os
from datetime import datetime
import imgaug.augmenters as iaa
from imgaug.augmentables import KeypointsOnImage
from scipy import interpolate
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/kaushiks/hulk-keypoints/')
from config import *

# No domain randomization
transform = transforms.Compose([transforms.ToTensor()])

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Domain randomization
img_transform = iaa.Sequential([
    iaa.flip.Fliplr(0.5),
    iaa.flip.Flipud(0.5),
    # iaa.Resize({"height": 200, "width": 200}),
    # sometimes(iaa.Affine(
    #     scale = {"x": (0.7, 1.3), "y": (0.7, 1.3)},
    #     rotate=(-30, 30),
    #     shear=(-30, 30)
    # ))
    ], random_order=True)

# No randomization
no_transform = iaa.Sequential([])

# New domain randomization
img_transform_new = iaa.Sequential([
    iaa.flip.Flipud(0.5),
    iaa.flip.Fliplr(0.5),
    # rotate 90, 180, or 270
    iaa.Rot90([0, 1, 2, 3]),
    sometimes(iaa.Affine(
        scale = {"x": (0.7, 1.3), "y": (0.7, 1.3)},
        rotate=(-30, 30),
        shear=(-30, 30)
        ))
    ], random_order=True)

def normalize(x):
    return F.normalize(x, p=1)

def gauss_2d_batch(width, height, sigma, U, V, normalize_dist=False, single=False):
    if not single:
        U.unsqueeze_(1).unsqueeze_(2)
        V.unsqueeze_(1).unsqueeze_(2)
    X,Y = torch.meshgrid([torch.arange(0., width), torch.arange(0., height)])
    X,Y = torch.transpose(X, 0, 1).cuda(), torch.transpose(Y, 0, 1).cuda()
    G=torch.exp(-((X-U.float())**2+(Y-V.float())**2)/(2.0*sigma**2))
    if normalize_dist:
        return (G/G.max()).double() * 2
    return G.double()

def gauss_2d_batch_efficient_np(width, height, sigma, U, V, weights):
    crop_size = 3 * sigma
    ret = np.zeros((height + 2*crop_size, width + 2*crop_size + 1))
    X,Y = np.meshgrid(np.arange(-crop_size, crop_size+1), np.arange(-crop_size, crop_size+1))
    gaussian = np.exp(-((X)**2+(Y)**2)/(2.0*sigma**2))
    for i in range(len(weights)):
        cur_weight = weights[i]
        y, x = int(V[i]) + crop_size, int(U[i]) + crop_size
        # print(y, x, width, height, ret.shape, ret[y-crop_size:y+crop_size+1, x-crop_size:x+crop_size+1].shape)
        if ret[y-crop_size:y+crop_size+1, x-crop_size:x+crop_size+1].shape == gaussian.shape:
           ret[y-crop_size:y+crop_size+1, x-crop_size:x+crop_size+1] = np.max((cur_weight * gaussian, ret[y-crop_size:y+crop_size+1, x-crop_size:x+crop_size+1]), axis=0)

    return ret[crop_size:crop_size+height, crop_size:crop_size+width]

def vis_gauss(img, gaussians, i):
    gaussians = gaussians.cpu().detach().numpy().transpose(1, 2, 0)
    img = (img.cpu().detach().numpy().transpose(1, 2, 0) * 255)
    gaussians = np.concatenate((gaussians, np.zeros_like(gaussians[:, :, :1]), np.zeros_like(gaussians[:, :, :1])), axis=2)
    h1 = gaussians
    output = cv2.normalize(h1, None, 0, 255, cv2.NORM_MINMAX)
    if not os.path.exists('./dataset_py_test'):
        os.mkdir('./dataset_py_test')
    cv2.imwrite(f'./dataset_py_test/test-gaussians_{i:05d}.png', output)
    cv2.imwrite(f'./dataset_py_test/test-img_{i:05d}.png', img[...,::-1])

def bimodal_gauss(G1, G2, normalize=False):
    bimodal = torch.max(G1, G2)
    if normalize:
        return normalize(bimodal)
    return bimodal

def get_gauss(w, h, sigma, U, V):
    gaussians = gauss_2d_batch(w, h, sigma, U, V)
    if gaussians.shape[0] > 0:
        mm_gauss = gaussians[0]
        for i in range(1, len(gaussians)):
            mm_gauss = bimodal_gauss(mm_gauss, gaussians[i])
        mm_gauss.unsqueeze_(0)
        return mm_gauss
    return torch.zeros(1, h, w).cuda().double()

class KeypointsDataset(Dataset):
    def __init__(self, folder, img_height, img_width, transform, gauss_sigma=8, augment=True, crop=True, condition_len=4, crop_width=100, 
                 pred_len=1, spacing=15, expt_type=ExperimentTypes.CLASSIFY_OVER_UNDER):
        self.img_height = img_height
        self.img_width = img_width
        self.gauss_sigma = gauss_sigma
        self.transform = transform
        self.img_transform = img_transform if augment else no_transform
        self.augment = augment
        self.condition_len = condition_len
        self.crop_width = crop_width
        self.crop_span = self.crop_width*2 + 1
        self.pred_len = pred_len
        self.spacing = spacing

        self.data = []
        self.expt_type = expt_type

        self.weights = np.geomspace(0.5, 1, self.condition_len)
        self.label_weights = np.ones(self.pred_len) #np.geomspace(1, 0.5, self.pred_len)

        # if folder is a list, then iterate over folders
        if not isinstance(folder, list):
            folders = [folder]
        else:
            folders = folder

        for folder in folders:
            if os.path.exists(folder):
                for fname in sorted(os.listdir(folder)):
                    self.data.append(os.path.join(folder, fname))

    def _get_evenly_spaced_points(self, pixels, num_points, start_idx, spacing, backward=True):
        # get evenly spaced points
        last_point = np.array(pixels[start_idx]).squeeze()
        points = [last_point]
        while len(points) < num_points and start_idx > 0 and start_idx < len(pixels):
            start_idx -= (int(backward) * 2 - 1)
            if np.linalg.norm(np.array(pixels[start_idx]).squeeze() - last_point) > spacing:
                last_point = np.array(pixels[start_idx]).squeeze()
                points.append(last_point)
        return np.array(points)[..., ::-1]

    def draw_spline(self, crop, x, y, label=False):
        if len(x) < 2:
            raise Exception("if drawing spline, must have 2 points minimum for label")
        k = len(x) - 1 if len(x) < 4 else 3
        tck,u     = interpolate.splprep( [x,y] ,s = 0, k=k)
        xnew,ynew = interpolate.splev( np.linspace( 0, 1, 100 ), tck,der = 0)
        xnew = np.array(xnew, dtype=int)
        ynew = np.array(ynew, dtype=int)

        x_in= np.where(xnew < crop.shape[0])
        xnew = xnew[x_in[0]]
        ynew = ynew[x_in[0]]
        y_in = np.where(ynew < crop.shape[1])
        xnew = xnew[y_in[0]]
        ynew = ynew[y_in[0]]

        spline = np.zeros(crop.shape[:2])
        if label:
            weights = np.ones(len(xnew))
        else:
            weights = np.geomspace(0.5, 1, len(xnew))

        spline[xnew, ynew] = weights
        spline = np.expand_dims(spline, axis=2)
        spline = np.tile(spline, 3)
        spline_dilated = cv2.dilate(spline, np.ones((5,5), np.uint8), iterations=1)
        return spline_dilated[:, :, 0]

    def __getitem__(self, data_index):
        start_time = time.time()
        loaded_data = np.load(self.data[data_index], allow_pickle=True).item()
        #TODO Jainil: this will be where most of your coding will happen. 
        # Lines 186-203 load the image and labels. you may need to add something to make this possible for your dataset.
        if self.expt_type == ExperimentTypes.TRACE_PREDICTION:
            img = loaded_data['img'][:, :, :3]
            pixels = loaded_data['pixels']
            crop = np.zeros(1)
            while not np.array_equal(crop.shape, np.array([self.crop_span, self.crop_span, 3])):
                start_idx = np.random.randint(0, len(pixels) - (self.condition_len + self.pred_len))
                condition_pixels = self._get_evenly_spaced_points(pixels, self.condition_len + self.pred_len, start_idx, self.spacing, backward=True)
                if len(condition_pixels) < self.condition_len + self.pred_len:
                    continue
                center_of_crop = condition_pixels[-self.pred_len-1]
                crop = img[max(0, center_of_crop[0] - self.crop_width): min(img.shape[0], center_of_crop[0] + self.crop_width + 1),
                           max(0, center_of_crop[1] - self.crop_width): min(img.shape[1], center_of_crop[1] + self.crop_width + 1)]
            img = crop
            top_left = [center_of_crop[0] - self.crop_width, center_of_crop[1] - self.crop_width]
            condition_pixels = [[pixel[0] - top_left[0], pixel[1] - top_left[1]] for pixel in condition_pixels]
        else if self.expt_type == ExperimentTypes.CAGE_PREDICTION:
            img = loaded_data['img'][:, :, :3]
            cage_point = loaded_data['cage_point']
            condition_pixels = loaded_data['spline_pixels']
        else:
            img = loaded_data['crop_img'][:, :, :3]
            condition_pixels = loaded_data['spline_pixels']
        
        if self.expt_type == ExperimentTypes.TRACE_PREDICTION:
            cond_pix_array = np.array(condition_pixels)[:, ::-1]
            jitter = np.random.uniform(-1, 1, size=cond_pix_array.shape)
            jitter[-1] = 0
            cond_pix_array = cond_pix_array + jitter
            kpts = KeypointsOnImage.from_xy_array(cond_pix_array, shape=img.shape)
            img, kpts = self.img_transform(image=img, keypoints=kpts)
            points = []
            for k in kpts:
                points.append([k.x,k.y])
            points = np.array(points)

            cable_mask = np.ones(img.shape[:2])
            cable_mask[img[:, :, 1] < 0.1] = 0
        
            img[:, :, 0] = self.draw_spline(img, points[:-self.pred_len,1], points[:-self.pred_len,0]) * cable_mask
            img[:, :, 1] = 1 - img[:, :, 0]
            combined = transform(img.copy()).cuda()

            if PRED_LEN == 1:
                label = torch.as_tensor(gauss_2d_batch_efficient_np(self.crop_span, self.crop_span, self.gauss_sigma, points[-self.pred_len:, 0], points[-self.pred_len:, 1], weights=self.label_weights))
            else:
                try:
                    label = torch.as_tensor(self.draw_spline(img, points[-self.pred_len:,1], points[-self.pred_len:,0], label=True)) 
                except:
                    label = torch.as_tensor(gauss_2d_batch_efficient_np(self.crop_span, self.crop_span, self.gauss_sigma, points[-self.pred_len:, 0], points[-self.pred_len:, 1], weights=self.label_weights))
            label = label * cable_mask
            label = label.unsqueeze_(0).cuda()
        elif self.expt_type == ExperimentTypes.CAGE_PINCH_PREDICTION:
            # TODO Jainil: change this to be check experiment type for cage pinch selection. 
            # Create code for adding the condition point into channel 0 of image. Generate the cage pinch label heatmaps. 
            # Use the code under "if self.expt_type == ExperimentTypes.TRACE_PREDICTION" to get an idea of how to do this
            pass
        else:
            # input processing
            condition_mask = np.zeros(img.shape)
            for condition_pixel in condition_pixels[:len(condition_pixels)//2]:
                condition_mask[int(condition_pixel[1]), int(condition_pixel[0])] = 1.0
            condition_with_cable = np.where(condition_mask > 0, 1, 0)
            if self.expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER:
                aug_input_concat_tuple = (img, condition_with_cable)
                label = torch.as_tensor(loaded_data['under_over']).double().cuda()
            elif self.expt_type == ExperimentTypes.OPPOSITE_ENDPOINT_PREDICTION:
                end_mask = np.zeros(img.shape)
                for condition in condition_pixels[len(condition_pixels)//2:]:
                    end_mask[int(condition[1]), int(condition[0])] = 1.0
                aug_input_concat_tuple = (img, condition_with_cable, end_mask)

        if self.expt_type != ExperimentTypes.TRACE_PREDICTION:
            pull_with_cable_and_img = self.img_transform(image=np.concatenate(aug_input_concat_tuple, axis=2))
            # split into img and mask again
            img = pull_with_cable_and_img[:, :, 0:3].copy()
            condition_with_cable = pull_with_cable_and_img[:, :, 3:6].copy()
            combined = cv2.resize(img.astype(np.float64), (self.img_width, self.img_height))
            combined = self.transform(combined).cuda().float()
            condition_with_cable = cv2.resize(condition_with_cable.astype(np.float64), (self.img_width, self.img_height))
            if condition_with_cable.sum() > 0:
                cond_V, cond_U = np.nonzero(condition_with_cable[:, :, 0])
                cond_U, cond_V = torch.from_numpy(np.array([cond_U, cond_V], dtype=np.int32)).cuda()
                combined[0] = 255.0 * get_gauss(self.img_width, self.img_height, self.gauss_sigma, cond_U, cond_V)
                # combined[1] = 255.0 * (1 - get_gauss(self.img_width, self.img_height, self.gauss_sigma, cond_U, cond_V))
            else:
                raise Exception("No condition")
        if self.expt_type == ExperimentTypes.OPPOSITE_ENDPOINT_PREDICTION:
            end_mask = pull_with_cable_and_img[:, :, 6:9].copy()
            end_mask = cv2.resize(end_mask.astype(np.float64), (self.img_width, self.img_height))
            if end_mask.sum() > 0:
                end_V, end_U = np.nonzero(end_mask[:, :, 0])
                end_U, end_V = torch.from_numpy(np.array([end_U, end_V], dtype=np.int32)).cuda()
                label = 1.0 * get_gauss(self.img_width, self.img_height, self.gauss_sigma, end_U, end_V)

        return combined, label
    
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    # TODO Jainil: run dataset.py to test if your dataloader works. If it works, then you can move onto training
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_dataset = KeypointsDataset('/home/kaushiks/hulk-keypoints/processed_sim_data/under_over_crossings_dataset/test',
                                    IMG_HEIGHT('oep'), 
                                    IMG_WIDTH('oep'), 
                                    transform, 
                                    gauss_sigma=GAUSS_SIGMA, 
                                    augment=True, 
                                    condition_len=6, 
                                    crop_width=50, 
                                    spacing=8, 
                                    expt_type=ExperimentTypes.OPPOSITE_ENDPOINT_PREDICTION, 
                                    pred_len=3)
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)
    for i_batch, sample_batched in enumerate(test_data):
        print(i_batch)
        img, gauss = sample_batched
        gauss = gauss.squeeze(0)
        img = img.squeeze(0)
        vis_gauss(img, gauss, i_batch)