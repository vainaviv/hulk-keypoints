import torch
import random
import cv2
import time
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import pickle
import os
from datetime import datetime
import imgaug.augmenters as iaa
from imgaug.augmentables import KeypointsOnImage
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
no_transform = iaa.Sequential([iaa.Resize({"height": 200, "width": 200})])

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
                 expt_type=ExperimentTypes.CLASSIFY_OVER_UNDER):
        self.img_height = img_height
        self.img_width = img_width
        self.gauss_sigma = gauss_sigma
        self.transform = transform
        self.img_transform = img_transform if augment else no_transform
        self.augment = augment
        self.condition_len = condition_len
        self.crop_width = crop_width
        self.crop_span = self.crop_width*2 + 1

        self.data = []
        self.expt_type = expt_type

        self.weights = np.geomspace(0.5, 1, condition_len)

        # if folder is a list, then iterate over folders
        if not isinstance(folder, list):
            folders = [folder]
        else:
            folders = folder

        for folder in folders:
            if os.path.exists(folder):
                for fname in sorted(os.listdir(folder)):
                    self.data.append(os.path.join(folder, fname))

    def _get_evenly_spaced_points_backward(self, pixels, num_points, start_idx, spacing):
        # get evenly spaced points
        last_point = np.array(pixels[start_idx]).squeeze()
        points = [last_point]
        while len(points) < num_points and start_idx > 0:
            start_idx -= 1
            if np.linalg.norm(np.array(pixels[start_idx]).squeeze() - last_point) > COND_POINT_DIST_PX:
                last_point = np.array(pixels[start_idx]).squeeze()
                points.append(last_point)
        return np.array(points)[..., ::-1]

    def __getitem__(self, data_index):
        start_time = time.time()
        loaded_data = np.load(self.data[data_index], allow_pickle=True).item()
        if self.expt_type == ExperimentTypes.TRACE_PREDICTION:
            img = loaded_data['img'][:, :, :3]
            pixels = loaded_data['pixels']
            # plt.clf()
            # plt.imshow(img)
            # for i in range(0, len(pixels), 10):
            #     plt.scatter(pixels[i][0][0], pixels[i][0][1], s=1)
            # plt.savefig(f'dataset_py_test/test_{data_index}.png')
            crop = np.zeros(1)
            while not np.array_equal(crop.shape, np.array([self.crop_span, self.crop_span, 3])):
                start_idx = np.random.randint(0, len(pixels) - (self.condition_len + 1))
                condition_pixels = self._get_evenly_spaced_points_backward(pixels, self.condition_len + 1, start_idx, COND_POINT_DIST_PX)   #[pixels[i][0][::-1] for i in range(start_idx, start_idx + (self.condition_len + 1))] #
                if len(condition_pixels) < self.condition_len + 1:
                    continue
                center_of_crop = condition_pixels[-2]
                crop = img[max(0, center_of_crop[0] - self.crop_width): min(img.shape[0], center_of_crop[0] + self.crop_width + 1),
                           max(0, center_of_crop[1] - self.crop_width): min(img.shape[1], center_of_crop[1] + self.crop_width + 1)]
            # print(center_of_crop, crop.shape, img.shape)
            img = crop
            top_left = [center_of_crop[0] - self.crop_width, center_of_crop[1] - self.crop_width]
            condition_pixels = [[pixel[0] - top_left[0], pixel[1] - top_left[1]] for pixel in condition_pixels]
        else:
            img = loaded_data['crop_img'][:, :, :3]
            condition_pixels = loaded_data['spline_pixels']
        
        # if img.max() <= 1.0:
        #     img = (img * 255.0).astype(np.uint8)
        if self.expt_type == ExperimentTypes.TRACE_PREDICTION:
            kpts = KeypointsOnImage.from_xy_array(np.array(condition_pixels)[:, ::-1], shape=img.shape)
            img, kpts = self.img_transform(image=img, keypoints=kpts)
            points = []
            for k in kpts:
                points.append([k.x,k.y])
            points = np.array(points)
            # Slow way of doing it
            # gaussians = gauss_2d_batch(self.img_width, self.img_height, self.gauss_sigma, points[:-1,0], points[:-1,1])
            # print(f'Time to load data: {time.time() - start_time}')
            # mm_gauss = gaussians[0]
            # for i in range(1, len(gaussians) - 1):
            #     mm_gauss = bimodal_gauss(mm_gauss * 0.9, gaussians[i])

            # print(points[-1:, 0], points[-1:, 1])
            img[:, :, 0] = gauss_2d_batch_efficient_np(self.crop_span, self.crop_span, self.gauss_sigma, points[:-1,0], points[:-1,1], weights=self.weights)
            combined = transform(img.copy()).cuda()

            label = torch.as_tensor(gauss_2d_batch_efficient_np(self.crop_span, self.crop_span, self.gauss_sigma, points[-1:, 0], points[-1:, 1], weights=[1])).unsqueeze_(0).cuda()
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
            else:
                raise Exception("No condition")
        if self.expt_type == ExperimentTypes.OPPOSITE_ENDPOINT_PREDICTION:
            end_mask = pull_with_cable_and_img[:, :, 6:9].copy()
            end_mask = cv2.resize(end_mask.astype(np.float64), (self.img_width, self.img_height))
            if end_mask.sum() > 0:
                end_V, end_U = np.nonzero(end_mask[:, :, 0])
                end_U, end_V = torch.from_numpy(np.array([end_U, end_V], dtype=np.int32)).cuda()
                label = 1.0 * get_gauss(self.img_width, self.img_height, self.gauss_sigma, end_U, end_V)

        # print(f'load time: {time.time() - start_time}')
        return combined, label
    
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_dataset = KeypointsDataset('/home/vainavi/hulk-keypoints/src/test_data',
                           IMG_HEIGHT('trp'), IMG_WIDTH('trp'), transform, gauss_sigma=GAUSS_SIGMA, augment=True,expt_type=ExperimentTypes.TRACE_PREDICTION)
    for i in range(0, 10):
        img, gauss = test_dataset[i] #[-1] #
        vis_gauss(img, gauss, i)