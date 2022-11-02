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

# set torch GPU to 3
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# No domain randomization
transform = transforms.Compose([transforms.ToTensor()])

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Domain randomization
img_transform = iaa.Sequential([
    iaa.flip.Fliplr(0.5),
    iaa.flip.Flipud(0.5),
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
        # return normalize(G).double()
        return (G/G.max()).double() * 2
    return G.double()

def vis_gauss(img, gaussians, i):
    gaussians = gaussians.cpu().numpy().transpose(1, 2, 0)
    img = img.cpu().numpy().transpose(1, 2, 0) * 255.0
    # repeat last dimension 3 times
    # gaussians = np.tile(gaussians, (1, 1, 1))
    # print(gaussians.shape)
    gaussians = np.concatenate((gaussians, np.zeros_like(gaussians[:, :, :1]), np.zeros_like(gaussians[:, :, :1])), axis=2)
    h1 = gaussians
    # print(h1.shape)
    output = cv2.normalize(h1, None, 0, 255, cv2.NORM_MINMAX)
    # print("writing images")
    cv2.imwrite(f'dataset_py_test/test-gaussians_{i:05d}.png', output)
    cv2.imwrite(f'dataset_py_test/test-img_{i:05d}.png', img[...,::-1])

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
    def __init__(self, folder, img_height, img_width, transform, gauss_sigma=8, augment=True, crop=True,
                 only_full=False, condition=False, sim=False, trace_imgs=False, expt_type=ExperimentTypes.CLASSIFY_OVER_UNDER):
        self.img_height = img_height
        self.img_width = img_width
        self.gauss_sigma = gauss_sigma
        self.transform = transform
        self.img_transform = img_transform if augment else no_transform
        self.augment = augment

        self.imgs = []
        self.labels = []
        self.new_versions = []

        self.crop = crop
        self.only_full = only_full
        self.trace_imgs = trace_imgs

        self.total_label_count = 0
        self.expt_type = expt_type

        # if folder is a list, then iterate over folders
        if not isinstance(folder, list):
            folders = [folder]
        else:
            folders = folder

        for folder in folders:
            if os.path.exists(folder):
                for fname in sorted(os.listdir(folder)):
                    try:
                        img_and_annot = np.load(os.path.join(folder, fname), allow_pickle=True).item()
                    except:
                        continue
                    self.imgs.append(os.path.join(folder, fname))
                    self.labels.append({'under_over': img_and_annot['under_over'], 'spline_pixels': img_and_annot['spline_pixels']})
                    self.total_label_count += 1
        print('Loaded %d images'%len(self.imgs))

    def __getitem__(self, data_index):
        loaded_data = np.load(self.imgs[data_index], allow_pickle=True).item()
        img = loaded_data['crop_img'][:, :, :3]
        annots = self.labels[data_index]

        condition_pixels = annots['spline_pixels']

        # fill in rope between bbox corners
        if img.max() <= 1.0:
            # sim images
            img = (img * 255.0).astype(np.uint8)
            masked_img = img
        else:
            # pass
            masked_img = img.copy()
        # print(f"masked_img: {masked_img.shape}")
        img_mask = np.where(masked_img > 100, 255.0, 0.0)

        condition_mask = np.zeros(img.shape)
        for condition_pixel in condition_pixels[:len(condition_pixels)//2]:
            condition_mask[int(condition_pixel[1]), int(condition_pixel[0])] = 1.0
        # print("condition_mask sum", condition_mask.sum())
        condition_with_cable = np.where(condition_mask > 0, 1, 0)
        # print("condition with cable sum", condition_with_cable.sum())

        aug_input_concat_tuple = (masked_img, condition_with_cable)

        if self.expt_type == ExperimentTypes.OPPOSITE_ENDPOINT_PREDICTION:
            end_mask = np.zeros(img.shape)
            for condition in condition_pixels[len(condition_pixels)//2:]:
                end_mask[int(condition[1]), int(condition[0])] = 1.0

            aug_input_concat_tuple = (masked_img, condition_with_cable, end_mask)

        pull_with_cable_and_masked_img = self.img_transform(image=np.concatenate(aug_input_concat_tuple, axis=2))
        # split into img and mask again
        masked_img = pull_with_cable_and_masked_img[:, :, 0:3].copy()
        condition_with_cable = pull_with_cable_and_masked_img[:, :, 3:6].copy()
        if self.expt_type == ExperimentTypes.OPPOSITE_ENDPOINT_PREDICTION:
            end_mask = pull_with_cable_and_masked_img[:, :, 6:9].copy()

        combined = cv2.resize(masked_img.astype(np.float64), (self.img_width, self.img_height))
        combined = self.transform(combined).cuda().float()
        condition_with_cable = cv2.resize(condition_with_cable.astype(np.float64), (self.img_width, self.img_height))

        if condition_with_cable.sum() > 0:
            cond_V, cond_U = np.nonzero(condition_with_cable[:, :, 0])
            cond_U, cond_V = torch.from_numpy(np.array([cond_U, cond_V], dtype=np.int32)).cuda()
            combined[0] = 255.0 * get_gauss(self.img_width, self.img_height, self.gauss_sigma, cond_U, cond_V)
        else:
            raise Exception("No condition")

        if self.expt_type == ExperimentTypes.OPPOSITE_ENDPOINT_PREDICTION:
            end_mask = cv2.resize(end_mask.astype(np.float64), (self.img_width, self.img_height))
            if end_mask.sum() > 0:
                end_V, end_U = np.nonzero(end_mask[:, :, 0])
                end_U, end_V = torch.from_numpy(np.array([end_U, end_V], dtype=np.int32)).cuda()
                label = 1.0 * get_gauss(self.img_width, self.img_height, self.gauss_sigma, end_U, end_V)
            else:
                raise Exception("No end")
        else:
            label = torch.as_tensor(annots['under_over']).double().cuda()

        return combined / 255.0, label
    
    def __len__(self):
        return len(self.labels)

TEST_DIR = "hulkL_trace"
if __name__ == '__main__':
    IMG_WIDTH = 200
    IMG_HEIGHT = 200
    GAUSS_SIGMA = 20
    test_dataset = KeypointsDataset('/home/kaushiks/hulk-keypoints/processed_sim_data/under_over_crossings_dataset/train/',
                           IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA, only_full=True, condition=True, sim=False, trace_imgs=True, expt_type=ExperimentTypes.OPPOSITE_ENDPOINT_PREDICTION)
    for i in range(0, 10):
        img, overunder = test_dataset[i] #[-1] #
        vis_gauss(img, overunder, i)