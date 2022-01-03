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

# set torch GPU to 3
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# No domain randomization
transform = transforms.Compose([transforms.ToTensor()])

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Domain randomization
img_transform = iaa.Sequential([
    iaa.flip.Fliplr(0.5),
    iaa.flip.Flipud(0.5),
    sometimes(iaa.Affine(
        scale = {"x": (0.7, 1.3), "y": (0.7, 1.3)},
        rotate=(-30, 30),
        shear=(-30, 30)
    ))
    ], random_order=True)

# No randomization
no_transform = iaa.Sequential([])

# New domain randomization
img_transform_new = iaa.Sequential([
    iaa.flip.Flipud(0.5),
    iaa.flip.Fliplr(0.5),
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
        return normalize(G).double()
    return G.double()

def vis_gauss(img, gaussians):
    gaussians = gaussians.cpu().numpy().transpose(1, 2, 0)
    img = img.cpu().numpy().transpose(1, 2, 0) * 255.0
    # repeat last dimension 3 times
    gaussians = np.tile(gaussians, (1, 1, 3))
    h1 = gaussians
    output = cv2.normalize(h1, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite('test-gaussians.png', output)
    cv2.imwrite('test-img.png', img)

def bimodal_gauss(G1, G2, normalize=False):
    bimodal = torch.max(G1, G2)
    if normalize:
        return normalize(bimodal)
    return bimodal

class KeypointsDataset(Dataset):
    def __init__(self, folder, img_height, img_width, transform, gauss_sigma=8, augment=True, crop=True, only_full=False, condition=False):
        self.img_height = img_height
        self.img_width = img_width
        self.gauss_sigma = gauss_sigma
        self.transform = transform
        self.img_transform = img_transform if augment else no_transform

        self.imgs = []
        self.labels = []
        self.new_versions = []

        labels_folder = os.path.join(folder, 'annots')
        img_folder = os.path.join(folder, 'images')

        self.crop = crop
        self.only_full = only_full
        self.condition = condition

        if not only_full:
            for i in range(len(os.listdir(labels_folder))):
                label = np.load(os.path.join(labels_folder, '%05d.npy'%i))
                if len(label) > 0:
                    label[:,0] = np.clip(label[:, 0], 0, self.img_width-1)
                    label[:,1] = np.clip(label[:, 1], 0, self.img_height-1)

                    self.imgs.append(os.path.join(img_folder, '%05d.png'%i))
                    self.labels.append(label)
                    self.new_versions.append(0)

        more_folder = os.path.join(folder, 'more_knot_crops')
        if os.path.exists(more_folder):
            for j in range(len(os.listdir(more_folder))):
                img_and_annot = np.load(os.path.join(more_folder, '%05d.npy'%j), allow_pickle=True).item()
                if len(img_and_annot['annots']) == 0:
                    continue
                self.imgs.append(os.path.join(more_folder, '%05d.npy'%j))
                self.labels.append(img_and_annot['annots'])
                self.new_versions.append(1)

    def __getitem__(self, index):
        new_version = self.new_versions[index]
        keypoints = self.labels[index]
        condition_with_cable_cropped = torch.zeros(self.img_height, self.img_width)

        if not new_version:
            img = cv2.imread(self.imgs[index])
            kpts = KeypointsOnImage.from_xy_array(keypoints, shape=img.shape)
            img, labels = self.img_transform(image=img, keypoints=kpts)
            img = img[:, :, 0:3].copy() #ignore depth in the fourth channel
            img = self.transform(img).cuda()
            # plt.imshow(img.cpu().numpy().transpose(1, 2, 0))
            # plt.show()
            labels_np = []
            for l in labels:
                labels_np.append([l.x,l.y])
            labels = torch.from_numpy(np.array(labels_np, dtype=np.int32)).cuda()
            U = labels[0:,0]
            V = labels[0:,1]

            combined = torch.cat((img.cuda().double(),), dim=0).float()
        else:
            loaded_data = np.load(self.imgs[index], allow_pickle=True).item()
            # print(list(loaded_data.keys()))
            img = loaded_data['img']
            # keypoints = loaded_data['annots']
            starting_idx = np.random.randint(0, len(keypoints)/8)*8
            bbox_corners = keypoints[starting_idx: starting_idx+2]
            condition_part = keypoints[starting_idx+2: starting_idx+4]
            pull1 = keypoints[starting_idx+4: starting_idx+6]
            pull2 = keypoints[starting_idx+6: starting_idx+8]

            # fill in rope between bbox corners
            masked_img = np.where(img > 100, img, 0)

            pull_mask = np.zeros(img.shape)
            pull_mask[min(pull1[0][1], pull1[1][1]):max(pull1[0][1], pull1[1][1]), min(pull1[0][0], pull1[1][0]):max(pull1[0][0], pull1[1][0])] = 1
            pull_mask[min(pull2[0][1], pull2[1][1]):max(pull2[0][1], pull2[1][1]), min(pull2[0][0], pull2[1][0]):max(pull2[0][0], pull2[1][0])] = 1
            pull_with_cable = np.where(pull_mask > 0, masked_img, 0)
            # cv2.imwrite('pull_mask.png', pull_mask*255.0)

            condition_mask = np.zeros(img.shape)
            condition_mask[min(condition_part[0][1], condition_part[1][1]):max(condition_part[0][1], condition_part[1][1]), min(condition_part[0][0], condition_part[1][0]):max(condition_part[0][0], condition_part[1][0])] = 1
            condition_with_cable = np.where(condition_mask > 0, masked_img, 0)
            
            # add opposite corners to bbox_corners
            bbox_corners_opp = np.array([[bbox_corners[0][0], bbox_corners[1][1]], [bbox_corners[1][0], bbox_corners[0][1]]])
            bbox_corners = np.concatenate((bbox_corners, bbox_corners_opp), axis=0)
            # print(bbox_corners, img.shape)
            bbox_corner_kpts = KeypointsOnImage.from_xy_array(bbox_corners, shape=img.shape)
            pull_with_cable_and_masked_img, bbox_corners = self.img_transform(image=np.concatenate((pull_with_cable, masked_img, condition_with_cable), axis=2), keypoints=bbox_corner_kpts)
            
            # split into img and mask again
            pull_with_cable = pull_with_cable_and_masked_img[:, :, 0:pull_with_cable.shape[2]].copy()
            masked_img = pull_with_cable_and_masked_img[:, :, 3:6].copy()
            condition_with_cable = pull_with_cable_and_masked_img[:, :, 6:9].copy()
            # print(pull_with_cable.shape, masked_img.shape)

            if self.crop:
                random_padding = np.random.randint(0, 50, size=(4,))
                minx = max(int(min(bbox_corners.keypoints[0].x, bbox_corners.keypoints[1].x, bbox_corners.keypoints[2].x, bbox_corners.keypoints[3].x)) - random_padding[0], 0)
                miny = max(int(min(bbox_corners.keypoints[0].y, bbox_corners.keypoints[1].y, bbox_corners.keypoints[2].y, bbox_corners.keypoints[3].y)) - random_padding[1], 0)
                maxx = min(int(max(bbox_corners.keypoints[0].x, bbox_corners.keypoints[1].x, bbox_corners.keypoints[2].x, bbox_corners.keypoints[3].x)) + random_padding[2], img.shape[1]-1)
                maxy = min(int(max(bbox_corners.keypoints[0].y, bbox_corners.keypoints[1].y, bbox_corners.keypoints[2].y, bbox_corners.keypoints[3].y)) + random_padding[3], img.shape[0]-1)

                # cv2.imwrite('masked_img.png', masked_img)
                # cv2.imwrite('pull_with_cable.png', pull_with_cable)
                # print(minx, miny, maxx, maxy)

                combined = masked_img[miny:maxy, minx:maxx]
                pull_with_cable_cropped = pull_with_cable[miny:maxy, minx:maxx]
                condition_with_cable_cropped = condition_with_cable[miny:maxy, minx:maxx] * np.random.randint(0, 2) # on or off condition

            # cv2.imwrite('combined.png', combined)
            # cv2.imwrite('pull_with_cable_cropped.png', pull_with_cable_cropped)

            combined = cv2.resize(combined, (self.img_width, self.img_height))
            combined = self.transform(combined).cuda().float()
            pull_with_cable_cropped = cv2.resize(pull_with_cable_cropped, (self.img_width, self.img_height))
            condition_with_cable_cropped = cv2.resize(condition_with_cable_cropped, (self.img_width, self.img_height))

            V, U = np.nonzero(pull_with_cable_cropped[:, :, 0])
            U, V = torch.from_numpy(np.array([U, V], dtype=np.int32)).cuda()

        if self.condition and condition_with_cable_cropped.sum() > 0:
            cond_V, cond_U = np.nonzero(condition_with_cable_cropped[:, :, 0])
            cond_U, cond_V = torch.from_numpy(np.array([cond_U, cond_V], dtype=np.int32)).cuda()
            gaussians = gauss_2d_batch(self.img_width, self.img_height, self.gauss_sigma, cond_U, cond_V)
            mm_gauss = gaussians[0]
            for i in range(1, len(gaussians)):
                mm_gauss = bimodal_gauss(mm_gauss, gaussians[i])
            mm_gauss.unsqueeze_(0)
            # print(gaussians.shape)
            combined[0] = mm_gauss
        else:
            combined[0] = 0

        gaussians = gauss_2d_batch(self.img_width, self.img_height, self.gauss_sigma, U, V)
        mm_gauss = gaussians[0]
        for i in range(1, len(gaussians)):
            mm_gauss = bimodal_gauss(mm_gauss, gaussians[i])
        mm_gauss.unsqueeze_(0)

        return combined, mm_gauss
    
    def __len__(self):
        return len(self.labels)

TEST_DIR = "hulkL_seg"
if __name__ == '__main__':
    IMG_WIDTH = 200
    IMG_HEIGHT = 200
    GAUSS_SIGMA = 8
    test_dataset = KeypointsDataset('/host/%s/train'%TEST_DIR,
                           IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA, condition=True)
    img, gaussians = test_dataset[-1] #[-1] #
    vis_gauss(img, gaussians)
 

