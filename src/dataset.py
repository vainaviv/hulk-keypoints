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
from config import *

# set torch GPU to 3
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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
    # gaussians = np.tile(gaussians, (1, 1, 1))
    gaussians = np.concatenate((gaussians, np.zeros_like(gaussians[:, :, :1])), axis=2)
    h1 = gaussians
    output = cv2.normalize(h1, None, 0, 255, cv2.NORM_MINMAX)
    print("writing images")
    cv2.imwrite('test-gaussians.png', output)
    cv2.imwrite('test-img.png', img)

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
    def __init__(self, folder, img_height, img_width, transform, gauss_sigma=8, augment=True, crop=True, only_full=False, condition=False, sim=False):
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

        for extra_folder in ['more_knot_crops', 'hulkL_detectron_fail_anal_new_aug21']:
            more_folder = os.path.join(folder, extra_folder)
            if os.path.exists(more_folder):
                for fname in os.listdir(more_folder):
                    img_and_annot = np.load(os.path.join(more_folder, fname), allow_pickle=True).item()
                    if len(img_and_annot['annots']) == 0 or len(img_and_annot['annots']) % 8 != 0:
                        continue
                    self.imgs.append(os.path.join(more_folder, fname))
                    self.labels.append(img_and_annot['annots'])
                    self.new_versions.append(1)

        def get_box_around_point(pt, size=2):
            return [[pt[0]-size, pt[1]-size], [pt[0]+size, pt[1]+size]]

        if sim:
            sim_folder = os.path.join(folder, 'sim')
            if os.path.exists(sim_folder):
                for fname in os.listdir(sim_folder):
                    img_and_annot = np.load(os.path.join(sim_folder, fname), allow_pickle=True).item()
                    constructed_annots = [[0, 0], [IMG_HEIGHT - 1, IMG_WIDTH - 1]]
                    pixels = img_and_annot['pixels']
                    cond_point = get_box_around_point(pixels[img_and_annot['condition'][0]][0])
                    pull_point = get_box_around_point(pixels[img_and_annot['condition'][1]][0])
                    hold_point = get_box_around_point(pixels[img_and_annot['condition'][2]][0])
                    constructed_annots += cond_point
                    constructed_annots += pull_point
                    constructed_annots += hold_point
                    if np.min(constructed_annots) < 0 or np.max(constructed_annots) >= self.img_width:
                        continue

                    self.imgs.append(os.path.join(sim_folder, fname))
                    # plt.imshow(img_and_annot['img'])
                    # plt.savefig('test_load_img.png')
                    self.labels.append(constructed_annots)
                    self.new_versions.append(1)

        print('Loaded %d images'%len(self.imgs))

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
            mm_gauss = get_gauss(self.img_width, self.img_height, self.gauss_sigma, U, V)
        else:
            loaded_data = np.load(self.imgs[index], allow_pickle=True).item()
            img = loaded_data['img'][:, :, :3]
            # plt.imshow(img)
            # plt.savefig('img.png')
            starting_idx = np.random.randint(0, len(keypoints)/8)*8
            bbox_corners = keypoints[starting_idx: starting_idx+2]
            condition_part = keypoints[starting_idx+2: starting_idx+4]
            pull1 = keypoints[starting_idx+4: starting_idx+6]
            pull2 = keypoints[starting_idx+6: starting_idx+8]

            # fill in rope between bbox corners
            # print(img.max(), img.min())
            if img.max() <= 1.0:
                # sim images
                img = (img * 255.0).astype(np.uint8)
                masked_img = img
            else:
                # pass
                masked_img = img.copy()
            img_mask = np.where(masked_img > 100, 255.0, 0.0)
            # plt.clf()
            # plt.imshow(masked_img)
            # plt.savefig('masked_img.png')

            pull_mask1 = np.zeros(img.shape)
            pull_mask1[min(pull1[0][1], pull1[1][1]):max(pull1[0][1], pull1[1][1]), min(pull1[0][0], pull1[1][0]):max(pull1[0][0], pull1[1][0])] = 1
            # plt.clf()
            # plt.imshow(pull_mask1)
            # plt.savefig('pull_mask1.png')
            
            pull_mask2 = np.zeros(img.shape)
            pull_mask2[min(pull2[0][1], pull2[1][1]):max(pull2[0][1], pull2[1][1]), min(pull2[0][0], pull2[1][0]):max(pull2[0][0], pull2[1][0])] = 1
            # plt.clf()
            # plt.imshow(pull_mask2)
            # plt.savefig('pull_mask2.png')

            pull_with_cable1 = np.where(pull_mask1 > 0, img_mask, 0)
            pull_with_cable2 = np.where(pull_mask2 > 0, img_mask, 0)
            # cv2.imwrite('pull_mask.png', pull_mask*255.0)

            condition_mask = np.zeros(img.shape)
            condition_mask[min(condition_part[0][1], condition_part[1][1]):max(condition_part[0][1], condition_part[1][1]), min(condition_part[0][0], condition_part[1][0]):max(condition_part[0][0], condition_part[1][0])] = 1
            # plt.clf()
            # plt.imshow(condition_mask)
            # plt.savefig("condition_mask.png")
            # print(condition_mask.min(), condition_mask.max())
            # print(condition_mask.shape)
            # plt.imsave('condition_mask.png', condition_mask)
            condition_with_cable = np.where(condition_mask > 0, img_mask, 0)

            # add opposite corners to bbox_corners
            bbox_corners_opp = np.array([[bbox_corners[0][0], bbox_corners[1][1]], [bbox_corners[1][0], bbox_corners[0][1]]])
            bbox_corners = np.concatenate((bbox_corners, bbox_corners_opp), axis=0)
            # print(bbox_corners, img.shape)
            bbox_corner_kpts = KeypointsOnImage.from_xy_array(bbox_corners, shape=img.shape)
            pull_with_cable_and_masked_img, bbox_corners = self.img_transform(image=np.concatenate((pull_with_cable1, pull_with_cable2, masked_img, condition_with_cable), axis=2), keypoints=bbox_corner_kpts)
            
            # split into img and mask again
            pull_with_cable1 = pull_with_cable_and_masked_img[:, :, 0:pull_with_cable1.shape[2]].copy()
            pull_with_cable2 = pull_with_cable_and_masked_img[:, :, pull_with_cable1.shape[2]:pull_with_cable1.shape[2]*2].copy()
            masked_img = pull_with_cable_and_masked_img[:, :, 6:9].copy()
            condition_with_cable = pull_with_cable_and_masked_img[:, :, 9:12].copy()

            # plt.imshow(condition_with_cable)
            # plt.savefig('condition_with_cable.png')

            if self.crop:
                random_padding = np.random.randint(1, 80, size=(4,)) # used to be 50
                minx = max(int(min(bbox_corners.keypoints[0].x, bbox_corners.keypoints[1].x, bbox_corners.keypoints[2].x, bbox_corners.keypoints[3].x)) - random_padding[0], 0)
                miny = max(int(min(bbox_corners.keypoints[0].y, bbox_corners.keypoints[1].y, bbox_corners.keypoints[2].y, bbox_corners.keypoints[3].y)) - random_padding[1], 0)
                maxx = min(int(max(bbox_corners.keypoints[0].x, bbox_corners.keypoints[1].x, bbox_corners.keypoints[2].x, bbox_corners.keypoints[3].x)) + random_padding[2], img.shape[1]-1)
                maxy = min(int(max(bbox_corners.keypoints[0].y, bbox_corners.keypoints[1].y, bbox_corners.keypoints[2].y, bbox_corners.keypoints[3].y)) + random_padding[3], img.shape[0]-1)

                # cv2.imwrite('masked_img.png', masked_img)
                # cv2.imwrite('pull_with_cable.png', pull_with_cable)
                # print(minx, miny, maxx, maxy)

                # print("SIZE WARNING", maxx, minx, maxy, miny)
                if maxx-minx <= 0 or maxy-miny <= 0:
                    print("SIZE WARNING", index, maxx, minx, maxy, miny, random_padding, bbox_corners.keypoints)
                    return self.__getitem__(index)
                combined = masked_img[miny:maxy, minx:maxx]
                pull_with_cable_cropped1 = pull_with_cable1[miny:maxy, minx:maxx]
                pull_with_cable_cropped2 = pull_with_cable2[miny:maxy, minx:maxx]
                condition_with_cable_cropped = condition_with_cable[miny:maxy, minx:maxx] * 1 #np.random.randint(0, 2) # on or off condition

            # plt.imshow(condition_with_cable_cropped)
            # plt.savefig('condition_with_cable_cropped.png')

            # cv2.imwrite('combined.png', combined)
            # cv2.imwrite('pull_with_cable_cropped.png', pull_with_cable_cropped)

            print(combined.shape, self.img_width, self.img_height)
            combined = cv2.resize(combined, (self.img_width, self.img_height))
            combined = self.transform(combined).cuda().float()
            pull_with_cable_cropped1 = cv2.resize(pull_with_cable_cropped1, (self.img_width, self.img_height))
            pull_with_cable_cropped2 = cv2.resize(pull_with_cable_cropped2, (self.img_width, self.img_height))
            condition_with_cable_cropped = cv2.resize(condition_with_cable_cropped, (self.img_width, self.img_height))

            V, U = np.nonzero(pull_with_cable_cropped1[:, :, 0])
            U, V = torch.from_numpy(np.array([U, V], dtype=np.int32)).cuda()

            mm_gauss_1 = get_gauss(self.img_width, self.img_height, self.gauss_sigma, U, V)

            V, U = np.nonzero(pull_with_cable_cropped2[:, :, 0])
            U, V = torch.from_numpy(np.array([U, V], dtype=np.int32)).cuda()

            mm_gauss_2 = get_gauss(self.img_width, self.img_height, self.gauss_sigma, U, V)

            # now concat the gausses along first axis
            mm_gauss = torch.cat((mm_gauss_1, mm_gauss_2), dim=0)

        if self.condition and condition_with_cable_cropped.sum() > 0:
            cond_V, cond_U = np.nonzero(condition_with_cable_cropped[:, :, 0])
            cond_U, cond_V = torch.from_numpy(np.array([cond_U, cond_V], dtype=np.int32)).cuda()
            combined[0] = get_gauss(self.img_width, self.img_height, self.gauss_sigma, cond_U, cond_V)
        else:
            combined[0] = 0

        return combined, mm_gauss
    
    def __len__(self):
        return len(self.labels)

TEST_DIR = "hulkL_seg"
if __name__ == '__main__':
    IMG_WIDTH = 200
    IMG_HEIGHT = 200
    GAUSS_SIGMA = 8
    test_dataset = KeypointsDataset('/host/%s/train'%TEST_DIR,
                           IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA, only_full=True, condition=True, sim=False)
    img, gaussians = test_dataset[-3] #[-1] #
    vis_gauss(img, gaussians)
 

