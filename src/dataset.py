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

# No domain randomization
transform = transforms.Compose([transforms.ToTensor()])

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Domain randomization
img_transform = iaa.Sequential([
    iaa.LinearContrast((0.95, 1.05), per_channel=0.25), 
    iaa.Add((-10, 10), per_channel=False),
    iaa.GammaContrast((0.95, 1.05)),
    iaa.GaussianBlur(sigma=(0.0, 0.6)),
    iaa.MultiplySaturation((0.95, 1.05)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.0125*255)),
    iaa.flip.Flipud(0.5),
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

def vis_gauss(gaussians):
    gaussians = gaussians.cpu().numpy()
    h1 = gaussians
    output = cv2.normalize(h1, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite('test.png', output)

def bimodal_gauss(G1, G2, normalize=False):
    bimodal = torch.max(G1, G2)
    if normalize:
        return normalize(bimodal)
    return bimodal

class KeypointsDataset(Dataset):
    def __init__(self, img_folder, labels_folder, num_keypoints, img_height, img_width, transform, gauss_sigma=8):
        self.num_keypoints = num_keypoints
        self.img_height = img_height
        self.img_width = img_width
        self.gauss_sigma = gauss_sigma
        self.transform = transform
        self.img_transform = img_transform

        self.imgs = []
        self.labels = []
        for i in range(len(os.listdir(labels_folder))-2):
            label = np.load(os.path.join(labels_folder, '%05d.npy'%i))
            if len(label) > 0:
                label[:,0] = np.clip(label[:, 0], 0, self.img_width-1)
                label[:,1] = np.clip(label[:, 1], 0, self.img_height-1)
                self.imgs.append(os.path.join(img_folder, '%05d.png'%i))
                self.labels.append(label)

    def __getitem__(self, index):
        keypoints = self.labels[index]
        img = cv2.imread(self.imgs[index])
        kpts = KeypointsOnImage.from_xy_array(keypoints, shape=img.shape)
        img, labels = self.img_transform(image=img, keypoints=kpts)
        img = img.copy()
        img = self.transform(img).cuda()
        labels_np = []
        for l in labels:
            labels_np.append([l.x,l.y])
        labels = torch.from_numpy(np.array(labels_np, dtype=np.int32)).cuda()
        given = labels[0]
        given_gauss = gauss_2d_batch(self.img_width, self.img_height, self.gauss_sigma, given[0], given[1], single=True)
        given_gauss = torch.unsqueeze(given_gauss, 0).cuda()
        combined = torch.cat((img.cuda().double(), given_gauss), dim=0).float()
        U = labels[1:,0]
        V = labels[1:,1]
        gaussians = gauss_2d_batch(self.img_width, self.img_height, self.gauss_sigma, U, V)
        mm_gauss = gaussians[0]
        for i in range(1, len(gaussians)):
            mm_gauss = bimodal_gauss(mm_gauss, gaussians[i])
        mm_gauss.unsqueeze_(0)
        return combined, mm_gauss
    
    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    NUM_KEYPOINTS = 1
    IMG_WIDTH = 200
    IMG_HEIGHT = 200
    GAUSS_SIGMA = 8
    TEST_DIR = "hulkL_seg"
    test_dataset = KeypointsDataset('/host/train_sets/%s/test/images'%TEST_DIR,
                           '/host/train_sets/%s/test/annots'%TEST_DIR, NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA)
    img, gaussians = test_dataset[0]
    vis_gauss(gaussians)
 
