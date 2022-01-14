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
    #iaa.GammaContrast((0.95, 1.05)),
    #iaa.GaussianBlur(sigma=(0.0, 0.6)),
    #iaa.MultiplySaturation((0.95, 1.05)),
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

def gauss_2d_batch(width, height, sigma, U, V, normalize_dist=False):
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
    h1,h2,h3,h4 = gaussians
    output = cv2.normalize(h1, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite('test.png', output)

class KeypointsDataset(Dataset):
    def __init__(self, img_folder, num_keypoints, img_height, img_width, transform, gauss_sigma=8):
        self.num_keypoints = num_keypoints
        self.img_height = img_height
        self.img_width = img_width
        self.gauss_sigma = gauss_sigma
        self.transform = img_transform

        self.imgs = []
        self.labels = []
        for folder in os.listdir(img_folder):
            label = folder 
            for img in os.listdir(os.path.join(img_folder, folder)):
                self.imgs.append(os.path.join(os.path.join(img_folder, folder), img))
                if label == "knot":
                    self.labels.append(torch.from_numpy(np.array([1, 0, 0])).cuda())
                elif label == "endpoint":
                    self.labels.append(torch.from_numpy(np.array([0, 1, 0])).cuda())
                else:
                    self.labels.append(torch.from_numpy(np.array([0, 0, 1])).cuda())

    def __getitem__(self, index):
        img_load = np.load(self.imgs[index])
        img = self.transform(image=img_load).copy()
        labels = self.labels[index]
        return torch.as_tensor(img).cuda(), labels
    
    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    NUM_KEYPOINTS = 4
    IMG_WIDTH = 640
    IMG_HEIGHT = 480
    GAUSS_SIGMA = 10
    test_dataset = KeypointsDataset('/host/data/undo_reid_term/train/images',
                           '/host/data/undo_reid_term/train/actions', NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA)
    img, gaussians = test_dataset[0]
    vis_gauss(gaussians)
