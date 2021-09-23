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

# No domain randomization
transform = transforms.Compose([transforms.ToTensor()])

# Domain randomization
#transform = transforms.Compose([
#    iaa.Sequential([
#        iaa.AddToHueAndSaturation((-20, 20)),
#        iaa.LinearContrast((0.85, 1.2), per_channel=0.25), 
#        iaa.Add((-10, 30), per_channel=True),
#        iaa.GammaContrast((0.85, 1.2)),
#        iaa.GaussianBlur(sigma=(0.0, 0.6)),
#        iaa.ChangeColorTemperature((5000,35000)),
#        iaa.MultiplySaturation((0.95, 1.05)),
#        iaa.AdditiveGaussianNoise(scale=(0, 0.0125*255)),
#    ], random_order=True).augment_image,
#    transforms.ToTensor()
#])

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
    h1,h2,h3,h4 = gaussians
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

        self.imgs = []
        self.labels = []
        for i in range(len(os.listdir(labels_folder))-1):
            label = np.load(os.path.join(labels_folder, '%05d.npy'%i))
            if len(label) > 0:
                label[:,0] = np.clip(label[:, 0], 0, self.img_width-1)
                label[:,1] = np.clip(label[:, 1], 0, self.img_height-1)
                self.imgs.append(os.path.join(img_folder, '%05d.png'%i))
                self.labels.append(torch.from_numpy(label).cuda())

    def __getitem__(self, index):  
        img = self.transform(cv2.imread(self.imgs[index]))
        labels = self.labels[index]
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
    NUM_KEYPOINTS = 4
    IMG_WIDTH = 640
    IMG_HEIGHT = 480
    GAUSS_SIGMA = 10
    TEST_DIR = ""
    test_dataset = KeypointsDataset('/host/data/%s/test/images'%TEST_DIR,
                           '/host/data/%s/test/keypoints'%TEST_DIR, NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA)
    img, gaussians = test_dataset[0]
    vis_gauss(gaussians)
 
