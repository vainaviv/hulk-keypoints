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
import matplotlib.pyplot as plt

# No domain randomization
transform = transforms.Compose([transforms.ToTensor()])

augment = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    # # iaa.Affine(
    # #     rotate=(-40, 40),
    # #     shear=(-5, 5),
    # #     scale=(1.0, 1.1),
    # #     mode='constant'
    # # ),
    #iaa.TranslateX(px=(-2, 2)),
    #iaa.TranslateY(px=(-2, 2)),
])

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

def vis_gauss(image, gaussians):
    # pad gaussians with zeros on first axis until 3
    image = image.cpu().numpy()
    gaussians = gaussians.cpu().numpy()
    #print(gaussians.shape)
    h1 = gaussians #,h2,h3,h4 = gaussians
    #output = cv2.normalize(h1, None, 0, 255, cv2.NORM_MINMAX)
    h1 = np.transpose(h1, (1, 2, 0))[:, :, :3]
    h1 = cv2.merge([h1, h1, h1])[:, :, :3]
    plt.imsave('gaussian.png', h1)
    plt.imsave('image.png', np.transpose(image[:3], (1, 2, 0)))
    h2 = np.transpose(image[3:], (1, 2, 0))
    plt.imsave('given.png', cv2.merge([h2, h2, h2]))

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
            label = np.load(os.path.join(labels_folder, '%05d.npy'%i), allow_pickle=True)
            if len(label) > 2:
                #label[:,0] = np.clip(label[:, 0], 0, self.img_width-1)
                #label[:,1] = np.clip(label[:, 1], 0, self.img_height-1)
                self.imgs.append(os.path.join(img_folder, '%05d.png'%i))
                self.labels.append(label)

    def __getitem__(self, index):  
        index = random.randint(0, len(self.labels)-1)

        orig_img = cv2.imread(self.imgs[index])
        img = self.transform(cv2.imread(self.imgs[index]))
        labels = self.labels[index]

        new_labels = list(labels)
        # parse labels
        if 'e' in labels:
            # get last index of e
            e_idx = np.where(labels == 'e')[0][-1]
            # coin flip to decide whether to use first or second endpoint
            if random.random() > 0.5:
                new_labels = labels[:e_idx]
            else:
                new_labels = labels[e_idx+1:]
            
            #print(len(new_labels), len(labels))

        # eliminate all letters from new_labels
        new_labels = [x for x in new_labels if x not in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']]

        endpoint = new_labels[0]
        new_labels = new_labels[1:] 

        given = torch.from_numpy(np.array(endpoint)).cuda()
        given_gauss = gauss_2d_batch(self.img_width, self.img_height, self.gauss_sigma, given[0], given[1], single=True)
        given_gauss = torch.unsqueeze(given_gauss, 0).cuda()
        combined = torch.cat((img.cuda().double(), given_gauss), dim=0).float()
        
        combined_labels = None

        # visualize keypoints on image
        for i in range(len(new_labels)):
            label = new_labels[i]
            x, y = label[0], label[1]
            cv2.circle(orig_img, (int(x), int(y)), 3, (0, 0, 255), -1)
        cv2.imwrite('keypoints.png', orig_img)

        if len(new_labels) % 2:
            # something is wrong
            return self.__getitem__(index)

        for i in range(0, len(new_labels), 2):
            pt1, pt2 = torch.from_numpy(np.array(new_labels[i])).cuda(), torch.from_numpy(np.array(new_labels[i+1])).cuda()

            gauss1 = gauss_2d_batch(self.img_width, self.img_height, self.gauss_sigma, pt1[0], pt1[1], single=True)
            gauss2 = gauss_2d_batch(self.img_width, self.img_height, self.gauss_sigma, pt2[0], pt2[1], single=True)
            bmg = torch.unsqueeze(bimodal_gauss(gauss1, gauss2), 0)
            if combined_labels is None:
                combined_labels = bmg
            else:
                combined_labels = torch.cat((combined_labels, bmg), dim=0)

        return combined, combined_labels
    
    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    NUM_KEYPOINTS = 4
    IMG_WIDTH = 640
    IMG_HEIGHT = 480
    GAUSS_SIGMA = 10
    TEST_DIR = ""
    test_dataset = KeypointsDataset('../mm_kpts/images',
                           '../mm_kpts/annots', NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA)
    img, gaussians = test_dataset[0]
    vis_gauss(img, gaussians)
 
