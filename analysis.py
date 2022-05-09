import pickle
import cv2
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from config import *
from src.model import KeypointsGauss
from src.dataset import KeypointsDataset, transform
from src.prediction import Prediction
from datetime import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="1"

model_ckpt = "random_crops_pretrain/model_2_1_70_44.15676872485587_12.201170021092581.pth"

# model
keypoints = KeypointsGauss(NUM_KEYPOINTS, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, channels=3, attention=False).cuda()
keypoints.load_state_dict(torch.load('checkpoints/%s'%model_ckpt))

# cuda
use_cuda = torch.cuda.is_available()
#use_cuda = False
if use_cuda:
    torch.cuda.set_device(0)
    keypoints = keypoints.cuda()

prediction = Prediction(keypoints, 1, IMG_HEIGHT, IMG_WIDTH, use_cuda)
transform = transform = transforms.Compose([
    transforms.ToTensor()
])

dataset_dir = 'random_crops'
test_dataset = KeypointsDataset('train_sets/%s/test/images'%dataset_dir,
                           'train_sets/%s/test/annots'%dataset_dir, IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA, augment=False, pretrain=True)
test_data = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

total_error = 0
for i, f in enumerate(test_data):
    img_t = f[0]
    ground_truth = (f[1]).squeeze().detach().cpu().numpy().transpose((1,2,0))
    plt.imsave('preds/out%04d_gt.png'%i, ground_truth)
    # GAUSS
    reconstruction = prediction.predict(img_t)
    reconstruction = reconstruction.squeeze().detach().cpu().numpy().transpose((1,2,0))
    plt.imsave('preds/out%04d.png'%i, reconstruction)
    # total_error += prediction.plot(img_t.detach().cpu().numpy(), heatmap, ground_truth, image_id=i)

# print(total_error/len(test_data))
 
