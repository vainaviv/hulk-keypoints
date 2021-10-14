import pickle
import cv2
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from config import *
from src.model import KeypointsGauss
from src.dataset import KeypointsDataset, transform, gauss_2d_batch
from src.prediction import Prediction
from datetime import datetime
from PIL import Image
import numpy as np

model_ckpt = "cond_loop_detection/model_2_1_24.pth"

def crop_around(img, pt, size=90):
    result = img[pt[1] - size//2: pt[1] + size//2, pt[0] - size//2: pt[0] + size//2]
    # pad result to 90 by 90
    result = Image.fromarray(result)
    result = result.resize((size, size))
    return np.array(result)

def model_heatmap(img, heatmap):
    pass

# model
keypoints = KeypointsGauss(1, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, channels=4).cuda()
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

input_image_name = 'test_input.png'
start_point = [243, 174]

start_img = 

gauss_2d_batch()