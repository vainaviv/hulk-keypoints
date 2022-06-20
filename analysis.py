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

model_ckpt = "corresponding_segment_r50/model_2_1_70_3.6889334914035885_1.2756286068116467.pth"

os.environ["CUDA_VISIBLE_DEVICES"]="1"

# model
keypoints = KeypointsGauss(1, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, channels=4, attention=True).cuda()
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

dataset_dir = 'corresponding_segment_r50'
test_dataset = KeypointsDataset('train_sets/%s/test/images'%dataset_dir,
                           'train_sets/%s/test/annots'%dataset_dir, IMG_HEIGHT, IMG_WIDTH, gauss_sigma=GAUSS_SIGMA, augment=False)
test_data = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

total_error = 0
for i, f in enumerate(test_data):
    img_t = f[0]
    ground_truth = (f[1]).squeeze().detach().cpu().numpy()
    # GAUSS
    heatmap = prediction.predict(img_t)
    heatmap = heatmap.detach().cpu().numpy()
    total_error += prediction.plot(img_t.detach().cpu().numpy(), heatmap, ground_truth, image_id=i)

print(total_error/len(test_data))
