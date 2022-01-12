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

os.environ["CUDA_VISIBLE_DEVICES"]="1"

print("running")

model_ckpt = "loop_detectortrain_test_loss/model_2_1_164_1.3544122673336825_0.9428386548464311.pth" #"loop_detector/model_2_1_114_0.8913466039467413.pth"

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

dataset_dir = 'loop_detector'
test_dataset = KeypointsDataset('train_sets/%s/test/images'%dataset_dir,
                           'train_sets/%s/test/annots'%dataset_dir, IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA)
test_data = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

for i, f in enumerate(test_data):
    img_t = f[0]
    # GAUSS
    heatmap = prediction.predict(img_t)
    heatmap = heatmap.detach().cpu().numpy()
    prediction.plot(img_t.detach().cpu().numpy(), heatmap, image_id=i)
 
