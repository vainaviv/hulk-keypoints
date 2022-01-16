import pickle
import cv2
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from config import *
from src.model import Model
#from src.model_multi_headed import KeypointsGauss
from src.dataset import KeypointsDataset, transform
from src.prediction import Prediction
from datetime import datetime
from PIL import Image
import numpy as np
import torchvision.models as models

os.environ["CUDA_VISIBLE_DEVICES"]="2"

# model
#keypoints = models.resnet34(pretrained=False, num_classes=1).cuda()
#keypoints = Model(NUM_KEYPOINTS, pretrained=False, num_classes=1).cuda() 
keypoints =  Model(NUM_KEYPOINTS, pretrained=False, channels=2, num_classes=2, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, dropout=False).cuda()

#keypoints = KeypointsGauss(NUM_KEYPOINTS, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
keypoints.load_state_dict(torch.load('checkpoints/detect_ep/model_2_1_74_0.3408295752513512_0.6872417581696136.pth'))

# cuda
use_cuda = torch.cuda.is_available()
#use_cuda = False
if use_cuda:
    torch.cuda.set_device(0)
    keypoints = keypoints.cuda()

prediction = Prediction(keypoints, NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, use_cuda)
transform = transform = transforms.Compose([
    transforms.ToTensor()
])

image_dir = 'train_sets/detect_ep/test'
os.mkdir('endpoint')
os.mkdir('not_endpoint')
idx = 0
for folder in sorted(os.listdir(image_dir)):
    image_folder = os.path.join(image_dir, folder)
    for i, f in enumerate(sorted(os.listdir(image_folder))):
        #print(f)
        img = np.load(os.path.join(image_folder, f))
        img[0,:,:] = img[0,:,:]/255
        #print(img.shape)
        #img_t = transform(img)
        img_t = torch.tensor(img).cuda()
        #print(img_t.shape)
        value = prediction.predict(img_t)
        value = value.detach().cpu().numpy()
        img = img*255
        prediction.sort(img, value, image_id=idx) #will need to edit this
        idx += 1
