import pickle
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from config import *
# from src.model import Model
from src.dataset import KeypointsDataset, transform
# from src.resnet import resnet34
import torchvision.models as models

raid_dir = 'train_sets'
dir_name = 'knot_or_not_thresh'
dataset_dir = raid_dir + '/' + dir_name

model = models.resnet18()
model.fc=nn.Sequential(nn.Dropout(p=.0),nn.Linear(512,2),nn.Softmax())
model.load_state_dict(torch.load('model_justin/checkpoint_500.pth'))
model=model.cuda()
model.eval()

test_dataset = KeypointsDataset('%s/test'%dataset_dir, NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA,do_aug=False)
print(test_dataset.__len__())
test_data = DataLoader(test_dataset, batch_size=1, shuffle=True)
tot=0
corr=0
with torch.no_grad():
    for i_batch, sample_batched in enumerate(test_data):
        tot+=1
        img, gt_label = sample_batched
        img = Variable(img.cuda())
        pred_label = model(img)
        cl=torch.argmax(pred_label).cpu()
        if(gt_label.squeeze()[cl]==1):
            corr+=1
        print(pred_label,gt_label)
print(corr/tot)