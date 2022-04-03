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

MSE = torch.nn.MSELoss()
bceLoss = nn.BCELoss()
CEloss = nn.CrossEntropyLoss()
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def forward(sample_batched, model):
    img, gt_label = sample_batched
    img = Variable(img.cuda() if use_cuda else img)
    pred_label = model(img)
    #pred_label = pred_label.view(pred_label.shape[0], 4, 640*480).double()
    # gt_label += 1e-300
    # gt_label = gt_label.double()
    # gt_label = torch.zeros(2)
    # gt_label[]
    #loss = F.kl_div(gt_gauss.cuda().log(), pred_gauss, None, None, 'mean')
    #loss = nn.BCELoss()(pred_label, gt_label)
    loss = bceLoss(pred_label, gt_label)
    return loss

def fit(train_data, test_data, model, epochs, checkpoint_path = ''):
    for epoch in range(epochs):
        print(epoch)
        train_loss = 0.0
        model.train()
        n_batch=0
        for i_batch, sample_batched in enumerate(train_data):
            optimizer.zero_grad()
            loss = forward(sample_batched, model)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i_batch + 1, loss.item()), end='')
            print('\r', end='')
            n_batch+=1
        print('train loss:', train_loss / n_batch)
        train_loss = train_loss/ i_batch
        
        test_loss = 0.0
        model.eval()
        n_batch=0
        for i_batch, sample_batched in enumerate(test_data):
            loss = forward(sample_batched, model)
            test_loss += loss.item()
            n_batch+=1
        print('test loss:', test_loss / n_batch)
        if epoch%10 == 0:
            torch.save(model.state_dict(), f'model_justin/checkpoint_{epoch}.pth')

# dataset
workers=0
raid_dir = 'train_sets'
dir_name = 'knot_or_not_thresh'
dataset_dir = raid_dir + '/' + dir_name
output_dir = 'checkpoints'
save_dir = os.path.join(output_dir, dir_name)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)

train_dataset = KeypointsDataset('%s/train'%dataset_dir, NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA)
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

test_dataset = KeypointsDataset('%s/test'%dataset_dir, NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA)
print(train_dataset.__len__())
test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

use_cuda = torch.cuda.is_available()

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
if use_cuda:
    torch.cuda.set_device(0)

# model
# model = Model(NUM_KEYPOINTS, pretrained=False, channels=2, num_classes=2, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, dropout=True).cuda()
model = models.resnet18()
model.fc=nn.Sequential(nn.Dropout(.9),nn.Linear(512,2),nn.Softmax())
model=model.cuda()
# optimizer
optimizer = optim.Adam(model.parameters(),weight_decay=1e-4)

fit(train_data, test_data, model, epochs=epochs, checkpoint_path=save_dir)
