import pickle
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from config import *
from src.model import ClassificationModel
from src.dataset import TEST_DIR, KeypointsDataset, transform
import matplotlib.pyplot as plt
MSE = torch.nn.MSELoss()
bceLoss = nn.BCELoss

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def forward(sample_batched, model):
    img, gt_gauss = sample_batched
    img = Variable(img.cuda() if use_cuda else img)
    pred_gauss = model.forward(img).double()
    #pred_gauss = pred_gauss.view(pred_gauss.shape[0], 4, 640*480).double()
    #gt_gauss += 1e-300
    #loss = F.kl_div(gt_gauss.cuda().log(), pred_gauss, None, None, 'mean')
    loss = nn.BCELoss()(pred_gauss, gt_gauss)
    return loss

def fit(train_data, test_data, model, epochs, checkpoint_path = ''):
    train_epochs = []
    test_epochs = []
    test_losses = []
    train_losses = []
    for epoch in range(epochs):
        train_loss = 0.0
        for i_batch, sample_batched in enumerate(train_data):
            optimizer.zero_grad()
            loss = forward(sample_batched, model)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i_batch + 1, loss.item()), end='')
            print('\r', end='')
        print('train loss:', train_loss / i_batch)
        train_epochs.append(epoch)
        train_losses.append(train_loss / i_batch)

        if epoch % 10 == 9:
            test_loss = 0.0
            for i_batch, sample_batched in enumerate(test_data):
                loss = forward(sample_batched, model)
                test_loss += loss.item()
            print('test loss:', test_loss / i_batch)
            test_epochs.append(epoch)
            test_losses.append(test_loss)

            np.save(f"logs/losses_{expt_name}.npy", test_losses)
            np.save(f"logs/train_losses_{expt_name}.npy", train_losses)
            plt.clf()
            plt.title(f"losses {expt_name}")
            plt.plot(test_epochs, test_losses, label='test loss')
            plt.plot(train_epochs, train_losses, label='train loss')
            plt.legend()
            plt.savefig(f"logs/losses_{expt_name}_graph.png")
        if epoch%10 == 9:
            torch.save(keypoints.state_dict(), checkpoint_path + '/model_2_1_' + str(epoch) + '_' + str(test_loss) + '.pth')

# dataset
workers=0
dataset_dir ='processed_sim_data/under_over_crossings_dataset'
expt_name = 'over_under_model_1'
output_dir = 'checkpoints'
save_dir = os.path.join(output_dir, expt_name)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# TEST_DIR = 'hulkL_seg'
train_dataset = KeypointsDataset(['/host/%s/train'%dataset_dir],
                           IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA, condition=True, only_full=True, sim=False, trace_imgs=True)
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

test_dataset = KeypointsDataset('/host/%s/test'%dataset_dir,
                           IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA, condition=True, only_full=True, sim=False, trace_imgs=True)
test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

use_cuda = torch.cuda.is_available()

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
if use_cuda:
    torch.cuda.set_device(0)

# model
keypoints = ClassificationModel(num_classes=1, img_height=IMG_HEIGHT, img_width=IMG_WIDTH).cuda()

# optimizer
optimizer = optim.Adam(keypoints.parameters(), lr=1.0e-5, weight_decay=1.0e-4)
#optimizer = optim.Adam(keypoints.parameters(), lr=0.0001)

fit(train_data, test_data, keypoints, epochs=epochs, checkpoint_path=save_dir)
