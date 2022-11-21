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
from src.model import KeypointsGauss, ClassificationModel
from src.dataset import KeypointsDataset, transform
import matplotlib.pyplot as plt
import argparse

os.environ["CUDA_VISIBLE_DEVICES"]="2"

# parse command line flags
parser = argparse.ArgumentParser()
parser.add_argument('--expt_name', type=str, default='default')
parser.add_argument('--expt_type', type=str, default='trp')

flags = parser.parse_args()

experiment_time = time.strftime("%Y%m%d-%H%M%S")
expt_name = flags.expt_name
expt_type = flags.expt_type

if expt_type not in ALLOWED_EXPT_TYPES:
    raise ValueError(f"expt_type must be one of {ALLOWED_EXPT_TYPES}")

def forward(sample_batched, model):
    img, gt_gauss = sample_batched
    img = Variable(img.cuda() if use_cuda else img)
    pred_gauss = model.forward(img).double()
    loss = nn.BCELoss()(pred_gauss.squeeze(), gt_gauss.squeeze())
    return loss

def fit(train_data, test_data, model, epochs, optimizer, checkpoint_path = ''):
    train_epochs = []
    test_epochs = []
    test_losses = []
    train_losses = []
    last_checkpoint_epoch = -1
    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0.0
        num_iters = len(train_data) / batch_size
        for i_batch, sample_batched in enumerate(train_data):
            optimizer.zero_grad()
            loss = forward(sample_batched, model)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i_batch + 1, loss.item()), end='') #,f"\t epoch est. time. left {((time.time() - start_time) * (num_iters) / (i_batch + 1)) * (epochs - epoch)}", end='')
            print('\r', end='')
        print('train loss:', train_loss / (i_batch + 1))
        train_epochs.append(epoch)
        train_losses.append(train_loss / (i_batch + 1))

        if epoch % EVAL_CHECKPT_FREQ == (EVAL_CHECKPT_FREQ - 1):
            test_loss = 0.0
            for i_batch, sample_batched in enumerate(test_data):
                loss = forward(sample_batched, model)
                test_loss += loss.item()
            test_loss_per_batch = test_loss / (i_batch + 1)
            print('test loss:', test_loss_per_batch)
            test_epochs.append(epoch)
            test_losses.append(test_loss_per_batch)

            np.save(f"logs/test_losses_{expt_name}.npy", test_losses)
            np.save(f"logs/train_losses_{expt_name}.npy", train_losses)
            plt.clf()
            plt.title(f"losses {expt_name}")
            plt.plot(test_epochs, test_losses, label='test loss')
            plt.plot(train_epochs, train_losses, label='train loss')
            plt.legend()
            plt.savefig(f"logs/losses_{expt_name}_graph.png")

            if len(test_losses) <= 1 or test_loss_per_batch < np.min(test_losses[:-1]) or epoch - last_checkpoint_epoch >= MIN_CHECKPOINT_FREQ:
                torch.save(keypoints.state_dict(), os.path.join(checkpoint_path, f'model_2_1_{epoch}_{test_loss_per_batch:.3f}.pth'))
                last_checkpoint_epoch = epoch

# dataset
workers=0
output_dir = 'checkpoints'
save_dir = os.path.join(output_dir, expt_name)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

train_dataset = KeypointsDataset(['%s/train'%get_dataset_dir(expt_type)],
                                IMG_HEIGHT(expt_type), 
                                IMG_WIDTH(expt_type), 
                                transform, 
                                gauss_sigma=GAUSS_SIGMA, 
                                augment=True, 
                                expt_type=expt_type, 
                                condition_len=CONDITION_LEN, 
                                pred_len=PRED_LEN,
                                crop_width=CROP_WIDTH, 
                                spacing=COND_POINT_DIST_PX)
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

test_dataset = KeypointsDataset('%s/test'%get_dataset_dir(expt_type),
                           IMG_HEIGHT(expt_type), IMG_WIDTH(expt_type), transform, gauss_sigma=GAUSS_SIGMA, augment=False, expt_type=expt_type, condition_len=CONDITION_LEN, crop_width=CROP_WIDTH, spacing=COND_POINT_DIST_PX)
test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)


use_cuda = torch.cuda.is_available()

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
if use_cuda:
    torch.cuda.set_device(0)

# model
if not is_point_pred(expt_type):
    keypoints = ClassificationModel(num_classes=1, img_height=IMG_HEIGHT, img_width=IMG_WIDTH).cuda()
else:
    keypoints = KeypointsGauss(num_keypoints=1, img_height=IMG_HEIGHT, img_width=IMG_WIDTH).cuda()

# optimizer
optimizer = optim.Adam(keypoints.parameters(), lr=1.0e-5, weight_decay=1.0e-4)

# save the config to a file
save_config_params(save_dir, expt_type=expt_type)
fit(train_data, test_data, keypoints, epochs=epochs, optimizer=optimizer, checkpoint_path=save_dir)
