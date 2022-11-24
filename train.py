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
from config import ALL_EXPERIMENTS_CONFIG, get_dataset_dir, is_point_pred, save_config_params
from src.model import KeypointsGauss, ClassificationModel
from src.dataset import KeypointsDataset, transform
import matplotlib.pyplot as plt
import argparse

os.environ["CUDA_VISIBLE_DEVICES"]="2"

# parse command line flags
parser = argparse.ArgumentParser()
parser.add_argument('--expt_name', type=str, default='')
parser.add_argument('--expt_class', type=str, default='trp')

flags = parser.parse_args()

# get time in PST
experiment_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
expt_name = flags.expt_name
expt_class = flags.expt_class

if expt_class not in ALL_EXPERIMENTS_CONFIG:
    raise ValueError(f"expt_class must be one of {list(ALL_EXPERIMENTS_CONFIG.keys())}")

config = ALL_EXPERIMENTS_CONFIG[expt_class]()

if expt_name == '':
    expt_name = f"{experiment_time}_{expt_class}"

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
        num_iters = len(train_data) / config.batch_size
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

        if epoch % config.eval_checkpoint_freq == (config.eval_checkpoint_freq - 1):
            test_loss = 0.0
            for i_batch, sample_batched in enumerate(test_data):
                loss = forward(sample_batched, model)
                test_loss += loss.item()
            test_loss_per_batch = test_loss / (i_batch + 1)
            print('test loss:', test_loss_per_batch)
            test_epochs.append(epoch)
            test_losses.append(test_loss_per_batch)

            np.save(os.path.join(checkpoint_path, f"test_losses_{expt_name}.npy"), test_losses)
            np.save(os.path.join(checkpoint_path, f"train_losses_{expt_name}.npy"), train_losses)

            if len(test_losses) <= 1 or test_loss_per_batch < np.min(test_losses[:-1]) or epoch - last_checkpoint_epoch >= config.min_checkpoint_freq:
                torch.save(keypoints.state_dict(), os.path.join(checkpoint_path, f'model_{epoch}_{test_loss_per_batch:.5f}.pth'))
                last_checkpoint_epoch = epoch

# dataset
workers=0
output_dir = 'checkpoints'
save_dir = os.path.join(output_dir, expt_name)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

train_dataset = KeypointsDataset(['%s/train'%get_dataset_dir(config.expt_type)],
                                config.img_height, 
                                config.img_width, 
                                transform, 
                                gauss_sigma=config.gauss_sigma, 
                                augment=True, 
                                expt_type=config.expt_type, 
                                condition_len=config.condition_len, 
                                pred_len=config.pred_len,
                                crop_width=config.crop_width, 
                                spacing=config.cond_point_dist_px)
train_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=workers)

test_dataset = KeypointsDataset('%s/test'%get_dataset_dir(config.expt_type),
                           config.img_height, config.img_width, transform, gauss_sigma=config.gauss_sigma, augment=False, expt_type=config.expt_type, condition_len=config.condition_len, crop_width=config.crop_width, spacing=config.cond_point_dist_px)
test_data = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=workers)


use_cuda = torch.cuda.is_available()

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
if use_cuda:
    torch.cuda.set_device(0)

# model
if not is_point_pred(config.expt_type):
    keypoints = ClassificationModel(num_classes=1, img_height=config.img_height, img_width=config.img_width).cuda()
else:
    keypoints = KeypointsGauss(num_keypoints=1, img_height=config.img_height, img_width=config.img_width, resnet_type=config.resnet_type, pretrained=config.pretrained).cuda()

# optimizer
optimizer = optim.Adam(keypoints.parameters(), lr=1.0e-5, weight_decay=1.0e-4)

# save the config to a file
save_config_params(save_dir, config)
fit(train_data, test_data, keypoints, epochs=config.epochs, optimizer=optimizer, checkpoint_path=save_dir)
