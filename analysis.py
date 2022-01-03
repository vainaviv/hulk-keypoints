import pickle
import cv2
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from config import *
from src.model import KeypointsGauss
from src.dataset import KeypointsDataset, transform, gauss_2d_batch, bimodal_gauss
from src.prediction import Prediction
from datetime import datetime, time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

# WITHOUT COND
#model_ckpts = ["hulkL_aug_crop_1/model_2_1_498_1.154286119789344.pth", "hulkL_aug_crop_2/model_2_1_498_1.0253424582349897.pth", "hulkL_aug_crop_3/model_2_1_498_1.0670859171776241.pth"]

# WITH COND
model_ckpts = ["hulkL_aug_cond_1/model_2_1_498_0.9576036110338699.pth", "hulkL_aug_cond_2/model_2_1_498_1.1168840242554805.pth", "hulkL_aug_cond_3/model_2_1_498_1.0491497036272723.pth"]

# model
keypoints_models = []
for model_ckpt in model_ckpts:
    keypoints = KeypointsGauss(1, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, channels=3).cuda()
    keypoints.load_state_dict(torch.load('checkpoints/%s'%model_ckpt))
    keypoints_models.append(keypoints)

# cuda
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(0)
    for keypoints in keypoints_models:
        keypoints = keypoints.cuda()

predictions = []

for keypoints in keypoints_models:
    prediction = Prediction(keypoints, 1, IMG_HEIGHT, IMG_WIDTH, use_cuda)
    predictions.append(prediction)

transform = transform = transforms.Compose([
    transforms.ToTensor()
])

# test_dataset = KeypointsDataset('hulkL_seg/test/', IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA, augment=False, crop=False, only_full=True, condition=False)
# test_data = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

### OPTIONALLY LOAD CUSTOM IMAGE ###
custom_image = "/host/bounding_boxes/130520.png"
if custom_image:
    img = cv2.imread(custom_image)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    crop_min_x, crop_max_x = 0, img.shape[1] - 0
    crop_min_y, crop_max_y = 0, img.shape[0] - 0

    cond_on = True
    cond_min_x, cond_max_x = 0, 75#, 200
    cond_min_y, cond_max_y = 0, 30

    img[:, :, 0] = 0
    img[cond_min_y:cond_max_y, cond_min_x:cond_max_x, 0] = img[cond_min_y:cond_max_y, cond_min_x:cond_max_x, 1]
    img = img[crop_min_y:crop_max_y, crop_min_x:crop_max_x]

    V, U = np.nonzero(img[:, :, 0])

    # convert to torch tensor
    img = transforms.ToTensor()(img).cuda()

    gauss = gauss_2d_batch(img.shape[2], img.shape[1], GAUSS_SIGMA, torch.as_tensor(U).cuda(), torch.as_tensor(V).cuda())
    mm_gauss = gauss[0]
    for i in range(1, len(gauss)):
        mm_gauss = bimodal_gauss(mm_gauss, gauss[i])
    mm_gauss.unsqueeze_(0)
    img[0] = mm_gauss if cond_on else 0

    img = img.unsqueeze(0)
    
    test_data = [img]

for i, f in enumerate(test_data):
    img_t = f
    # GAUSS

    # display image and user will click on two points
    plt.clf()
    # print(img_t[0].detach().cpu().numpy().shape)
    plt.imshow(img_t[0].detach().cpu().numpy().transpose(1,2,0))
    plt.savefig(f'preds_custom/test_full_img_{i}_{cond_on}_{cond_min_x}_{cond_max_x}_{cond_min_y}_{cond_max_y}' + custom_image.split('/')[-1])

    # # get the points the user clicked
    # points = plt.ginput(2)
    # points = np.array(points)
    # points = points.astype(np.int32)
    
    # plot one heatmap for each model with matplotlib
    plt.figure()

    # create len(predictions) subplots
    for j, prediction in enumerate(predictions):
        heatmap = prediction.predict(img_t)
        heatmap = heatmap.detach().cpu().numpy()
        overlay = prediction.plot(img_t.detach().cpu().numpy(), heatmap, image_id=i, write_image=False)

        plt.subplot(1, len(predictions), j+1)
        plt.imshow(overlay)
        plt.title("Model %d"%(j+1))
    plt.savefig(f'preds_custom/test_heatmaps_{i}_{cond_on}_{cond_min_x}_{cond_max_x}_{cond_min_y}_{cond_max_y}' + custom_image.split('/')[-1])

    # TODO: WHAT IS THE REGION THAT HULK_L SHOULD REALLY BE FOCUSING ON?
    # TODO: WHAT ARE ALL THE WAYS OF THINKING ABOUT HOW HUMANS DO FROM ENDPOINT UNTANGLING
