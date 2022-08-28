import pickle
import cv2
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from config import *
from src.model import KeypointsGauss
from src.dataset import KeypointsDataset, transform, gauss_2d_batch, bimodal_gauss, get_gauss
from src.prediction import Prediction
from datetime import datetime, time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

os.environ["CUDA_VISIBLE_DEVICES"]="1"
# torch.cuda.set_device(1)

# WITHOUT COND
#model_ckpts = ["hulkL_aug_crop_1/model_2_1_498_1.154286119789344.pth", "hulkL_aug_crop_2/model_2_1_498_1.0253424582349897.pth", "hulkL_aug_crop_3/model_2_1_498_1.0670859171776241.pth"]

# WITH COND
model_ckpts = ["hulkL_aug_cond_1/model_2_1_498_0.9576036110338699.pth", "hulkL_aug_cond_2/model_2_1_498_1.1168840242554805.pth", "hulkL_aug_cond_3/model_2_1_498_1.0491497036272723.pth"]

# WITH COND LARGE
# model_ckpts = ["hulkL_aug_cond_only_LARGE/model_2_1_498_0.40725811529030426.pth"]

# WITH TWO HEATMAPS
# folder_name = "hulkL_aug_cond_only_LARGE_2heatmaps_RESUME_sim"

# model_ckpts = [f"{folder_name}/model_2_1_480_0.23921896447263052.pth"]

# cuda
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(0)
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

# model
keypoints_models = []
for model_ckpt in model_ckpts:
    keypoints = KeypointsGauss(NUM_KEYPOINTS, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, channels=3).cuda()
    keypoints.load_state_dict(torch.load('checkpoints/%s'%model_ckpt))
    keypoints_models.append(keypoints)

if use_cuda:
    for keypoints in keypoints_models:
        keypoints = keypoints.cuda()

predictions = []

for keypoints in keypoints_models:
    prediction = Prediction(keypoints, NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, use_cuda)
    predictions.append(prediction)

transform = transform = transforms.Compose([
    transforms.ToTensor()
])

test_dataset = KeypointsDataset('hulkL_seg/train/', IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA, augment=False, only_full=True, condition=True, sim=True)
# test_data = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
custom_image = None

### OPTIONALLY LOAD CUSTOM IMAGE ###
# custom_img_name = "1640295787"
# custom_image = f"/host/data_bank/figure8_drop/{custom_img_name}/color_0.npy"
# if custom_image:
#     #img = cv2.imread(custom_image)
#     img = np.load(custom_image)
#     # print(img.max(), img.min())

#     # FROM 0 to image size
#     crop_min_x, crop_max_x = 400, 570 #img.shape[1] #- 200
#     crop_min_y, crop_max_y = 100, 270 #img.shape[0]# - 200

#     # From 0 to 200
#     cond_on = True
#     cond_min_x, cond_max_x = 0, 200 #img.shape[1]
#     cond_min_y, cond_max_y = 190, 200 #img.shape[0]

#     img[:, :, 0] = 0
#     img = img[crop_min_y:crop_max_y, crop_min_x:crop_max_x, :]
#     img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

#     img[cond_min_y:cond_max_y, cond_min_x:cond_max_x, 0] = img[cond_min_y:cond_max_y, cond_min_x:cond_max_x, 1] > 50

#     V, U = np.nonzero(img[:, :, 0])

#     # convert to torch tensor
#     img = transforms.ToTensor()(img).cuda()

#     # gauss = gauss_2d_batch(img.shape[2], img.shape[1], GAUSS_SIGMA, torch.as_tensor(U).cuda(), torch.as_tensor(V).cuda())
#     # mm_gauss = gauss[0]
#     # for i in range(1, len(gauss)):
#     #     mm_gauss = bimodal_gauss(mm_gauss, gauss[i])
#     # mm_gauss.unsqueeze_(0)
#     img[0] = get_gauss(img.shape[2], img.shape[1], GAUSS_SIGMA, torch.as_tensor(U).cuda(), torch.as_tensor(V).cuda()) if cond_on else 0

#     img = img.unsqueeze(0)
    
#     test_data = [img]

# Load custom folder
output_folder_name = f'preds_custom_ontrain_{folder_name}'
if not os.path.exists(output_folder_name):
    os.mkdir(output_folder_name)

# input_folder_name = 'detectron_pred_bb'
# test_data = []
# for file in os.listdir(input_folder_name):
#     if 'points' not in file:
#         continue
    
#     #img = cv2.imread(custom_image)
#     img_cond = np.load(os.path.join(input_folder_name, file), allow_pickle=True).item()
#     img = img_cond['image']
#     cond = np.array(img_cond['points'])
#     if len(cond) == 0:
#         continue
#     # print(img.max(), img.min())

#     # From 0 to 200
#     cond_on = True
#     cond_min_x, cond_max_x = min(cond[:, 1]), max(cond[:, 1])
#     cond_min_y, cond_max_y = min(cond[:, 0]), max(cond[:, 0])

#     img[:, :, 0] = 0
#     img[cond_min_y:cond_max_y, cond_min_x:cond_max_x, 0] = 255*img[cond_min_y:cond_max_y, cond_min_x:cond_max_x, 1] > 50
#     img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

#     V, U = np.nonzero(img[:, :, 0])

#     # convert to torch tensor
#     img = transforms.ToTensor()(img).cuda()

#     # gauss = gauss_2d_batch(img.shape[2], img.shape[1], GAUSS_SIGMA, torch.as_tensor(U).cuda(), torch.as_tensor(V).cuda())
#     # mm_gauss = gauss[0]
#     # for i in range(1, len(gauss)):
#     #     mm_gauss = bimodal_gauss(mm_gauss, gauss[i])
#     # mm_gauss.unsqueeze_(0)
#     img[0] = get_gauss(img.shape[2], img.shape[1], GAUSS_SIGMA, torch.as_tensor(U).cuda(), torch.as_tensor(V).cuda()) if cond_on else 0

#     img = img.unsqueeze(0)
    
#     test_data.append(img)
    

for i, f in enumerate(test_dataset):
    img_t = f[0]
    if (len(img_t.shape) < 4):
        img_t = img_t.unsqueeze(0)
    # print(img_t.shape)
    # print(img_t.shape)
    # GAUSS

    # if (not custom_image):
    #     custom_image = str(i)
    #     cond_on = True
    #     cond_min_x, cond_max_x = "na", "na"
    #     cond_min_y, cond_max_y = "na", "na"

    # display image and user will click on two points
    plt.clf()
    plt.imshow(img_t[0].squeeze().detach().cpu().numpy().transpose(1,2,0))
    # plt.savefig(f'{output_folder_name}/test_full_img_{i}_{cond_on}_{cond_min_x}_{cond_max_x}_{cond_min_y}_{cond_max_y}' + '.png')

    # # get the points the user clicked
    # points = plt.ginput(2)
    # points = np.array(points)
    # points = points.astype(np.int32)
    
    # plot one heatmap for each model with matplotlib
    plt.figure()

    # create len(predictions) subplots
    for j, prediction in enumerate(predictions):
        heatmap = prediction.predict(img_t[0])
        heatmap = heatmap.detach().cpu().numpy()
        horiz_concat = None
        for layer in range(NUM_KEYPOINTS):
            overlay = prediction.plot(img_t.detach().cpu().numpy(), heatmap, image_id=i, write_image=False, heatmap_id=layer)
            if (horiz_concat is None):
                horiz_concat = overlay
            else:
                horiz_concat = np.hstack((horiz_concat, overlay))

        plt.subplot(1, len(predictions), j+1)
        plt.imshow(horiz_concat)
        plt.title("Model %d"%(j+1))
    plt.savefig(f'{output_folder_name}/test_heatmaps_{i}'+ '.png') #_{cond_on}_{cond_min_x}_{cond_max_x}_{cond_min_y}_{cond_max_y}' + '.png')

    # TODO: WHAT IS THE REGION THAT HULK_L SHOULD REALLY BE FOCUSING ON?
    # TODO: WHAT ARE ALL THE WAYS OF THINKING ABOUT HOW HUMANS DO FROM ENDPOINT UNTANGLING
