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
#model_ckpts = ["hulkL_aug_cond_1/model_2_1_498_0.9576036110338699.pth", "hulkL_aug_cond_2/model_2_1_498_1.1168840242554805.pth", "hulkL_aug_cond_3/model_2_1_498_1.0491497036272723.pth"]

# WITH COND LARGE
# model_ckpts = ["hulkL_aug_cond_only_LARGE/model_2_1_498_0.40725811529030426.pth"]

# WITH TWO HEATMAPS
model_ckpts = ["hulkL_seg_2/model_2_1_218_0.2925601195755608.pth"]

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

# test_dataset = KeypointsDataset('hulkL_seg/val/', IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA, augment=False, only_full=False, condition=False)
# test_data = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

image_dir = './train_sets/hulkL_seg_2/val'
for i, f in enumerate(sorted(os.listdir(image_dir))):
    img = cv2.imread(os.path.join(image_dir, f))
    img_t = transform(img)
    img_t = img_t.cuda()

    print(img.shape)

    plt.figure()

    # create len(predictions) subplots
    for j, prediction in enumerate(predictions):
        heatmap = prediction.predict(img_t[0])
        heatmap = heatmap.detach().cpu().numpy()
        horiz_concat = None
        for layer in range(NUM_KEYPOINTS):
            overlay = prediction.plot(img_t[0].detach().cpu().numpy(), heatmap, image_id=i, write_image=False, heatmap_id=layer)
            if (horiz_concat is None):
                horiz_concat = overlay
            else:
                horiz_concat = np.hstack((horiz_concat, overlay))

        plt.subplot(1, len(predictions), j+1)
        plt.imshow(horiz_concat)
        plt.title("Model %d"%(j+1))
    plt.savefig(f'preds/test_heatmaps_{i}_{cond_on}_{cond_min_x}_{cond_max_x}_{cond_min_y}_{cond_max_y}' + custom_img_name + '.png')



# for i, f in enumerate(test_data):
#     img_t = f
#     if (len(img_t.shape) == 4):
#         img_t = img_t.unsqueeze(0)
#     # GAUSS
#     print(img_t.shape)

#     if (not custom_image):
#         custom_image = str(i)
#         cond_on = False
#         cond_min_x, cond_max_x = "na", "na"
#         cond_min_y, cond_max_y = "na", "na"

#     # display image and user will click on two points
#     plt.clf()
#     plt.imshow(img_t[0].squeeze().detach().cpu().numpy().transpose(1,2,0))
#     plt.savefig(f'preds/test_full_img_{i}_{cond_on}_{cond_min_x}_{cond_max_x}_{cond_min_y}_{cond_max_y}' + custom_img_name + '.png')

#     # # get the points the user clicked
#     # points = plt.ginput(2)
#     # points = np.array(points)
#     # points = points.astype(np.int32)
    
#     # plot one heatmap for each model with matplotlib
#     plt.figure()

#     # create len(predictions) subplots
#     for j, prediction in enumerate(predictions):
#         heatmap = prediction.predict(img_t[0])
#         heatmap = heatmap.detach().cpu().numpy()
#         horiz_concat = None
#         for layer in range(NUM_KEYPOINTS):
#             overlay = prediction.plot(img_t[0].detach().cpu().numpy(), heatmap, image_id=i, write_image=False, heatmap_id=layer)
#             if (horiz_concat is None):
#                 horiz_concat = overlay
#             else:
#                 horiz_concat = np.hstack((horiz_concat, overlay))

#         plt.subplot(1, len(predictions), j+1)
#         plt.imshow(horiz_concat)
#         plt.title("Model %d"%(j+1))
#     plt.savefig(f'preds/test_heatmaps_{i}_{cond_on}_{cond_min_x}_{cond_max_x}_{cond_min_y}_{cond_max_y}' + custom_img_name + '.png')

    # TODO: WHAT IS THE REGION THAT HULK_L SHOULD REALLY BE FOCUSING ON?
    # TODO: WHAT ARE ALL THE WAYS OF THINKING ABOUT HOW HUMANS DO FROM ENDPOINT UNTANGLING
