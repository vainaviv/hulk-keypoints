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
import pickle

os.environ["CUDA_VISIBLE_DEVICES"]="2"

# with thresh level around 65%, epoch 18 best
train_set = "detect_ep"
image_dir = 'train_sets/' + train_set
test_dir = image_dir + "/test"
train_dir = image_dir + '/train'
classes = {0: "trivial", 1:"non-trivial", 2:"endpoint"}
test_accuracy = {}
train_accuracy = {}
idx = 0
for checkpoint in sorted(os.listdir("checkpoints/" + train_set)):
    train_correct = 0
    test_correct = 0
    # model
    keypoints =  Model(NUM_KEYPOINTS, pretrained=False, channels=2, num_classes=2, img_height=IMG_HEIGHT, img_width=IMG_WIDTH).cuda()
    keypoints.load_state_dict(torch.load('checkpoints/' + train_set + "/" + checkpoint))
    
    # cuda
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(0)
        keypoints = keypoints.cuda()

    prediction = Prediction(keypoints, NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, use_cuda)
    transform = transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    for folder in sorted(os.listdir(test_dir)):
        image_folder = os.path.join(test_dir, folder)
        expected_label = None
        if folder == "endpoint":
            expected_label = 0
        # elif folder == "endpoint":
        #     expected_label = 1
        else:
            expected_label = 1

        for i, f in enumerate(sorted(os.listdir(image_folder))):
            img = np.load(os.path.join(image_folder, f))
            img[0,:,:] = img[0,:,:]/255.
            img_t = torch.tensor(img).cuda()
            value = prediction.predict(img_t)
            value = value.detach().cpu().numpy()
            predicted = np.argmax(value)
            if predicted == expected_label:
                test_correct += 1

    for folder in sorted(os.listdir(train_dir)):
        image_folder = os.path.join(train_dir, folder)
        expected_label = None
        if folder == "endpoint":
            expected_label = 0
        # elif folder == "endpoint":
        #     expected_label = 1
        else:
            expected_label = 1

        for i, f in enumerate(sorted(os.listdir(image_folder))):
            img = np.load(os.path.join(image_folder, f))
            img[0,:,:] = img[0,:,:]/255.
            img_t = torch.tensor(img).cuda()
            value = prediction.predict(img_t)
            value = value.detach().cpu().numpy()
            predicted = np.argmax(value)
            if predicted == expected_label:
                train_correct += 1

    check = (checkpoint.split("_"))[3]
    test_acc = test_correct/45 #45 test, 300 train
    test_accuracy[check] = test_acc
    train_acc = train_correct/200
    train_accuracy[check] = train_acc

    print(checkpoint)
    print("Test accuracy: ", test_acc)
    print("Train accuracy:", train_acc)
    print("========")

output = open('test_accuracies.pkl', 'wb')
pickle.dump(test_accuracy, output)
output.close()

output = open('train_accuracies.pkl', 'wb')
pickle.dump(train_accuracy, output)
output.close()
