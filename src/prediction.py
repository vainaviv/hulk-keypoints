import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import numpy as np

class Prediction:
    def __init__(self, model, num_keypoints, img_height, img_width, use_cuda):
        self.model = model
        self.num_keypoints = num_keypoints
        self.img_height  = img_height
        self.img_width   = img_width
        self.use_cuda = use_cuda
        
    def predict(self, imgs):
        # img: torch.Tensor(3, height, width) 
        if len(imgs.shape) == 4:
            imgs = imgs.view(-1, imgs.shape[1], imgs.shape[2], imgs.shape[3])    
        elif len(imgs.shape) == 3:
            imgs = imgs.view(-1, imgs.shape[0], imgs.shape[1], imgs.shape[2])
            
        heatmap = self.model.forward(Variable(imgs))
        return heatmap

    def softmax(self,x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def expectation(self, d):
        width, height = d.T.shape
        d = d.T.ravel()
        d_norm = self.softmax(d)
        x_indices = np.array([i % width for i in range(width*height)])
        y_indices = np.array([i // width for i in range(width*height)])
        exp_val = [int(np.dot(d_norm, x_indices)), int(np.dot(d_norm, y_indices))]
        return exp_val
    
    def plot(self, input1, heatmap, target, image_id=0, cls=None, classes=None):
        print("Running inferences on image: %d"%image_id)
        input1 = np.transpose(input1[0], (1,2,0))
        img = input1[:, :, :3] * 255
        img = img.astype(np.uint8)
        t = (target*255).astype(np.uint8)
        true_y, true_x = np.unravel_index(t.argmax(), t.shape)
        all_overlays = []
        h = heatmap[0][0]
        pred_y, pred_x = np.unravel_index(h.argmax(), h.shape)
        vis = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.65, vis, 0.35, 0)
        overlay = cv2.circle(overlay, (pred_x,pred_y), 4, (255,255,255), -1)
        point = input1[:, :, 3] * 255
        tmp = self.expectation(h)
        h = input1[:, :, 3] * 255
        given_y, given_x = np.unravel_index(h.argmax(), h.shape)
        vis = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        overlay = cv2.circle(overlay, (given_x,given_y), 4, (0,0,255), -1)
        overlay = cv2.circle(overlay, (true_x,true_y), 4, (0,255,0), -1)
        #y, x = np.unravel_index(h.argmax(), h.shape)
        #heatmap_val = h[y,x]
        #while heatmap_val > 0.5:
        #    overlay = cv2.circle(overlay, (x,y), 4, (255,255,255), -1)
        #    h[max(0,y-15):min(y+15, len(h)), max(0,x-15):min(x+15, len(h[0]))] = 0
        #    y, x = np.unravel_index(h.argmax(), h.shape)
        #    heatmap_val = h[y,x]
        cv2.imwrite('preds/out%04d.png'%image_id, overlay)
        error = np.linalg.norm(np.array([pred_x - true_x,pred_y - true_y]))
        return error