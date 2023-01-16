import pickle
import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model import ClassificationModel, KeypointsGauss
from config import *
import imgaug.augmenters as iaa
from imgaug.augmentables import KeypointsOnImage
from torchvision import transforms, utils

class Tracer:
    def __init__(self) -> None:
        self.trace_config = TRCR32_CL3_12_UNet34_B64_OS_MedleyFix_MoreReal_Sharp()
        self.trace_model =  KeypointsGauss(1, img_height=self.trace_config.img_height, img_width=self.trace_config.img_width, channels=3, resnet_type=self.trace_config.resnet_type, pretrained=self.trace_config.pretrained)
        self.trace_model.load_state_dict(torch.load('/home/vainavi/hulk-keypoints/checkpoints/2023-01-13-20-41-47_TRCR32_CL3_12_UNet34_B64_OS_MedleyFix_MoreReal_Sharp/model_36_0.00707.pth'))
        augs = []
        augs.append(iaa.Resize({"height": self.img_height, "width": self.img_width}))
        self.real_img_transform = iaa.Sequential(augs, random_order=False)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def _get_evenly_spaced_points(self, pixels, num_points, start_idx, spacing, img_size, backward=True, randomize_spacing=True):
        def is_in_bounds(pixel):
            return pixel[0] >= 0 and pixel[0] < img_size[0] and pixel[1] >= 0 and pixel[1] < img_size[1]
        def get_rand_spacing(spacing):
            return spacing * np.random.uniform(0.8, 1.2) if randomize_spacing else spacing
        # get evenly spaced points
        last_point = np.array(pixels[start_idx]).squeeze()
        points = [last_point]
        if not is_in_bounds(last_point):
            return np.array([])
        rand_spacing = get_rand_spacing(spacing)
        while len(points) < num_points and start_idx > 0 and start_idx < len(pixels):
            start_idx -= (int(backward) * 2 - 1)
            cur_spacing = np.linalg.norm(np.array(pixels[start_idx]).squeeze() - last_point)
            if cur_spacing > rand_spacing and cur_spacing < 2*rand_spacing:
                last_point = np.array(pixels[start_idx]).squeeze()
                rand_spacing = get_rand_spacing(spacing)
                if is_in_bounds(last_point):
                    points.append(last_point)
                else:
                    return np.array([])
        return np.array(points)[..., ::-1]

    def center_pixels_on_cable(self, image, pixels):
        # for each pixel, find closest pixel on cable
        image_mask = image[:, :, 0] > 100
        # erode white pixels
        kernel = np.ones((2,2),np.uint8)
        image_mask = cv2.erode(image_mask.astype(np.uint8), kernel, iterations=1)
        white_pixels = np.argwhere(image_mask)

        processed_pixels = []
        for pixel in pixels:
            # find closest pixel on cable
            distances = np.linalg.norm(white_pixels - pixel[::-1], axis=1)
            closest_pixel = white_pixels[np.argmin(distances)]
            processed_pixels.append([closest_pixel])
        return np.array(processed_pixels)[:, ::-1]

    def call_img_transform(self, img, kpts, is_real_example=False):
        img = img.copy()
        img = (img * 255.0).astype(np.uint8)
        img, keypoints = self.real_img_transform(image=img, keypoints=kpts)
        img = (img / 255.0).astype(np.float32)
        return img, keypoints

    def get_crop_and_cond_pixels(self, img, condition_pixels, center_around_last=False):
        center_of_crop = condition_pixels[-self.pred_len*(1 - int(center_around_last))-1]
        img = np.pad(img, ((self.crop_width, self.crop_width), (self.crop_width, self.crop_width), (0, 0)), 'constant')
        center_of_crop = center_of_crop.copy() + self.crop_width

        crop = img[max(0, center_of_crop[0] - self.crop_width): min(img.shape[0], center_of_crop[0] + self.crop_width + 1),
                    max(0, center_of_crop[1] - self.crop_width): min(img.shape[1], center_of_crop[1] + self.crop_width + 1)]
        img = crop
        top_left = [center_of_crop[0] - self.crop_width, center_of_crop[1] - self.crop_width]
        condition_pixels = [[pixel[0] - top_left[0] + self.crop_width, pixel[1] - top_left[1] + self.crop_width] for pixel in condition_pixels]

        return img, np.array(condition_pixels)[:, ::-1], top_left

    def get_trp_model_input(self, crop, crop_points, center_around_last=False, is_real_example=False):
        kpts = KeypointsOnImage.from_xy_array(crop_points, shape=crop.shape)
        img, kpts = self.call_img_transform(img=crop, kpts=kpts, is_real_example=is_real_example)

        points = []
        for k in kpts:
            points.append([k.x,k.y])
        points = np.array(points)

        points_in_image = []
        for i, point in enumerate(points):
            px, py = int(point[0]), int(point[1])
            if px not in range(img.shape[1]) or py not in range(img.shape[0]):
                continue
            points_in_image.append(point)
        points = np.array(points_in_image)

        angle = 0
        if self.rot_cond:
            if center_around_last:
                dir_vec = points[-1] - points[-2]
            else:
                dir_vec = points[-self.pred_len-1] - points[-self.pred_len-2]
            angle = np.arctan2(dir_vec[1], dir_vec[0])

            # rotate image specific angle using cv2.rotate
            M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle*180/np.pi, 1)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


        # rotate all points by angle around center of image
        points = points - np.array([img.shape[1]/2, img.shape[0]/2])
        points = np.matmul(points, np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]))
        points = points + np.array([img.shape[1]/2, img.shape[0]/2])

        if center_around_last:
            img[:, :, 0] = self.draw_spline(img, points[:,1], points[:,0])# * cable_mask
        else:
            img[:, :, 0] = self.draw_spline(img, points[:-self.pred_len,1], points[:-self.pred_len,0])# * cable_mask

        cable_mask = np.ones(img.shape[:2])
        cable_mask[img[:, :, 1] < 0.4] = 0

        return self.transform(img.copy()).cuda(), points, cable_mask, angle

    def trace(self, image, start_points, exact_path_len, viz=False, model=None):    
        num_condition_points = self.trace_config.condition_len
        if start_points is None or len(start_points) < num_condition_points:
            raise ValueError(f"Need at least {num_condition_points} start points")
        path = [start_point for start_point in start_points]
        disp_img = (image.copy() * 255.0).astype(np.uint8)

        for iter in range(exact_path_len):
            condition_pixels = [p for p in path[-num_condition_points:]]
            
            crop, cond_pixels_in_crop, top_left = self.get_crop_and_cond_pixels(image, condition_pixels, center_around_last=True)
            ymin, xmin = np.array(top_left) - self.trace_config.crop_width

            model_input, _, cable_mask, angle = self.get_trp_model_input(crop, cond_pixels_in_crop, center_around_last=True)

            crop_eroded = cv2.erode((cable_mask).astype(np.uint8), np.ones((2, 2)), iterations=1)

            if viz:
                cv2.imshow('model input', model_input.detach().cpu().numpy().transpose(1, 2, 0))
                cv2.waitKey(1)

            model_output = model(model_input.unsqueeze(0)).detach().cpu().numpy().squeeze()
            model_output *= crop_eroded.squeeze()
            model_output = cv2.resize(model_output, (crop.shape[1], crop.shape[0]))

            # undo rotation if done in preprocessing
            M = cv2.getRotationMatrix2D((model_output.shape[1]/2, model_output.shape[0]/2), -angle*180/np.pi, 1)
            model_output = cv2.warpAffine(model_output, M, (model_output.shape[1], model_output.shape[0]))

            argmax_yx = np.unravel_index(model_output.argmax(), model_output.shape)# * np.array([crop.shape[0] / config.img_height, crop.shape[1] / config.img_width])

            # get angle of argmax yx
            global_yx = np.array([argmax_yx[0] + ymin, argmax_yx[1] + xmin]).astype(int)
            path.append(global_yx)

            disp_img = cv2.circle(disp_img, (global_yx[1], global_yx[0]), 1, (0, 0, 255), 2)
            # add line from previous to current point
            if len(path) > 1:
                disp_img = cv2.line(disp_img, (path[-2][1], path[-2][0]), (global_yx[1], global_yx[0]), (0, 0, 255), 2)

            if viz:
                cv2.imshow("disp_img", disp_img)
                cv2.waitKey(1)
        return path

    def _trace(self, img, prev_pixels):
        pixels = self.center_pixels_on_cable(img, prev_pixels)[..., ::-1]
        for j in range(len(pixels)):
            cur_pixel = pixels[j][0]
            if cur_pixel[0] >= 0 and cur_pixel[1] >= 0 and cur_pixel[1] < img.shape[0] and cur_pixel[0] < img.shape[1]:
                start_idx = j
                break
        try:
            starting_points = self._get_evenly_spaced_points(pixels, self.trace_config.condition_len, start_idx + 1, self.trace_config.cond_point_dist_px, img.shape, backward=False, randomize_spacing=False)
        except:
            starting_points = []
        if len(starting_points) < self.trace_config.condition_len:
            raise Exception("Not enough starting points")
        spline = self.trace(img, starting_points, exact_path_len=10, model=self.trace_model, viz=False)
        return spline