import numpy as np
import cv2
import torch
from knot_detection import KnotDetector
from src.graspability import Graspability
from src.dataset import KeypointsDataset
from src.model import ClassificationModel
from src.prediction import Prediction
from config import *
from torchvision import transforms, utils
import time

class TracerKnotDetector():
    def __init__(self, test_data):
        self.img = test_data['img']
        self.pixels = test_data['pixels']
        self.pixels_so_far = self.pixels[:5]
        self.output_vis_dir = '/home/jainilajmera/hulk-keypoints/test_tkd/'
        
        self.graspability = Graspability()
        self.knot_detector = KnotDetector()

        self.uon_crop_size = 20
        self.uon_config = UNDER_OVER_NONE()
        self.uon_kpts = KeypointsDataset('',
                                    transform=transforms.Compose([transforms.ToTensor()]), 
                                    augment=True, 
                                    config=self.uon_config)
        self.uon_model = ClassificationModel(num_classes=self.uon_config.classes, img_height=self.uon_config.img_height, img_width=self.uon_config.img_width, channels=3)
        self.uon_model.load_state_dict(torch.load('/home/mkparu/hulk-keypoints/checkpoints/2023-01-11-02-15-52_UNDER_OVER_NONE_all_crossings_regen_test/model_11_0.12860.pth'))

        self.uo_config = UNDER_OVER()
        self.uo_model = ClassificationModel(num_classes=self.uo_config.classes, img_height=self.uo_config.img_height, img_width=self.uo_config.img_width, channels=3)
        self.uo_model.load_state_dict(torch.load('/home/vainavi/hulk-keypoints/checkpoints/2023-01-06-23-11-13_UNDER_OVER_under_over_2/model_6_0.40145.pth'))

    def _getuonitem(self, uon_data):
        uon_img = (uon_data['crop_img'][:, :, :3]).copy()
        condition_pixels = np.array(uon_data['spline_pixels'], dtype=np.float64)
        if uon_img.max() > 1:
            uon_img = (uon_img / 255.0).astype(np.float32)
        cable_mask = np.ones(uon_img.shape[:2])
        cable_mask[uon_img[:, :, 1] < 0.35] = 0
        if self.uon_kpts.augment:
            uon_img = self.uon_kpts.call_img_transform(uon_img)
        if self.uon_kpts.sweep:
            uon_img[:, :, 0] = self.uon_kpts.draw_spline(uon_img, condition_pixels[:, 1], condition_pixels[:, 0], label=True)
        else:
            uon_img[:, :, 0] = gauss_2d_batch_efficient_np(self.uon_kpts.crop_span, self.uon_kpts.crop_span, self.uon_kpts.gauss_sigma, condition_pixels[:-self.uon_kpts.pred_len,0], condition_pixels[:-self.uon_kpts.pred_len,1], weights=self.uon_kpts.weights)
        uon_img, _= self.uon_kpts.rotate_condition(uon_img, condition_pixels, center_around_last=True)
        uon_model_input = self.uon_kpts.transform(uon_img.copy())
        return uon_model_input

    def _visualize(self, img, file_name):
        cv2.imwrite(self.output_vis_dir + file_name + '.png', img)

    def _visualize_full(self):
        file_name = 'full_img'
        clr = (255, 0, 0)
        for x, y in self.pixels:
            cv2.circle(self.img, (x, y), 3, clr, -1)
        cv2.imwrite(self.output_vis_dir + file_name + '.png', self.img)

    def _visualize_tensor(self, tensor, file_name):
        img = tensor.clone().detach()
        img = img.squeeze(0)
        img = img.cpu().detach().numpy().transpose(1, 2, 0) * 255
        cv2.imwrite(self.output_vis_dir + file_name, img[..., ::-1])

    def _crop_img(self, img, center_pixel, crop_size):
        x, y = center_pixel
        # note: x, y reversed on img
        crop_img = img[y - crop_size // 2:y + crop_size // 2, x - crop_size // 2:x + crop_size // 2]
        return crop_img

    def _get_pixel_at(self, step):
        if step not in range(len(self.pixels)):
            raise Exception('Step not in range!')
        return self.pixels[step]

    def _get_pixel_so_far_at(self, idx):
        if idx not in range(len(self.pixels_so_far)):
            raise Exception('Index not in range!')
        return self.pixels_so_far[idx]

    def _get_buffer_pixels(self, center_pixel, latest_step, crop_size):
        if latest_step not in range(len(self.pixels)):
            return
        latest_trace_pixel = self._get_pixel_at(latest_step)
        while np.linalg.norm(latest_trace_pixel - center_pixel, ord=np.inf) <= crop_size // 2:
            if latest_step not in range(len(self.pixels_so_far)):
                self.pixels_so_far = np.append(self.pixels_so_far, np.array([latest_trace_pixel]), axis=0)
            latest_step += 1
            if latest_step not in range(len(self.pixels)):
                return
            latest_trace_pixel = self._get_pixel_at(latest_step)

    def _get_spline_pixels(self, center_idx, crop_size):
        # add all spline pixels before and after the crossing pixel that are within the crop size
        spline_pixels = []
        center_pixel = self._get_pixel_so_far_at(center_idx)
        top_left_pixel = np.array([int(center_pixel[0]) -  crop_size // 2, int(center_pixel[1]) - crop_size // 2])

        for curr_idx in range(center_idx + 1, len(self.pixels_so_far)):
            if np.linalg.norm(self._get_pixel_so_far_at(curr_idx) - center_pixel, ord=np.inf) > crop_size // 2:
                break
            spline_pixels.append(self._get_pixel_so_far_at(curr_idx) - top_left_pixel)

        for curr_idx in range(center_idx, 0, -1):
            if np.linalg.norm(self._get_pixel_so_far_at(curr_idx) - center_pixel, ord=np.inf) > crop_size // 2:
                break
            spline_pixels.insert(0, self._get_pixel_so_far_at(curr_idx) - top_left_pixel)
        
        if len(spline_pixels) < 2:
            return
    
        return spline_pixels

    def _determine_pinch(self):
        idx = -1
        pinch = self._get_pixel_so_far_at(idx)
        # TODO: need to tune graspability
        while not self.graspability.find_pixel_point_graspability(pinch, self.pixels_so_far): 
            idx -= 1
            pinch = self._get_pixel_so_far_at(idx)
        return pinch 

    def _determine_cage(self):
        # go back until you're at the trace part that corresponds to overcrossing
        idx = self.knot_detector.get_crossing_pos(self.pixels_so_far[-1])
        cage = self._get_pixel_so_far_at(idx)
        # then trace from there forward and stop once you're in a graspable region
        while not self.graspability.find_pixel_point_graspability(cage, self.pixels_so_far):
            idx += 1
            cage = self._get_pixel_so_far_at(idx)
        return cage

    def trace_and_detect_knot(self):
        # go pixel wise 
        use_cuda = torch.cuda.is_available()

        for model_step in range(5, len(self.pixels)):
            # have not reached model step in trace yet
            if model_step not in range(len(self.pixels_so_far)):
                self.pixels_so_far = np.append(self.pixels_so_far, np.array([self._get_pixel_at(model_step)]), axis=0)
            
            center_pixel = self._get_pixel_so_far_at(model_step)          
            # trace a little extra (buffer) to get pixels for conditioning
            self._get_buffer_pixels(center_pixel, model_step + 1, self.uon_crop_size)
            
            # generate a 20 x 20 crop around the pixel
            uon_data = {}
            uon_data['crop_img'] = self._crop_img(self.img, center_pixel, self.uon_crop_size)
            uon_data['spline_pixels'] = self._get_spline_pixels(model_step, self.uon_crop_size)
            self._visualize(uon_data['crop_img'], f'uon_{model_step}_p.png')

            # get input to UON classifier
            uon_model_input = self._getuonitem(uon_data)
            self._visualize_tensor(uon_model_input, f'uon_{model_step}.png')

            # call model
            predictor = Prediction(self.uon_model, self.uon_config.num_keypoints, self.uon_config.img_height, self.uon_config.img_width, use_cuda)
            pred = np.argmax(predictor.predict(uon_model_input).detach().numpy())

            if pred != 2:
                uo_model_input = uon_model_input
                predictor = Prediction(self.uo_model, self.uo_config.num_keypoints, self.uo_config.img_height, self.uo_config.img_width, use_cuda)
                updated_pred = np.argmax(predictor.predict(uo_model_input).detach().numpy())
                print(model_step, pred, updated_pred)
            else:
                print(model_step, pred)

if __name__ == '__main__':
    test_data = np.load("/home/vainavi/hulk-keypoints/real_data/real_data_for_tracer/test/00000.npy", allow_pickle=True).item()
    tkd = TracerKnotDetector(test_data)
    tkd.trace_and_detect_knot()









