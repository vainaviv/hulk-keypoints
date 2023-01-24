import numpy as np
import cv2
import time
import shutil
import os
import sys
import argparse
import torch
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from torchvision import transforms, utils
from scipy import interpolate
from collections import OrderedDict
from itertools import chain

# from untangling.tracer_knot_detect.knot_detection import KnotDetector # Uncomment for triton4
# from untangling.tracer_knot_detect.src.graspability import Graspability # Uncomment for triton4
# from untangling.tracer_knot_detect.src.model import ClassificationModel # Uncomment for triton4
# from untangling.tracer_knot_detect.src.prediction import Prediction # Uncomment for triton4
# from untangling.tracer_knot_detect.config import * # Uncomment for triton4
# from untangling.tracer_knot_detect.tracer import Tracer, TraceEnd # Uncomment for triton4

sys.path.insert(0, '..') # Uncomment for bajcsy
from triton_trace_files.knot_detection import KnotDetector # Uncomment for bajcsy
from triton_trace_files.tracer import Tracer, TraceEnd # Uncomment for bajcsy
from src.graspability import Graspability # Uncomment for bajcsy
from src.model import ClassificationModel # Uncomment for bajcsy
from src.prediction import Prediction # Uncomment for bajcsy
from config import * # Uncomment for bajcsy

class TracerKnotDetector():
    def __init__(self, parallel=False):
        self.img = None
        self.pixels = []
        self.starting_pixels_for_trace = None
        self.parallel = parallel
        self.output_vis_dir = './test_tkd/'
        self.vis_idx = 0
        if os.path.exists(self.output_vis_dir):
            shutil.rmtree(self.output_vis_dir)
        os.makedirs(self.output_vis_dir)
        
        self.graspability = Graspability()
        self.local_crossing_stream = []
        self.crossing_locs = []
        self.line_segment_to_crossing_loc = OrderedDict()
        self.num_steps_min_for_crossing = 1
        self.detector = KnotDetector()
        self.knot = None
        self.last_trace_step_in_knot = None
        self.under_crossing_after_knot = None
        self.under_crossing_before_knot = None
        self.trace_end = None
        self.gauss_sigma = 1
        self.threshold = 0.275

        self.crop_size = 20
        self.crop_width = self.crop_size // 2
        self.uon_config = UNDER_OVER_NONE

        augs = []
        augs.append(iaa.Resize({'height': self.uon_config.img_height, 'width': self.uon_config.img_width}))
        self.real_img_transform = iaa.Sequential(augs, random_order=False)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.uon_model = ClassificationModel(num_classes=self.uon_config.classes, img_height=self.uon_config.img_height, img_width=self.uon_config.img_width, channels=3)
        # self.uon_model.load_state_dict(torch.load('/home/justin/yumi/cable-untangling/untangling/tracer_knot_detect/models/under_over_none/model_11_0.12860.pth')) # Uncomment for triton4
        self.uon_model.load_state_dict(torch.load('/home/mkparu/hulk-keypoints/checkpoints/2023-01-11-02-15-52_UNDER_OVER_NONE_all_crossings_regen_test/model_11_0.12860.pth')) # Uncomment for bajcsy

        self.uo_config = UNDER_OVER_RNet34_lr1e5_medley_03Hard2
        self.uo_model = ClassificationModel(num_classes=self.uo_config.classes, img_height=self.uo_config.img_height, img_width=self.uo_config.img_width, channels=3)
        # self.uo_model.load_state_dict(torch.load('/home/justin/yumi/cable-untangling/untangling/tracer_knot_detect/models/under_over/88_class/model_7_0.17465.pth')) # Uncomment for triton4
        self.uo_model.load_state_dict(torch.load('/home/vainavi/hulk-keypoints/checkpoints/2023-01-21-21-21-37_UNDER_OVER_RNet34_lr1e4_medley_03Hard2_wReal_B16_recentered_mark_crossing_smaller/model_7_0.17465.pth')) # Uncomment for bajcsy

        self.tracer = Tracer()

    def _set_data(self, img, starting_pixels):
        self.detector._reset()
        self.img = img
        self.local_crossing_stream = []
        self.crossing_locs = []
        self.crossings = []
        self.pixels = []
        self.starting_pixels_for_trace = None
        self.knot = None
        self.last_trace_step_in_knot = None
        self.under_crossing_after_knot = None
        self.under_crossing_before_knot = None
        self.trace_end = None
        # self.pixels = pixels
        # print('before calling interpolate', starting_pixels)
        self.starting_pixels_for_trace = self.interpolate_trace(starting_pixels)
        # print('starting_pixels_for_trace, after calling interpolate', self.starting_pixels_for_trace)
        vis_trace = self.tracer.visualize_path(self.img, self.starting_pixels_for_trace)
        cv2.imwrite('viz_start_points.png', vis_trace)

    def _call_img_transform(self, img):
        img = img.copy()
        normalize = False
        if np.max(img) <= 1.0:
            normalize = True
        if normalize:
            img = (img * 255.0).astype(np.uint8)
        img = self.real_img_transform(image=img)
        if normalize:
            img = (img / 255.0).astype(np.float32)
        return img

    def _rotate_condition(self, img, points, center_around_last=False, index=0):
        img = img.copy()
        angle = 0
        # points = self.deduplicate_points(points)
        if self.uon_config.rot_cond:
            if center_around_last:
                dir_vec = points[-1] - points[0]
            else:
                dir_vec = points[-self.pred_len - 1] - points[-self.pred_len - 2]
            # angle = np.arctan2(dir_vec[1], dir_vec[0]) * 180/np.pi
            angle = np.arctan2(dir_vec[0], dir_vec[1]) * 180 / np.pi
            if angle < -90.0:
                angle += 180
            elif angle > 90.0:
                angle -= 180
            # rotate image-specific angle using cv2.rotate
            M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        return img, angle

    def _draw_spline(self, crop, x, y, label=False):
        if len(x) < 2:
            raise Exception('if drawing spline, must have 2 points minimum for label')
        # x = list(OrderedDict.fromkeys(x))
        # y = list(OrderedDict.fromkeys(y))
        tmp = OrderedDict()
        for point in zip(x, y):
            tmp.setdefault(point[:2], point)
        mypoints = np.array(list(tmp.values()))
        x, y = mypoints[:, 0], mypoints[:, 1]
        k = len(x) - 1 if len(x) < 4 else 3
        if k == 0:
            x = np.append(x, np.array([x[0]]))
            y = np.append(y, np.array([y[0] + 1]))
            k = 1

        tck, u = interpolate.splprep([x, y], s=0, k=k)
        xnew, ynew = interpolate.splev(np.linspace(0, 1, 100), tck, der=0)
        xnew = np.array(xnew, dtype=int)
        ynew = np.array(ynew, dtype=int)

        x_in = np.where(xnew < crop.shape[0])
        xnew = xnew[x_in[0]]
        ynew = ynew[x_in[0]]
        x_in = np.where(xnew >= 0)
        xnew = xnew[x_in[0]]
        ynew = ynew[x_in[0]]
        y_in = np.where(ynew < crop.shape[1])
        xnew = xnew[y_in[0]]
        ynew = ynew[y_in[0]]
        y_in = np.where(ynew >= 0)
        xnew = xnew[y_in[0]]
        ynew = ynew[y_in[0]]

        spline = np.zeros(crop.shape[:2])
        if label:
            weights = np.ones(len(xnew))
        else:
            weights = np.geomspace(0.5, 1, len(xnew))

        spline[xnew, ynew] = weights
        spline = np.expand_dims(spline, axis=2)
        spline = np.tile(spline, 3)
        spline_dilated = cv2.dilate(spline, np.ones((3, 3), np.uint8), iterations=1)
        return spline_dilated[:, :, 0]
    
    def _gauss_2d_batch_efficient_np(self, width, height, U, V, weights, normalize=False):
        crop_size = 3 * self.gauss_sigma
        ret = np.zeros((height + 2 * crop_size, width + 2 * crop_size + 1))
        X, Y = np.meshgrid(np.arange(-crop_size, crop_size + 1), np.arange(-crop_size, crop_size + 1))
        gaussian = np.exp(-(X ** 2 + Y ** 2) / (2.0 * self.gauss_sigma ** 2))
        for i in range(len(weights)):
            cur_weight = weights[i]
            y, x = int(V[i]) + crop_size, int(U[i]) + crop_size
            if ret[y - crop_size:y + crop_size + 1, x - crop_size:x + crop_size + 1].shape == gaussian.shape:
                ret[y - crop_size:y + crop_size + 1, x - crop_size:x + crop_size + 1] = np.max((cur_weight * gaussian, ret[y - crop_size:y + crop_size + 1, x - crop_size:x + crop_size + 1]), axis=0)
        if normalize:
            ret = ret / ret.max()
        return ret[crop_size:crop_size + height, crop_size:crop_size + width]

    def _getuonitem(self, uon_data):
        uon_img = (uon_data['crop_img'][:, :, :3]).copy()
        condition_pixels = np.array(uon_data['spline_pixels'], dtype=np.float64)
        if uon_img.max() > 1:
            uon_img = (uon_img / 255.0).astype(np.float32)
        cable_mask = np.ones(uon_img.shape[:2])
        cable_mask[uon_img[:, :, 1] < 0.35] = 0
        uon_img = self._call_img_transform(uon_img)
        uon_img[:, :, 0] = self._draw_spline(uon_img, condition_pixels[:, 0], condition_pixels[:, 1], label=True)
        uon_img, _ = self._rotate_condition(uon_img, condition_pixels, center_around_last=True)
        uon_model_input = self.transform(uon_img.copy())
        return uon_model_input

    def _getuoitem(self, uo_data):
        uon_data = uo_data
        uon_model_input = self._getuonitem(uon_data)
        img = uon_model_input.clone().detach()
        img = img.squeeze(0).numpy().transpose((1, 2, 0))
        img[:, :, 1] = self._gauss_2d_batch_efficient_np(self.crop_size, self.crop_size, [self.crop_width], [self.crop_width], weights=[1.0])
        uo_model_input = self.transform(img.copy())
        return uo_model_input

    def _visualize(self, img, file_name):
        if not file_name.endswith('.png'):
            file_name += '.png'
        cv2.imwrite(self.output_vis_dir + file_name, img)

    def _visualize_all_crossings(self):
        img = self.img.copy()
        file_name = f'all_crossings_img_{self.vis_idx}'
        # red -> overcrossing, blue -> undercrossing (cv2 uses BGR)
        u_clr = (255, 0, 0)
        o_clr = (0, 0, 255)
        for ctr, crossing in enumerate(self.detector.crossings):
            y, x = crossing['loc']
            if crossing['ID'] == 0:
                cv2.circle(img, (x, y), 3, u_clr, -1)
                cv2.putText(img, str(ctr), (x + 2, y + 2), cv2.FONT_HERSHEY_PLAIN, 1, u_clr)
            if crossing['ID'] == 1:
                cv2.circle(img, (x, y), 3, o_clr, -1)
                cv2.putText(img, str(ctr), (x - 2, y - 2), cv2.FONT_HERSHEY_PLAIN, 1, o_clr)
        cv2.imwrite(self.output_vis_dir + file_name + '.png', img)

    def _visualize_all_cages_pinches(self, cages, pinches, idx=0):
        img = self.img.copy()
        cage_clr = (255, 0, 0)
        pinch_clr = (0, 0, 255)
        for key in cages:
            cage = self._get_pixel_at(key)[::-1]
            cv2.circle(img, cage, 3, cage_clr, -1)
        cv2.putText(img, 'cage', (cage[0]+2, cage[1]+2), cv2.FONT_HERSHEY_PLAIN, 1, cage_clr)
        
        for key in pinches:
            pinch = self._get_pixel_at(key)[::-1]
            cv2.circle(img, pinch, 3, pinch_clr, -1)
        cv2.putText(img, 'pinch', (pinch[0] + 2, pinch[1] + 2), cv2.FONT_HERSHEY_PLAIN, 1, pinch_clr)
        file_name = f'cages_pinches_{self.vis_idx}.png'
        path = os.path.join(self.output_vis_dir, file_name)
        cv2.imwrite(path , img)

    def _visualize_cage_pinch(self, cage, pinch, idx=0):
        img = self.img.copy()
        cage = cage.copy()[::-1]
        pinch = pinch.copy()[::-1]
        cage_clr = (255, 0, 0)
        pinch_clr = (0, 0, 255)
        cv2.circle(img, cage, 3, cage_clr, -1)
        cv2.putText(img, 'cage', (cage[0] + 2, cage[1] + 2), cv2.FONT_HERSHEY_PLAIN, 1, cage_clr)
        cv2.circle(img, pinch, 3, pinch_clr, -1)
        cv2.putText(img, 'pinch', (pinch[0] + 2, pinch[1] + 2), cv2.FONT_HERSHEY_PLAIN, 1, pinch_clr)
        file_name = f'cage_pinch_graspable_{self.vis_idx}.png'
        path = os.path.join(self.output_vis_dir, file_name)
        cv2.imwrite(path , img)
    

    def _visualize_full(self):
        img = self.img.copy()
        file_name = f'full_img_{self.vis_idx}'
        img = self.tracer.visualize_path(img, self.pixels)
        cv2.imwrite(self.output_vis_dir + file_name + '_with_trace' + '.png', img)

    def _visualize_knot(self):
        if not self.knot:
            raise Exception('No knot detected for visualization!')
        img = self.img.copy()
        file_name = f'knot_img_{self.vis_idx}'
        # red -> overcrossing, blue -> undercrossing (cv2 uses BGR)
        u_clr = (255, 0, 0)
        o_clr = (0, 0, 255)
        for ctr, crossing in enumerate(self.knot):
            y, x = crossing['loc']
            if crossing['ID'] == 0:
                cv2.circle(img, (x, y), 3, u_clr, -1)
                cv2.putText(img, str(ctr), (x + 2, y + 2), cv2.FONT_HERSHEY_PLAIN, 1, u_clr)
            if crossing['ID'] == 1:
                cv2.circle(img, (x, y), 3, o_clr, -1)
                cv2.putText(img, str(ctr), (x - 2, y - 2), cv2.FONT_HERSHEY_PLAIN, 1, o_clr)
       
        if self.under_crossing_after_knot:
            y, x = self.under_crossing_after_knot['loc']
            u_after_clr = (0, 255, 0)
            cv2.circle(img, (x, y), 3, u_after_clr, -1)
            cv2.putText(img, 'next U', (x, y), cv2.FONT_HERSHEY_PLAIN, 1, u_after_clr )

        cv2.imwrite(self.output_vis_dir + file_name + '.png', img)

    def _visualize_tensor(self, tensor, file_name):
        img = tensor.clone().detach()
        img = img.squeeze(0)
        img = img.cpu().detach().numpy().transpose(1, 2, 0).copy() * 255
        cv2.imwrite(self.output_vis_dir + file_name, img[..., ::-1])

    def _crop_img(self, img, center_pixel, crop_size):
        y, x = center_pixel
        # note: x, y reversed on img
        crop_img = img[y - crop_size // 2:y + crop_size // 2, x - crop_size // 2:x + crop_size // 2]
        img = crop_img[1:, 1:, :]
        if img.shape == (19,19,3):
            img = cv2.resize(img, (crop_size, crop_size))
            return img
        else:
            return None

    def _get_pixel_at(self, step):
        if step not in range(len(self.pixels)):
            raise Exception(f'Step {step} not in range of {len(self.pixels)}!')
        return self.pixels[step]

    def _get_spline_pixels(self, center_idx, crop_size):
        # add all spline pixels before and after the crossing pixel that are within the crop size
        spline_pixels = []
        center_pixel = self._get_pixel_at(center_idx)
        y, x = center_pixel
        # top_left_pixel = np.array([int(center_pixel[0]) -  crop_size // 2, int(center_pixel[1]) - crop_size // 2])
        top_left_pixel = np.array([int(y) -  crop_size // 2, int(x) - crop_size // 2])
        for curr_idx in range(center_idx + 1, len(self.pixels)):
            if np.linalg.norm(self._get_pixel_at(curr_idx) - center_pixel, ord=np.inf) > crop_size // 2:
                break
            spline_pixels.append(self._get_pixel_at(curr_idx) - top_left_pixel)

        for curr_idx in range(center_idx, 0, -1):
            if np.linalg.norm(self._get_pixel_at(curr_idx) - center_pixel, ord=np.inf) > crop_size // 2:
                break
            spline_pixels.insert(0, self._get_pixel_at(curr_idx) - top_left_pixel)
        if len(spline_pixels) < 2:
            return
    
        return spline_pixels

    def _predict_uon(self, uon_model_input, file_name=None):
        predictor = Prediction(self.uon_model, self.uon_config.num_keypoints, self.uon_config.img_height, self.uon_config.img_width, parallelize=self.parallel)
        prediction_prob_arr = predictor.predict(uon_model_input).cpu().detach().numpy().squeeze()
        pred = np.argmax(prediction_prob_arr)
        prediction_prob = prediction_prob_arr[pred]
        # call separate model for under/overcrossing
        if pred != 2:
            uo_model_input = uon_model_input
            img = uo_model_input.clone().detach()
            img = img.squeeze(0).numpy().transpose((1, 2, 0))
            img[:, :, 1] = self._gauss_2d_batch_efficient_np(self.crop_size, self.crop_size, [self.crop_width], [self.crop_width], weights=[1.0])
            uo_model_input = self.transform(img.copy())
            if file_name is not None:
                self._visualize_tensor(uo_model_input, file_name)
            predictor = Prediction(self.uo_model, self.uo_config.num_keypoints, self.uo_config.img_height, self.uo_config.img_width, parallelize=self.parallel)
            updated_prediction_prob = predictor.predict(uo_model_input).cpu().detach().numpy().squeeze()
            if updated_prediction_prob >= self.threshold:
                return 1, 0.5 + 0.5 * (updated_prediction_prob - self.threshold) / (1 - self.threshold)
            else: 
                return 0, 0.5 + 0.5 * (self.threshold - updated_prediction_prob) / self.threshold
        else:
            return pred, prediction_prob

    def _predict_uo(self, uo_model_input, file_name=None):
        if file_name is not None:
            self._visualize_tensor(uo_model_input, file_name)
        predictor = Prediction(self.uo_model, self.uo_config.num_keypoints, self.uo_config.img_height, self.uo_config.img_width, parallelize=self.parallel)
        updated_prediction_prob = predictor.predict(uo_model_input).cpu().detach().numpy().squeeze()
        if updated_prediction_prob >= self.threshold:
            return 1, 0.5 + 0.5 * (updated_prediction_prob - self.threshold) / (1 - self.threshold)
        else: 
            return 0, 0.5 + 0.5 * (self.threshold - updated_prediction_prob) / self.threshold
    
    # return processed under/overcrossing from stream
    def _vote_and_process_under_over_crossing(self):
        # using 1, -1 instead of 1, 0 so the confidence matters for U as well
        x_arr = []
        y_arr = []
        pixels_idxs = []
        weighted_sum = 0
        for crossing_dict in self.local_crossing_stream:
            if crossing_dict['uon'] == 0:
                weighted_sum -= crossing_dict['prob']
            else:
                weighted_sum += crossing_dict['prob']
            x_arr.append(crossing_dict['center_pixel'][0])
            y_arr.append(crossing_dict['center_pixel'][1])
            pixels_idxs.append(crossing_dict['pixels_idx'])
        
        x_arr, y_arr = np.array(x_arr), np.array(y_arr)
        avg_x, avg_y = int(np.mean(x_arr)), int(np.mean(y_arr))
        pixels_idx = int(np.median(pixels_idxs))
        med_x, med_y = int(np.median(x_arr)), int(np.median(y_arr))
        weighted_sum = weighted_sum / len(self.local_crossing_stream)

        if weighted_sum >= 0:
            return {'loc': (avg_x, avg_y), 'ID': 1, 'confidence': weighted_sum, 'pixels_idx': pixels_idx}
        else:
            return {'loc': (avg_x, avg_y), 'ID': 0, 'confidence': -weighted_sum, 'pixels_idx': pixels_idx}
    
    # add new crossing to stack and check if a knot is formed after adding this new crossing
    def _add_crossing_and_run_knot_detection(self, crossing):
        return self.detector.encounter_seg(crossing)

     # add new crossing to stack
    def _add_crossing(self, crossing):
        self.detector.encounter_seg(crossing)
        return

    # check if a knot is formed (with existing stack)
    def _run_knot_detection(self):
        return self.detector.knot
    
    def _get_knot_confidence(self):
        return self.knot[-1]['confidence'] * self.knot[0]['confidence']

    # return the first undercrossing (after undercrossing at start of knot), None if no next undercrossing within the trace
    def _get_prev_under_crossing_after_knot(self):
        if self.knot is None:
            raise Exception('No knot found so cannot detect undercrossing inside it')
    
        for crossing in self.detector.crossings:
            if crossing['pixels_idx'] <= self.knot[0]['pixels_idx']:
                continue
            if crossing['ID'] == 0:
                self.under_crossing_before_knot = crossing
                break
    
    # return the first undercrossing (after end of knot), None if no next undercrossing within the trace
    def _get_next_under_crossing_after_knot(self):
        if self.knot is None:
            raise Exception('No knot found so cannot detect undercrossing after it')

        for crossing in self.detector.crossings:
            if crossing['pixels_idx'] <= self.knot[-1]['pixels_idx']:
                continue
            if crossing['ID'] == 0:
                self.under_crossing_after_knot = crossing
                break
        
    def _determine_pinch(self, knot=True):
        if knot:
            idx = self.knot[-1]['pixels_idx']
        else:
            idx = float('inf')
            under = float('inf')
            loc = None
            for crossing in self.detector.crossings_stack:
                if crossing['ID'] == 0 and under == float('inf'):
                    under = crossing['pixels_idx']
                    loc = crossing['loc']
                if loc != None:
                    same_crossing = np.linalg.norm(np.array([loc[0], loc[1]]) - np.array([crossing['loc'][0], crossing['loc'][1]])) <= self.detector.eps
                    if crossing['ID'] == 1 and under != float('inf') and same_crossing:
                        idx = crossing['pixels_idx']
                        break
            if idx == float('inf'):
                return None, None, None

        pinch = self._get_pixel_at(idx)

        prev_under = None
        if knot:
            # prev_under = self.under_crossing_before_knot #should be an index in trace
            # print("Knot prev under: ", prev_under)
            # prev_under = prev_under['pixels_idx']
            for i in range(-2, -len(self.knot), -1):
                if self.knot[i]['ID'] == 0:
                    prev_under = self.knot[i]['pixels_idx']
                    break
        else:
            for crossing in self.detector.crossings_stack:
                if crossing['pixels_idx'] <= idx:
                    continue
                if crossing['ID'] == 0:
                    prev_under = crossing['pixels_idx']
                    break

        if prev_under == None:
            return None, None, None

        pinch_idx = idx
        hit_under = False
        points_explored = {}
        while not hit_under: 
            pinch = self._get_pixel_at(idx)
            graspability = self.graspability.find_pixel_point_graspability(pinch, self.img)
            points_explored[idx] = graspability
            idx -= 1
            hit_under = idx <= prev_under

        hit_under = False
        idx = pinch_idx - 1
        next_under = None
        if knot and self.under_crossing_after_knot:
            next_under = self.under_crossing_after_knot['pixels_idx']
            print("Got pinch next under: ", next_under)
            # for i in range(-2, -len(self.knot), -1):
            #     if self.knot[i]['ID'] == 0:
            #         prev_under = self.knot[i]['pixels_idx']
            #         break
            # for crossing in self.detector.crossings_stack:
            #     if crossing['pixels_idx'] <= idx:
            #         continue
            #     if crossing['ID'] == 0:
            #         next_under = crossing['pixels_idx']
            #         break
        else:
            next_under = None

        if next_under == None:
            hit_under = True

        while not hit_under:
            pinch = self._get_pixel_at(idx)
            graspability = self.graspability.find_pixel_point_graspability(pinch, self.img)
            points_explored[idx] = graspability
            idx += 1
            hit_under = idx >= next_under

        min_graspability = float('inf')
        min_grasp_idx = -1
        for key in points_explored:
            # print("pinch graspabilities: ", points_explored[key])
            if points_explored[key] < min_graspability:
                min_graspability = points_explored[key]
                min_grasp_idx = key

        min_grasp_idx = prev_under if min_graspability > 100 else min_grasp_idx
        print("min graspability: ", min_graspability)
        while min_graspability > 100:
            min_grasp_idx -= 1
            pinch = self._get_pixel_at(min_grasp_idx)
            min_graspability = self.graspability.find_pixel_point_graspability(pinch, self.img)
            print("min_graspability: ", min_graspability)

        pinch = self._get_pixel_at(min_grasp_idx)
        print('Graspable pinch:', pinch)
        return points_explored, pinch, min_grasp_idx

    def _determine_cage(self, pinch_idx, knot=True):
        # go back until you're at the trace part that corresponds to overcrossing
        if knot:
            idx = self.knot[0]['pixels_idx'] + 1
        else:
            idx = float('inf')
            for crossing in self.detector.crossings_stack:
                if crossing['ID'] == 0:
                    idx = crossing['pixels_idx'] + 1
                    break
            if idx == float('inf'):
                return None, None, None

        next_under = None
        if knot and self.under_crossing_before_knot:
            next_under = self.under_crossing_before_knot['pixels_idx']
        else:
            for crossing in self.detector.crossings_stack:
                if crossing['pixels_idx'] <= idx:
                    continue
                if crossing['ID'] == 0:
                    next_under = crossing['pixels_idx']
                    break

        if next_under == None:
            return None, None, None
    
        hit_under = False
        points_explored = {}
        # then trace from there forward and stop once you're at the next undercrossing
        while not hit_under:
            cage = self._get_pixel_at(idx)
            graspability = self.graspability.find_pixel_point_graspability(cage, self.img)
            points_explored[idx] = graspability
            idx += 1
            hit_under = idx >= next_under

        # far_from_crossing = {}
        # for key in points_explored:
        #     if key not in self.detector.crossings:
        #         far_from_crossing[key] = points_explored[key]

        min_graspability = float('inf')
        min_grasp_idx = -1
        for key in points_explored:
            # print("cage graspabilities: ", points_explored[key])
            if points_explored[key] < min_graspability:
                min_graspability = points_explored[key]
                min_grasp_idx = key

        min_grasp_idx = next_under if min_graspability > 100 else min_grasp_idx
        cage = self._get_pixel_at(min_grasp_idx)
        if pinch_idx is not None:
            start = np.min([min_grasp_idx, pinch_idx])
            end = np.max([min_grasp_idx, pinch_idx])
            dist = self.tracer.get_dist_cumsum(self.pixels[start:end])
        while min_graspability > 100 and ((pinch_idx is not None and dist > 0.04) or pinch_idx is None):
            min_grasp_idx += 1
            cage = self._get_pixel_at(min_grasp_idx)
            min_graspability = self.graspability.find_pixel_point_graspability(cage, self.img)
            if pinch_idx is not None:
                start = np.min([min_grasp_idx, pinch_idx])
                end = np.max([min_grasp_idx, pinch_idx])
                dist = self.tracer.get_dist_cumsum(self.pixels[start:end])

        cage = self._get_pixel_at(min_grasp_idx)
        print('Graspable cage: ', cage)
        return points_explored, cage, min_grasp_idx

    def interpolate_trace(self, pixels):
        k = pixels.shape[0] - 1 if pixels.shape[0] < 4 else 3
        x = pixels[:, 1]
        y = pixels[:, 0]
        tck, u = interpolate.splprep([x, y], s=0, per=False, k=k)
        xnew, ynew = interpolate.splev(np.linspace(0, 1, len(x)*3), tck)
        xnew = np.array(xnew, dtype=int)
        ynew = np.array(ynew, dtype=int)

        # print('image shape', self.img.shape, 'xnew length', xnew.shape, xnew, ynew)

        x_in = np.where(xnew < self.img.shape[1])
        xnew = xnew[x_in[0]]
        ynew = ynew[x_in[0]]
        x_in = np.where(xnew >= 0)
        xnew = xnew[x_in[0]]
        ynew = ynew[x_in[0]]
        y_in = np.where(ynew < self.img.shape[0])
        xnew = xnew[y_in[0]]
        ynew = ynew[y_in[0]]
        y_in = np.where(ynew >= 0)
        xnew = xnew[y_in[0]]
        ynew = ynew[y_in[0]]
        
        return_val = np.vstack((ynew.T,xnew.T)).T
        # print('RETURNING FROM INTERPOLATE TRACE', return_val)
        
        return return_val
    
    # return crossing(s) if new one(s) is formed from this uon detection, else None

    def _process_uon(self, uon, prob, center_pixel, model_step, first_step=False):
        crossings = []
        if uon != 2:
                self.local_crossing_stream.append({'center_pixel': center_pixel, 'uon': uon, 'prob': prob, 'pixels_idx': model_step})
            
        elif uon == 2 and len(self.local_crossing_stream) > 0:
            if len(self.local_crossing_stream) == 1 and first_step == False:
                # single under / over crossing - ignore and proceed
                self.local_crossing_stream = []
            else:
                # two under / over crossings (>2 WIP?)
                next_crossing_stream = []
                if len(self.local_crossing_stream) > 4:
                    crossing_border = (len(self.local_crossing_stream) + 1) // 2
                    next_crossing_stream = self.local_crossing_stream[crossing_border:]
                    self.local_crossing_stream = self.local_crossing_stream[:crossing_border]
                crossing = self._vote_and_process_under_over_crossing()
                crossings.append(crossing)
            
                # process second crossing, if it exists
                if next_crossing_stream:
                    self.local_crossing_stream = next_crossing_stream
                    crossing = self._vote_and_process_under_over_crossing()
                    crossings.append(crossing)
               
                self.local_crossing_stream = []

        return crossings

    def _get_crossing_locs(self, viz=False):
        line_segments = []

        for i in range(len(self.pixels) - 1):
            curr_pixel, next_pixel = self.pixels[i], self.pixels[i + 1] 
            line_segments.append(LineString([curr_pixel, next_pixel]))

        crossing_locs = []
        i = 0
        while i < len(line_segments):
            current_line_seg = line_segments[i]
            for j in chain(range(max(0, i - self.num_steps_min_for_crossing)), range(i + self.num_steps_min_for_crossing + 1, len(line_segments))):
                other_line_seg = line_segments[j]
                if current_line_seg.intersects(other_line_seg):
                    center_point = current_line_seg.intersection(other_line_seg)
                    if "POINT" in str(center_point):
                        center_pixel, pixels_idx = (int(center_point.x), int(center_point.y)), i
                        crossing_locs.append((center_pixel, pixels_idx))
                        i += self.num_steps_min_for_crossing
                        break
            i += 1
        
        if viz:
            img = self.img.copy()
            clr = (255, 0, 0)
            ctr = 0
            for crossing_loc in crossing_locs:
                y, x = crossing_loc[0]
                cv2.circle(img, (x, y), 3, clr, -1)
                cv2.putText(img, str(ctr), (x + 2, y + 2), cv2.FONT_HERSHEY_PLAIN, 1, clr)
                ctr += 1
            cv2.imwrite(self.output_vis_dir + 'all_crossings' + '.png', img)

        return crossing_locs

    def trace_and_detect_knot(self, endpoints=None):
        # import pdb; pdb.set_trace()
        self.pixels, self.trace_end = self.tracer.trace(self.img, self.starting_pixels_for_trace, endpoints=endpoints, viz=True, path_len=500)
        self.pixels = self.interpolate_trace(self.pixels)
        self.crossing_locs = self._get_crossing_locs()

        for i, crossing_loc in enumerate(self.crossing_locs):
            center_pixel, pixels_idx = crossing_loc
            uo_data = {}
            crop = self._crop_img(self.img, center_pixel, self.crop_size)
            
            if crop is None:
                print('HIT IMAGE EDGE')
                knot_output = self._run_knot_detection()
                # check if that new crossing being added to sequence creates a knot
                if knot_output:
                    print('FOUND KNOT')
                    self.knot = knot_output
                    self.last_trace_step_in_knot = self.knot[-1]['pixels_idx']
                    self._get_prev_under_crossing_after_knot()
                    self._get_next_under_crossing_after_knot()
                return
            
            uo_data['crop_img'] = crop
            spline_pixels = self._get_spline_pixels(pixels_idx, self.crop_size)
            
            # if at the very start of the trace
            if spline_pixels is None:
                continue
            uo_data['spline_pixels'] = spline_pixels

            # get input to UO classifier
            uo_model_input = self._getuoitem(uo_data)

            # predict UON on input
            # uo, prob = self._predict_uo(uo_model_input, file_name = f'uo_{pixels_idx}.png')
            uo, prob = self._predict_uo(uo_model_input)

            # add UON to stream and process stream                
            crossing = {'loc': center_pixel, 'ID': uo, 'confidence': prob, 'pixels_idx': pixels_idx, 'crossing_idx': i}
            
            # add crossing
            self._add_crossing(crossing)

        knot_output = self._run_knot_detection()
        # check if a knot is found
        if knot_output:
            print('FOUND KNOT')
            self.knot = knot_output
            self.last_trace_step_in_knot = self.knot[-1]['pixels_idx']
            self._get_prev_under_crossing_after_knot()
            self._get_next_under_crossing_after_knot()
            return
            
        # first_step = True
        # for model_step in range(len(self.pixels)):
        #     center_pixel = self._get_pixel_at(model_step)
        #     # generate a 20 x 20 crop around the pixel and get spline pixels
        #     uon_data = {}
        #     crop = self._crop_img(self.img, center_pixel, self.crop_size)
        #     if crop is None:
        #         print('HIT IMAGE EDGE')
        #         for crossing in crossings:
        #             knot_output = self._add_crossing_and_run_knot_detection(crossing)
        #             # check if that new crossing being added to sequence creates a
        #             if knot_output:
        #                 print('FOUND KNOT')
        #                 self.knot = knot_output
        #         return

        #     uon_data['crop_img'] = crop
        #     spline_pixels = self._get_spline_pixels(model_step, self.crop_size)
          
        #     # at the very start of the trace
        #     if(spline_pixels is None):
        #         continue
        #     uon_data['spline_pixels'] = spline_pixels

        #     # self._visualize(uon_data['crop_img'], f'uon_{model_step}_p.png')

        #     # get input to UON classifier
        #     uon_model_input = self._getuonitem(uon_data)
        #     self._visualize_tensor(uon_model_input, f'uon_{model_step}.png')

        #     # predict UON on input
        #     uon, prob = self._predict_uon(uon_model_input, file_name = f'uo_{model_step}.png')

        #     # add UON to stream and process stream
        #     crossings = self._process_uon(uon, prob, center_pixel, model_step, first_step)

        #     for crossing in crossings:
        #         knot_output = self._add_crossing_and_run_knot_detection(crossing)
        #         # check if that new crossing being added to sequence creates a
        #         if knot_output:
        #             print('FOUND KNOT')
        #             self.knot = knot_output
        #             return

        #     self.last_trace_step_in_knot = model_step

        #     if uon == 2:
        #         first_step = False
    
    def perception_pipeline(self, endpoints=None, idx=0, viz=False):
        self.vis_idx += 1
        self.trace_and_detect_knot(endpoints=endpoints)
        if viz:
            print('Visualizing and dumping.')
            self._visualize(self.img, f'full_img_{self.vis_idx}.png')
            self._visualize_full()
            if self.knot:
                self._visualize_knot()
            self._visualize_all_crossings()
        if not self.knot:
            print('No knots!')
            done_untangling = True
            pinches, pinch, pinch_idx = self._determine_pinch(knot=False)
            cages, cage, cage_idx = self._determine_cage(pinch_idx, knot=False)
            knot_confidence = None
        else:
            done_untangling = False
            pinches, pinch, pinch_idx = self._determine_pinch()
            cages, cage, cage_idx = self._determine_cage(pinch_idx)
            knot_confidence = self._get_knot_confidence()
        pinch_viz = [0,0] if pinch is None else pinch
        cage_viz = [0,0] if cage is None else cage
        pinches = {0:0} if pinches is None else pinches
        cages = {0:0} if cages is None else cages
        if viz:
            self._visualize_cage_pinch(cage_viz, pinch_viz, idx=idx)
            self._visualize_all_cages_pinches(cages, pinches, idx=idx)
        pull_apart_dist = self.tracer.get_dist_cumsum(self.pixels[:cage_idx])
        reveal_point = None
        if self.trace_end == TraceEnd.EDGE:
            reveal_point = self.pixels[-1]
        output = {}
        print('Pinch: ', pinch)
        print('Cage: ', cage)
        output['pinch'] = pinch
        output['cage'] = cage
        output['knot_confidence'] = knot_confidence
        output['pull_apart_dist'] = pull_apart_dist
        output['done_untangling'] = done_untangling
        output['trace_end'] = self.trace_end
        output['reveal_point'] = reveal_point
        return output
            
if __name__ == '__main__':
    # parse command line flags
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_index', type=str, default='')
    parser.add_argument('--parallel', action='store_true', default=False)

    flags = parser.parse_args()
    data_index = flags.data_index 
    parallel = flags.parallel

    if data_index == '':
        data_folder = '/home/vainavi/hulk-keypoints/real_data/real_data_for_tracer/test'
        tkd = TracerKnotDetector(parallel=parallel)
        for i, f in enumerate(np.sort(os.listdir(data_folder))):
            if i < 100:
                continue
            data_path = os.path.join(data_folder, f)
            test_data = np.load(data_path, allow_pickle=True).item()
            tkd._set_data(test_data['img'], np.flip(test_data['pixels'][:10], axis=1))
            print(data_path)
            print()
            try:
                tkd.perception_pipeline(viz=True, idx=i)
            except Exception as e:
                if 'Not enough starting points' in str(e):
                    continue
                else:
                    raise e
            tkd._visualize_full()
            tkd._visualize_all_crossings()
            if tkd.knot:
                print()
                print(tkd.knot)
                print(tkd._get_knot_confidence())
                tkd._visualize_knot()
    else:
        data_path = f'/home/vainavi/hulk-keypoints/real_data/real_data_for_tracer/test/{data_index}.npy'
        test_data = np.load(data_path, allow_pickle=True).item()
        tkd = TracerKnotDetector(parallel=parallel)
        tkd._set_data(test_data['img'], np.flip(test_data['pixels'][:10], axis=1))
        print(data_path)
        print()
        tkd.perception_pipeline(viz=True)
        tkd._visualize(test_data['img'], 'full_img.png')
        tkd._visualize_full()
        tkd._visualize_all_crossings()
        if tkd.knot:
            print()
            print(tkd.knot)
            print(tkd._get_knot_confidence())
            tkd._visualize_knot()
