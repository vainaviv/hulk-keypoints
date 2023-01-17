import numpy as np
import cv2
import time
import shutil
import os
import sys
import argparse
import torch
from torchvision import transforms, utils

from knot_detection import KnotDetector
from src.graspability import Graspability
from src.dataset import KeypointsDataset
from src.model import ClassificationModel, KeypointsGauss
from src.prediction import Prediction
from config import *
from tracer import Tracer


class TracerKnotDetector():
    def __init__(self, test_data, parallel=True):
        self.img = test_data['img']
        self.pixels = test_data['pixels']
        self.starting_pix = 10
        self.pixels_so_far = self.pixels[:self.starting_pix]
        self.parallel = parallel
        self.output_vis_dir = './test_tkd/'
        if os.path.exists(self.output_vis_dir):
            shutil.rmtree(self.output_vis_dir)
        os.makedirs(self.output_vis_dir)
        
        self.graspability = Graspability()
        self.local_crossing_stream = []
        self.detector = KnotDetector()
        self.knot = None

        self.uon_crop_size = 20
        self.uon_config = UNDER_OVER_NONE()
        self.uon_kpts = KeypointsDataset('',
                                    transform=transforms.Compose([transforms.ToTensor()]), 
                                    augment=True, 
                                    config=self.uon_config)
        self.uon_model = ClassificationModel(num_classes=self.uon_config.classes, img_height=self.uon_config.img_height, img_width=self.uon_config.img_width, channels=3)
        self.uon_model.load_state_dict(torch.load('/home/mkparu/hulk-keypoints/checkpoints/2023-01-11-02-15-52_UNDER_OVER_NONE_all_crossings_regen_test/model_11_0.12860.pth'))

        self.uo_config = UNDER_OVER_RNet50()
        self.uo_model = ClassificationModel(num_classes=self.uo_config.classes, img_height=self.uo_config.img_height, img_width=self.uo_config.img_width, channels=3)
        self.uo_model.load_state_dict(torch.load('/home/vainavi/hulk-keypoints/checkpoints/2023-01-17-00-57-01_UNDER_OVER_RNet34_lr1e5_medley_03Hard2/model_7_0.27108.pth'))

        self.tracer = Tracer()

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
        img = self.img.copy()
        file_name = 'full_img'
        cv2.imwrite(self.output_vis_dir + file_name + '.png', img)
        clr = (255, 0, 0)
        for x, y in self.pixels:
            cv2.circle(img, (x, y), 3, clr, -1)
        cv2.imwrite(self.output_vis_dir + file_name + '_with_trace' + '.png', img)

    def _visualize_knot(self):
        if not self.knot:
            raise Exception('No knot detected for visualization!')
        img = self.img.copy()
        file_name = 'knot_img'
        #red for over crossing, blue for under (colors flipped bc cv2)
        u_clr = (255, 0, 0)
        o_clr = (0, 0, 255)
        ctr = 0
        for crossing in self.knot:
            ctr += 1
            x, y = crossing['loc']
            if crossing['ID'] == 0:
                cv2.circle(img, (x, y), 3, u_clr, -1)
                cv2.putText(img, str(ctr), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, u_clr)
            if crossing['ID'] == 1:
                cv2.circle(img, (x, y), 3, o_clr, -1)
                cv2.putText(img, str(ctr), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, o_clr)
        cv2.imwrite(self.output_vis_dir + file_name + '.png', img)

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
                self.pixels_so_far = np.append(self.pixels_so_far, np.array([latest_trace_pixel]), axis=0) #TODO: a little confused what this is doing
            latest_step += 1
            if latest_step not in range(len(self.pixels)):
                return
            latest_trace_pixel = self._get_pixel_at(latest_step)
        print("pixels so far length after getting buffer pixels: ", len(self.pixels_so_far))

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

    def _predict_uon(self, uon_model_input):
        predictor = Prediction(self.uon_model, self.uon_config.num_keypoints, self.uon_config.img_height, self.uon_config.img_width, parallelize=self.parallel)
        prediction_prob_arr = predictor.predict(uon_model_input).cpu().detach().numpy().squeeze()
        pred = np.argmax(prediction_prob_arr)
        prediction_prob = prediction_prob_arr[pred]
        # calls separate model for under / over
        if pred != 2:
            uo_model_input = uon_model_input
            predictor = Prediction(self.uo_model, self.uo_config.num_keypoints, self.uo_config.img_height, self.uo_config.img_width, parallelize=self.parallel)
            updated_prediction_prob = predictor.predict(uo_model_input).cpu().detach().numpy().squeeze()
            if updated_prediction_prob >= 0.5:
                return 1, updated_prediction_prob
            else: 
                return 0, 1 - updated_prediction_prob
        else:
            return pred, prediction_prob
    
    def _vote_and_process_under_over_crossing(self):
        # using 1, -1 instead of 1, 0 so the confidence matters for U as well
        x_arr = []
        y_arr = []
        weighted_sum = 0
        for crossing_dict in self.local_crossing_stream:
            if crossing_dict['uon'] == 0:
                weighted_sum -= crossing_dict['prob']
            else:
                weighted_sum += crossing_dict['prob']
            x_arr.append(crossing_dict['center_pixel'][0])
            y_arr.append(crossing_dict['center_pixel'][1])
        
        x_arr, y_arr = np.array(x_arr), np.array(y_arr)
        avg_x, avg_y = int(np.mean(x_arr)), int(np.mean(y_arr))
        med_x, med_y = int(np.median(x_arr)), int(np.median(y_arr))
        weighted_sum = weighted_sum / len(self.local_crossing_stream)

        if weighted_sum >= 0:
            return self.detector.encounter_seg({'loc': (avg_x, avg_y), 'ID': 1, 'confidence': weighted_sum})
        else:
            return self.detector.encounter_seg({'loc': (avg_x, avg_y), 'ID': 0, 'confidence': -weighted_sum})

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
        idx = self.detector.get_crossing_pos(self.pixels_so_far[-1])
        cage = self._get_pixel_so_far_at(idx)
        # then trace from there forward and stop once you're in a graspable region
        while not self.graspability.find_pixel_point_graspability(cage, self.pixels_so_far):
            idx += 1
            cage = self._get_pixel_so_far_at(idx)
        return cage

    def trace_and_detect_knot(self):
        # go pixel wise 
        first_step = True
        path_len = 10
        for model_step in range(self.starting_pix-1, len(self.pixels), path_len): #every 10 steps collect more trace
            # have not reached model step in trace yet
            if model_step not in range(len(self.pixels_so_far)):
                spline = self.tracer._trace(self.img, self.pixels_so_far, path_len=path_len, viz=True) #turn off viz later
                self.pixels_so_far = np.append(self.pixels_so_far, spline, axis=0)
                print(len(self.pixels_so_far))
                # self.pixels_so_far = np.append(self.pixels_so_far, np.array([self._get_pixel_at(model_step)]), axis=0)
            
            print(model_step)
            center_pixel = self._get_pixel_so_far_at(max(model_step, len(self.pixels_so_far)-1))
            # trace a little extra (buffer) to get pixels for conditioning
            self._get_buffer_pixels(center_pixel, model_step + 1, self.uon_crop_size)
            
            # generate a 20 x 20 crop around the pixel
            uon_data = {}
            uon_data['crop_img'] = self._crop_img(self.img, center_pixel, self.uon_crop_size)
            uon_data['spline_pixels'] = self._get_spline_pixels(model_step, self.uon_crop_size) #TODO: right now this is None
            self._visualize(uon_data['crop_img'], f'uon_{model_step}_p.png')

            # get input to UON classifier
            uon_model_input = self._getuonitem(uon_data)
            self._visualize_tensor(uon_model_input, f'uon_{model_step}.png')

            # predict UON on input
            uon, prob = self._predict_uon(uon_model_input)
            print(model_step, uon, center_pixel, prob)

            if uon != 2:
                self.local_crossing_stream.append({'center_pixel': center_pixel, 'uon': uon, 'prob': prob})
            
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
                    knot_output = self._vote_and_process_under_over_crossing()
                    if knot_output: 
                        # a knot is detected
                        self.knot = knot_output
                        return
                    # process second crossing, if it exists
                    if next_crossing_stream:
                        self.local_crossing_stream = next_crossing_stream
                        knot_output = self._vote_and_process_under_over_crossing()
                        if knot_output:
                            # a knot is detected
                            self.knot = knot_output
                            return
                    print(self.detector.crossings_stack)
                    self.local_crossing_stream = []
            
            if uon == 2:
                first_step = False
    
    def perception_pipeline(self):
        self.trace_and_detect_knot()
        if not self.knot:
            return -1, -1 # Done untangling!
        pinch = self._determine_pinch()
        cage = self._determine_cage()
        return pinch, cage
            
if __name__ == '__main__':
    # parse command line flags
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_index', type=str, default='')
    parser.add_argument('--parallel', action='store_true', default=False)

    flags = parser.parse_args()
    data_index = flags.data_index 
    parallel = flags.parallel

    if data_index == '':
        raise Exception('Please provide the file number (e.g. --data_index 00000) as a command-line argument!')

    data_path = f"/home/vainavi/hulk-keypoints/real_data/real_data_for_tracer/test/{data_index}.npy"
    test_data = np.load(data_path, allow_pickle=True).item()
    tkd = TracerKnotDetector(test_data, parallel=parallel)
    print(data_path)
    print()
    tkd._visualize_full()
    tkd.trace_and_detect_knot()
    if tkd.knot:
        print()
        print(tkd.knot)
        tkd._visualize_knot()









