from knot_detection import KnotDetector
import numpy as np
import cv2
from src.graspability import Graspability

class TracerKnotDetector():
    def __init__(self, test_data):
        self.img = test_data['img']
        self.pixels = test_data['pixels']
        self.pixels_so_far = []
        self.output_vis_dir = '/home/jainilajmera/hulk-keypoints/test_tkd/'
        
        self.uon_crop_size = 20
        self.uon_augment = True
        self.uon_sweep = True
        self.uon_real_world = True

        self.graspability = Graspability()
        self.knot_detector = KnotDetector()

    def _getuonitem(self, uon_data):
        uon_img = uon_data['crop_img'][:, :, :3].copy()
        condition_pixels = np.array(uon_data['spline_pixels'], dtype=np.float64)
        if uon_img.max() > 1:
            uon_img = (uon_img / 255.0).astype(np.float32)
        cable_mask = np.ones(uon_img.shape[:2])
        cable_mask[uon_img[:, :, 1] < 0.35] = 0
        if self.uon_augment:
            uon_img = self.call_img_transform(uon_img)
        if self.uon_sweep:
            uon_img[:, :, 0] = self.draw_spline(uon_img, condition_pixels[:, 1], condition_pixels[:, 0], label=True)
        else:
            uon_img[:, :, 0] = gauss_2d_batch_efficient_np(self.crop_span, self.crop_span, self.gauss_sigma, condition_pixels[:-self.pred_len,0], condition_pixels[:-self.pred_len,1], weights=self.weights)
        if self.uon_real_world:
            uon_img = uon_img[1:, 1:, :]
            uon_imgimg = cv2.resize(uon_img, (2 * self.crop_width, 2 * self.crop_width))
        uon_img, _= self.rotate_condition(uon_img, condition_pixels, center_around_last=True, index=data_index)
        uon_model_input = transform(uon_img.copy()).cuda()
        return uon_model_input

    def _visualize(self):
        file_name = 'full_img'
        clr = (255, 0, 0)
        for x, y in self.pixels:
            cv2.circle(self.img, (x, y), 3, clr, -1)
        cv2.imwrite(self.output_vis_dir + file_name + '.png', self.img)

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
        if step not in range(len(self.pixels_so_far)):
            raise Exception('Index not in range!')
        return self.pixels_so_far[idx]

    def _get_buffer_pixels(self, center_pixel, latest_step, crop_size):
        if latest_step not in range(len(self.pixels)):
            return
        latest_trace_pixel = self._get_pixel_at(latest_step)
        while np.linalg.norm(latest_trace_pixel - center_pixel, ord=np.inf) <= crop_size // 2:
            if latest_step not in range(len(self.pixels_so_far)):
                self.pixels_so_far.append(latest_trace_pixel)
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
        for model_step in range(len(self.pixels)):
            # have not reached model step in trace yet
            if model_step not in range(len(self.pixels_so_far)):
                self.pixels_so_far.append(self._get_pixel_at(model_step))
            
            center_pixel = self.pixels_so_far[model_step]            
            # trace a little extra (buffer) to get pixels for conditioning
            self._get_buffer_pixels(center_pixel, model_step + 1, self.uon_crop_size)
            
            # generate a 20 x 20 crop around the pixel
            uon_data = {}
            uon_data['crop_img'] = self._crop_img(self.img, center_pixel, self.uon_crop_size)
            uon_data['spline_pixels'] = self._get_spline_pixels(model_step, self.uon_crop_size)

            # get input to UON classifier
            uon_model_input = self._getuonitem(uon_data)

            # keep tracing till you;re within 20 x 20 at current pixel - then stop and feed that crop in 

if __name__ == '__main__':
    test_data = np.load("/home/vainavi/hulk-keypoints/real_data/real_data_for_tracer/test/00000.npy", allow_pickle=True).item()
    tkd = TracerKnotDetector(test_data)
    tkd.trace_and_detect_knot()









