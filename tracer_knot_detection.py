from knot_detection import KnotDetector
import numpy as np
import cv2
from graspability import Graspability

class TracerKnotDetector():
    def __init__(self, test_data):
        self.img = test_data['img']
        self.pixels = test_data['pixels']
        self.pixels_so_far = []
        self.output_vis_dir = '/home/jainilajmera/hulk-keypoints/test_tkd/'
        self.uon_crop_size = 20
        self.graspability = Graspability()

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
            raise Exception('Step (index) not in range!')
        return self.pixels[step]

    def _get_pixel_so_far_at(self, step):
        if step not in range(len(self.pixels_so_far)):
            raise Exception('Step (index) not in range!')
        return self.pixels_so_far[step]

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

    def _get_spline_pixels(self, center_step, crop_size):
        # add all spline pixels before and after the crossing pixel that are within the crop size
        spline_pixels = []
        center_pixel = self._get_pixel_so_far_at(center_step)
        top_left_pixel = np.array([int(center_pixel[0]) -  crop_size // 2, int(center_pixel[1]) - crop_size // 2])

        for curr_step in range(center_step + 1, len(self.pixels_so_far)):
            if np.linalg.norm(self._get_pixel_so_far_at(curr_step) - center_pixel, ord=np.inf) > crop_size // 2:
                break
            spline_pixels.append(self._get_pixel_so_far_at(curr_step) - top_left_pixel)

        for curr_step in range(center_step, 0, -1):
            if np.linalg.norm(self._get_pixel_so_far_at(curr_step) - center_pixel, ord=np.inf) > crop_size // 2:
                break
            spline_pixels.insert(0, self._get_pixel_so_far_at(curr_step) - top_left_pixel)
        
        if len(spline_pixels) < 2:
            return
    
        return spline_pixels

    def determine_pinch(self):
        idx = -1
        pinch = self.pixels_so_far[idx]
        while not self.graspability.find_pixel_point_graspability(pinch, self.pixels_so_far): #TODO: need to tune this, also need full trace up to this point
            idx -= 1
            pinch = self.pixels_so_far[idx]
        return pinch 

    def determine_cage(self):
        # vainavi TODO: go back until you're at the trace part that corresponds to over crossing
        idx = -1
        # then trace from there forward and stop once you're in a graspable region
        cage = self.pixels_so_far[idx]
        while not self.graspability.find_pixel_point_graspability(cage, self.pixels_so_far):
            idx += 1
            cage = self.pixels_so_far[idx]
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
            self.uon_img = self._crop_img(self.img, center_pixel, self.uon_crop_size)

            print(center_pixel)
            print(self._get_spline_pixels(model_step, self.uon_crop_size))
            print()

            # end as input to UON classifier
            # keep tracing till you;re within 20 x 20 at current pixel - then stop and feed that crop in 

if __name__ == '__main__':
    test_data = np.load("/home/vainavi/hulk-keypoints/real_data/real_data_for_tracer/test/00000.npy", allow_pickle=True).item()
    tkd = TracerKnotDetector(test_data)
    tkd.trace_and_detect_knot()









