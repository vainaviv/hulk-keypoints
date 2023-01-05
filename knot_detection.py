import pickle
import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

class KnotDetector:
    def __init__(self) -> None:
        self.crossings_stack = []
        self.eps = 3.0

    def encounter_seg(self, crossing) -> None:
        # crossing must be in following format: {'loc': (x,y), 'ID': 0}
        # ID options: 0 (under), 1 (over)
        prev_crossing = self.crossings_stack.pop()
        # check if previous location is within delta of current location
        prev_x, prev_y = prev_crossing['loc']
        curr_x, curr_y = crossing['loc']
        if abs(prev_x - curr_x) <= self.eps and abs(prev_y - curr_y) <= self.eps:
            return 0
        else:
            self.crossings_stack.append(prev_crossing)
            self.crossings_stack.append(crossing)
            if self.encountered_knot():
                return 1
            return 0

    def encountered_knot(self) -> bool:
        # not a true knot definition atm. A slip knot and a twist would 
        # currently be identified as a knot
        if len(self.crossings_stack) < 3:
            return False
        first = self.crossings_stack[-3]['ID']
        second = self.crossings_stack[-2]['ID']
        third = self.crossings_stack[-1]['ID']
        if first != None and second != None and third != None \
            and (first != second) and (second != third):
            return True
        else:
            return False

    def determine_under_over(self, img):
        mask = np.ones(img.shape[:2])
        mask[img[:, :, 0] <= 100] = 0
        # kernel = np.ones((2, 2), np.uint8)/4
        # smooth = cv2.filter2D(img,-1,kernel)
        # erode_mask = cv2.erode(mask, kernel)
        # plt.imshow(smooth)
        # plt.savefig('smooth.png')

if __name__ == '__main__':
    detector = KnotDetector()
    img_path = 'eval_imgs/00001.png'
    img = cv2.imread(img_path)
    detector.determine_under_over(img)
