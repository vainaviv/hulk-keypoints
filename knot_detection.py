import pickle
import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

class KnotDetector:
    def __init__(self) -> None:
        self.crossings_stack = []
        self.loc_to_crossing = {}
        self.eps = 3.0
        self.knot = []

    def encounter_seg(self, seg) -> list | None:
        '''
        Called every time a new segment is encountered.
        Returns a list of crossings that constitute the knot if knot is encountered at current segment. 
        Else, returns None.
        '''
        # seg must be in following format: {'loc': (x, y), 'ID': 0/1/2}
        # ID options: 0 (under), 1 (over), 2 (not a crossing)
        # skip if not a crossing
        if seg['ID'] == 2:
            return

        if not self.crossings_stack:
            curr_x, curr_y = seg['loc']
            self.loc_to_crossing[(curr_x, curr_y)] = seg
            self.crossings_stack.append(seg)
            return
        
        # simplify if same crossing is immediately re-encountered (over -> under or under -> over)
        prev_crossing = self.crossings_stack.pop()
        prev_x, prev_y = prev_crossing['loc']
        curr_x, curr_y = seg['loc']
        prev_id, curr_id = prev_crossing['ID'], seg['ID']
        self.loc_to_crossing.pop((prev_x, prev_y))

        if abs(curr_x - prev_x) <= self.eps and abs(curr_y - prev_y) <= self.eps and prev_id != curr_id:
            return

        self.loc_to_crossing[(prev_x, prev_y)] = prev_crossing
        self.loc_to_crossing[(curr_x, curr_y)] = seg
        self.crossings_stack.append(prev_crossing)
        self.crossings_stack.append(seg)

        if self.is_knot_encountered():
            return self.knot

    def is_knot_encountered(self) -> bool:
        '''
        Checks if the latest crossing results in a knot.
        Only accounts for trivial loops.
        '''

        # no knot encountered if < 3 crossings have been seen
        if len(self.crossings_stack) < 3:
            return False
            
        crossing = self.crossings_stack[-1]
        pos = self.get_crossing_pos(crossing)
        # no knot encountered if crossing hasn't been seen before
        if pos == -1:
            return False

        self.knot = self.crossings_stack[pos:]
        return True

    def get_crossing_pos(self, crossing) -> int:
        '''
        Returns the index of previous sighting on the stack.
        If crossing has not been seen previously, returns -1.
        '''
        curr_x, curr_y = crossing['loc']
        curr_id = crossing['ID']

        # only look at crossings prior to most recently added crossing
        for pos in range(len(self.crossings_stack) - 1):
            prev_x, prev_y = self.crossings_stack[pos]['loc']
            prev_id = self.crossings_stack[pos]['ID']
            if abs(curr_x - prev_x) <= self.eps and abs(curr_y - prev_y) <= self.eps and prev_id != curr_id:
                return pos
        
        return -1

    def detect_knot(self, segs) -> list | None:
        for seg in segs:
            if self.encounter_seg(seg):
                return self.knot

    def determine_under_over(self, img):
        mask = np.ones(img.shape[:2])
        mask[img[:, :, 0] <= 100] = 0
        # kernel = np.ones((2, 2), np.uint8)/4
        # smooth = cv2.filter2D(img,-1,kernel)
        # erode_mask = cv2.erode(mask, kernel)
        # plt.imshow(smooth)
        # plt.savefig('smooth.png')

def test_knot_detector_basic(detector):
    segs = [
        {'loc': (0, 0), 'ID': 0}, # U at P0
        {'loc': (10, 9), 'ID': 0}, # U at P1
        {'loc': (20, 21), 'ID': 0}, # U at P2
        {'loc': (30, 29), 'ID': 1}, # O at P3
        {'loc': (100, 100), 'ID': 2}, # NC
        {'loc': (40, 39), 'ID': 0}, # U at P4
        {'loc': (21, 20), 'ID': 1}, # O at P2
        {'loc': (29, 30), 'ID': 0}, # U at P3
        {'loc': (41, 40), 'ID': 1}, # O at P4
        {'loc': (9, 10), 'ID': 1} # O at P1
    ]

    assert detector.detect_knot(segs) == [{'loc': (20, 21), 'ID': 0}, {'loc': (30, 29), 'ID': 1}, {'loc': (40, 39), 'ID': 0}, {'loc': (21, 20), 'ID': 1}]


if __name__ == '__main__':
    detector = KnotDetector()
    # img_path = 'eval_imgs/00001.png'
    # img = cv2.imread(img_path)
    # detector.determine_under_over(img)
    test_knot_detector_basic(detector)