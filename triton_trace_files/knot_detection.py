import pickle
import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

class KnotDetector:
    def __init__(self) -> None:
        self.crossings_stack = []
        self.crossings = []
        self.eps = 10
        self.knot = []
        self.start_idx = float('inf')
        
    def _reset(self):
        self.crossings_stack = []
        self.crossings = []
        self.knot = []
        self.start_idx = float('inf')

    def _correct_crossings(self):
        seg = self.crossings[-1]
        curr_x, curr_y = seg['loc']
        curr_id = seg['ID']
        curr_confidence = seg['confidence']

        for prev_idx in range(len(self.crossings) - 1):
            prev_crossing = self.crossings[prev_idx]
            prev_x, prev_y = prev_crossing['loc']
            prev_id = prev_crossing['ID']
            if np.linalg.norm(np.array([curr_x, curr_y]) - np.array([prev_x, prev_y])) <= self.eps:
                if prev_id == curr_id:
                    print("Crossing correction at: ", prev_x, prev_y, ". Originally: ", prev_id)
                    prev_confidence = prev_crossing['confidence']
                    if curr_confidence >= prev_confidence:
                        prev_crossing['ID'] = 1 - prev_id
                        prev_crossing['confidence'] = seg['confidence']
                    else:
                        seg['ID'] = 1 - curr_id
                        seg['confidence'] = prev_crossing['confidence']

    def encounter_seg(self, seg):
        '''
        Called every time a new segment is encountered.
        Returns a list of crossings that constitute the knot if knot is encountered at current segment. 
        Else, returns None.
        '''
        # seg must be in following format: {'loc': (x, y), 'ID': 0/1/2, 'confidence': [0, 1]}
        # ID options: 0 (under), 1 (over), 2 (not a crossing)
        # skip if not a crossing

        if seg['ID'] == 2:
            return
        
        self.crossings.append(seg)
        self._correct_crossings()

        if not self.crossings_stack:
            curr_x, curr_y = seg['loc']
            self.crossings_stack.append(seg)
            return
        
        # simplify if same crossing is immediately re-encountered (over -> under or under -> over)
        # TODO: check popping
        prev_crossing = self.crossings_stack.pop()
        prev_x, prev_y = prev_crossing['loc']
        curr_x, curr_y = seg['loc']
        prev_id, curr_id = prev_crossing['ID'], seg['ID']

        if np.linalg.norm(np.array([curr_x, curr_y]) - np.array([prev_x, prev_y])) <= self.eps:
            return

        self.crossings_stack.append(prev_crossing)
        self.crossings_stack.append(seg)

        if self.is_knot_encountered():
            return self.knot

    def is_knot_encountered(self) -> bool:
        '''
        Checks if the latest crossing results in a knot.
        Only accounts for trivial loops.
        '''
        # no knot encountered if < 3 crossings have been seen (?)
        # if len(self.crossings_stack) < 3:
        #     return False
            
        crossing = self.crossings_stack[-1]
        pos = self.get_crossing_pos(crossing)
        # no knot encountered if crossing hasn't been seen before
        if pos == -1:
            return False

        # no knot encountered if O...U (?)
        if self.crossings_stack[pos]['ID'] == 1 and self.crossings_stack[-1]['ID'] == 0:
            return False

        # intermediate crossing = crossing in between start and end crossing (exclusive)
        # no knot encountered if every intermediate crossing is an undercrossing
        if all([intermediate_crossing['ID'] == 0 for intermediate_crossing in self.crossings_stack[pos + 1:-1]]):
            return False
                    
        start_idx = self.crossings_stack[pos]['crossing_idx']
        if not self.knot or start_idx < self.start_idx:
            end_idx = self.crossings_stack[-1]['crossing_idx']
            self.knot = self.crossings[start_idx:end_idx + 1]
            self.start_idx = start_idx
            self.crossings_stack = self.crossings_stack[:pos]

        return True

    def get_crossing_pos(self, crossing) -> int:
        '''
        Returns the index of previous sighting on the stack.
        If crossing has not been seen previously, returns -1.
        '''
        curr_x, curr_y = crossing['loc']
        curr_id = crossing['ID']
        curr_crossing_idx = crossing['crossing_idx']

        # print(crossing)
        # print(self.crossings_stack)
        # print()

        # only look at crossings prior to most recently added crossing
        for pos in range(len(self.crossings_stack) - 1):
            prev_crossing = self.crossings_stack[pos]
            prev_x, prev_y = prev_crossing['loc']
            prev_id = prev_crossing['ID']
            prev_crossing_idx = prev_crossing['crossing_idx']
            if np.linalg.norm(np.array([curr_x, curr_y]) - np.array([prev_x, prev_y])) <= self.eps:
                if prev_id == curr_id:
                    raise Exception('Crossing not corrected!')
                return pos
        
        return -1

    def detect_knot(self, segs):
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

# make sure O...U is not counted as a knot - same as other input with swapped U and O
def test_knot_detector_edge_case(detector):
    segs = [
        {'loc': (0, 0), 'ID': 1}, # O at P0
        {'loc': (10, 9), 'ID': 1}, # O at P1
        {'loc': (20, 21), 'ID': 1}, # O at P2
        {'loc': (30, 29), 'ID': 0}, # U at P3
        {'loc': (100, 100), 'ID': 2}, # NC
        {'loc': (40, 39), 'ID': 1}, # O at P4
        {'loc': (21, 20), 'ID': 0}, # U at P2
        {'loc': (29, 30), 'ID': 1}, # O at P3
        {'loc': (41, 40), 'ID': 0}, # U at P4
        {'loc': (9, 10), 'ID': 0} # U at P1
    ]
    assert detector.detect_knot(segs) == [{'loc': (30, 29), 'ID': 0}, {'loc': (40, 39), 'ID': 1}, {'loc': (21, 20), 'ID': 0}, {'loc': (29, 30), 'ID': 1}]

if __name__ == '__main__':
    detector = KnotDetector()
    test_knot_detector_basic(detector)

    detector2 = KnotDetector()
    test_knot_detector_edge_case(detector2)