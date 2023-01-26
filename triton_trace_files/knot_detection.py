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
        self.eps = 0
        self.knot = []
        self.start_idx = float('inf')
        
    def _reset(self):
        self.crossings_stack = []
        self.crossings = []
        self.knot = []
        self.start_idx = float('inf')

    def correct_all_crossings(self): 
        # correct all crossings in self.crossings (prior to adding to stack)       
        for curr_idx in range(len(self.crossings)): 
            curr_crossing = self.crossings[curr_idx]
            curr_x, curr_y = curr_crossing['loc']
            curr_id = curr_crossing['ID']
            curr_confidence = curr_crossing['confidence']
            for prev_idx in range(curr_idx):
                prev_crossing = self.crossings[prev_idx]
                prev_x, prev_y = prev_crossing['loc']
                prev_id = prev_crossing['ID']
                if np.linalg.norm(np.array([curr_x, curr_y]) - np.array([prev_x, prev_y])) <= self.eps and prev_id == curr_id:
                    print("Crossing correction at: " + str(prev_x) + ", " + str(prev_y) + ". Originally: " + str(prev_id))
                    prev_confidence = prev_crossing['confidence']
                    if curr_confidence >= prev_confidence:
                        prev_crossing['ID'] = 1 - prev_id
                        prev_crossing['confidence'] = curr_crossing['confidence']
                    else:
                        curr_crossing['ID'] = 1 - curr_id
                        curr_crossing['confidence'] = prev_crossing['confidence']
                    break

    def encounter_crossing(self, crossing):
        '''
        Called every time a new crossing is encountered.
        Returns a list of crossings that constitute the knot if knot is encountered at current crossing. 
        Else, returns None.
        '''
        # crossing must be in following format: {'loc': (x, y), 'ID': 0/1, 'confidence': [0, 1], 'crossing_idx': [0, ..., n], 'pixels_idx': [0, ..., p]}
        # ID options: 0 (under), 1 (over)
        # skip if not a crossing

        if crossing['ID'] == 2:
            return
        
        self.crossings.append(crossing)

    def add_crossing_to_stack(self, crossing):
        '''
        Runs cancellation and optionally adds crossing to stack.
        '''
        if not self.crossings_stack:
            curr_x, curr_y = crossing['loc']
            self.crossings_stack.append(crossing)
            return
        
        # R1: simplify if same crossing is immediately re-encountered (over -> under or under -> over)
        # TODO: check popping
        prev_crossing = self.crossings_stack.pop()
        prev_x, prev_y = prev_crossing['loc']
        curr_x, curr_y = crossing['loc']
        prev_id, curr_id = prev_crossing['ID'], crossing['ID']

        if np.linalg.norm(np.array([curr_x, curr_y]) - np.array([prev_x, prev_y])) <= self.eps:
            return

        # R2: simplify if UU/OO is encountered later as OO/UU (at the same locations, in no order) 
        if curr_id == prev_id:
            curr_pos = self.get_crossing_pos(crossing, len(self.crossings_stack))
            prev_pos = self.get_crossing_pos(prev_crossing, len(self.crossings_stack))

            if curr_pos != -1 and prev_pos != -1 and abs(curr_pos - prev_pos) == 1:
                min_pos = min(curr_pos, prev_pos)
                self.crossings_stack = self.crossings_stack[:min_pos] + self.crossings_stack[min_pos + 1:]
                return

        self.crossings_stack.append(prev_crossing)
        self.crossings_stack.append(crossing)
        self.check_for_knot()

    def check_for_knot(self) -> bool:
        '''
        Checks if the latest crossing results in a knot.
        Only accounts for trivial loops.
        '''
        # no knot encountered if < 3 crossings have been seen (?)
        # if len(self.crossings_stack) < 3:
        #     return False
            
        crossing = self.crossings_stack[-1]
        pos = self.get_crossing_pos(crossing, len(self.crossings_stack) - 1)
        # no knot encountered if crossing hasn't been seen before
        if pos == -1:
            return

        # no knot encountered if O...U (?)
        if self.crossings_stack[pos]['ID'] == 1 and self.crossings_stack[-1]['ID'] == 0:
            return

        # intermediate crossing = crossing in between start and end crossing (exclusive)
        # no knot encountered if every intermediate crossing is an undercrossing
        if all([intermediate_crossing['ID'] == 0 for intermediate_crossing in self.crossings_stack[pos + 1:-1]]):
            return
                    
        start_idx = self.crossings_stack[pos]['crossing_idx']
        # if no knots previously identified / knot in consideration is present earlier than previously identified knot,
        # update self.knot
        if not self.knot or start_idx < self.start_idx:
            end_idx = self.crossings_stack[-1]['crossing_idx']
            self.knot = self.crossings[start_idx:end_idx + 1]
            self.start_idx = start_idx
            self.crossings_stack = self.crossings_stack[:pos]

    def get_crossing_pos(self, crossing, end_stack_idx) -> int:
        '''
        Returns the index of previous sighting on the stack (before end_stack_idx).
        If crossing has not been seen previously, returns -1.
        '''
        curr_x, curr_y = crossing['loc']
        curr_id = crossing['ID']
        curr_crossing_idx = crossing['crossing_idx']

        # print(crossing)
        # print(self.crossings_stack)
        # print()

        # only look at crossings prior to most recently added crossing
        for pos in range(end_stack_idx):
            prev_crossing = self.crossings_stack[pos]
            prev_x, prev_y = prev_crossing['loc']
            prev_id = prev_crossing['ID']
            prev_crossing_idx = prev_crossing['crossing_idx']
            if np.linalg.norm(np.array([curr_x, curr_y]) - np.array([prev_x, prev_y])) <= self.eps:
                return pos
        
        return -1        

    def find_knot_from_corrected_crossings(self):
        '''
        Returns knot, if it exists.
        '''
        for crossing in self.crossings:
            self.add_crossing_to_stack(crossing)
        return self.knot

    def determine_under_over(self, img):
        mask = np.ones(img.shape[:2])
        mask[img[:, :, 0] <= 100] = 0
        # kernel = np.ones((2, 2), np.uint8)/4
        # smooth = cv2.filter2D(img,-1,kernel)
        # erode_mask = cv2.erode(mask, kernel)
        # plt.imshow(smooth)
        # plt.savefig('smooth.png')

def test_knot_detector_basic_r2(detector):
    detector.crossings = [
        {'loc': (0, 0), 'ID': 0}, # U at P0
        {'loc': (10, 10), 'ID': 0}, # U at P1
        {'loc': (0, 0), 'ID': 1}, # O at P0
        {'loc': (10, 10), 'ID': 1}, # O at P1
    ]

    for i in range(len(detector.crossings)):
        detector.crossings[i]['crossing_idx'] = i
        detector.crossings[i]['pixels_idx'] = i
        detector.crossings[i]['confidence'] = 1

    assert detector.find_knot_from_corrected_crossings() == []

def test_knot_detector_basic_r1r2(detector):
    detector.crossings = [
        {'loc': (0, 0), 'ID': 0}, # U at P0
        {'loc': (10, 10), 'ID': 0}, # U at P1
        {'loc': (20, 20), 'ID': 0}, # U at P2
        {'loc': (20, 20), 'ID': 1}, # O at P2
        {'loc': (0, 0), 'ID': 1}, # O at P0
        {'loc': (10, 10), 'ID': 1}, # O at P1
    ]

    for i in range(len(detector.crossings)):
        detector.crossings[i]['crossing_idx'] = i
        detector.crossings[i]['pixels_idx'] = i
        detector.crossings[i]['confidence'] = 1

    assert detector.find_knot_from_corrected_crossings() == []

def test_knot_detector_double_overhand(detector):
    # DEF ABC ABC DEF
    detector.crossings = [
        {'loc': (0, 0), 'ID': 0}, # U at D
        {'loc': (10, 10), 'ID': 1}, # O at E
        {'loc': (20, 20), 'ID': 0}, # U at F

        {'loc': (30, 30), 'ID': 0}, # U at A
        {'loc': (40, 40), 'ID': 1}, # O at B
        {'loc': (50, 50), 'ID': 0}, # U at C

        {'loc': (30, 30), 'ID': 1}, # O at A
        {'loc': (40, 40), 'ID': 0}, # U at B
        {'loc': (50, 50), 'ID': 1}, # O at C

        {'loc': (0, 0), 'ID': 1}, # O at D
        {'loc': (10, 10), 'ID': 0}, # U at E
        {'loc': (20, 20), 'ID': 1}, # O at F
    ]

    for i in range(len(detector.crossings)):
        detector.crossings[i]['crossing_idx'] = i
        detector.crossings[i]['pixels_idx'] = i
        detector.crossings[i]['confidence'] = 1

    assert detector.find_knot_from_corrected_crossings() == detector.crossings[:10]

if __name__ == '__main__':
    detector = KnotDetector()
    test_knot_detector_basic_r2(detector)
    test_knot_detector_basic_r1r2(detector)
    test_knot_detector_double_overhand(detector)