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
        if not self.crossings_stack:
            curr_x, curr_y = crossing['loc']
            self.crossings_stack.append(crossing)
            return
        
        # simplify if same crossing is immediately re-encountered (over -> under or under -> over)
        # TODO: check popping
        prev_crossing = self.crossings_stack.pop()
        prev_x, prev_y = prev_crossing['loc']
        curr_x, curr_y = crossing['loc']
        prev_id, curr_id = prev_crossing['ID'], crossing['ID']

        if np.linalg.norm(np.array([curr_x, curr_y]) - np.array([prev_x, prev_y])) <= self.eps:
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
        pos = self.get_crossing_pos(crossing)
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
                return pos
        
        return -1        

    def find_knot_from_corrected_crossings(self):
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

# def test_knot_detector_basic(detector):
#     crossings = [
#         {'loc': (0, 0), 'ID': 0}, # U at P0
#         {'loc': (10, 9), 'ID': 0}, # U at P1
#         {'loc': (20, 21), 'ID': 0}, # U at P2
#         {'loc': (30, 29), 'ID': 1}, # O at P3
#         {'loc': (100, 100), 'ID': 2}, # NC
#         {'loc': (40, 39), 'ID': 0}, # U at P4
#         {'loc': (21, 20), 'ID': 1}, # O at P2
#         {'loc': (29, 30), 'ID': 0}, # U at P3
#         {'loc': (41, 40), 'ID': 1}, # O at P4
#         {'loc': (9, 10), 'ID': 1} # O at P1
#     ]

#     assert detector.detect_knot(crossings) == [{'loc': (20, 21), 'ID': 0}, {'loc': (30, 29), 'ID': 1}, {'loc': (40, 39), 'ID': 0}, {'loc': (21, 20), 'ID': 1}]

# # make sure O...U is not counted as a knot - same as other input with swapped U and O
# def test_knot_detector_edge_case(detector):
#     crossings = [
#         {'loc': (0, 0), 'ID': 1}, # O at P0
#         {'loc': (10, 9), 'ID': 1}, # O at P1
#         {'loc': (20, 21), 'ID': 1}, # O at P2
#         {'loc': (30, 29), 'ID': 0}, # U at P3
#         {'loc': (100, 100), 'ID': 2}, # NC
#         {'loc': (40, 39), 'ID': 1}, # O at P4
#         {'loc': (21, 20), 'ID': 0}, # U at P2
#         {'loc': (29, 30), 'ID': 1}, # O at P3
#         {'loc': (41, 40), 'ID': 0}, # U at P4
#         {'loc': (9, 10), 'ID': 0} # U at P1
#     ]
#     assert detector.detect_knot(crossings) == [{'loc': (30, 29), 'ID': 0}, {'loc': (40, 39), 'ID': 1}, {'loc': (21, 20), 'ID': 0}, {'loc': (29, 30), 'ID': 1}]

if __name__ == '__main__':
    pass
    
    # TODO: Update test cases to match updated code structure

    # detector = KnotDetector()
    # test_knot_detector_basic(detector)

    # detector2 = KnotDetector()
    # test_knot_detector_edge_case(detector2)