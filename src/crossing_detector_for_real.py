import enum
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shutil
import random
import math

from shapely.geometry import LineString

NUM_STEPS_MIN_FOR_CROSSING = 5
DIST_THRESH = 0.1
# DIST_THRESH = 2
NOT_CROSSING_THRESH = 10
COS_ANGLE_THRESH = 0.85
input_file_path = 'real_data_improved_dense_points_jan16'

out_file_path = 'under_over_REAL_centered'
limit = 10000 # for test, for train 5000 per

#calculates the cosine angle between two line segments
def calculate_cos_angle(lineseg1, lineseg2):
    p1, p2 = lineseg1.coords
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2=p2[1]
    vector1 = [x2-x1, y2-y1]

    p1, p2 = lineseg2.coords
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2=p2[1]
    vector2 = [x2-x1, y2-y1]

    return np.dot(vector1, vector2)/ (np.linalg.norm(vector1) * np.linalg.norm(vector2))

# if os.path.exists(out_file_path):
#     shutil.rmtree(out_file_path)

if not os.path.exists(out_file_path):
    os.makedirs(out_file_path)

files = sorted(os.listdir(input_file_path))
under_done = False
over_done = False
num_under = 0
num_over = 0
num_none = 0

start_indx = int(input('Enter start index: '))
for file in files[start_indx:]:
    if under_done and over_done:
        break
    print(file)
    if not file.endswith('.npy'):
        continue

    np_data = np.load(os.path.join(input_file_path, file), allow_pickle=True).item()

    # identify all locations of crossings
    pixels_dict = np_data['pixels']
    pixels = np.zeros((len(pixels_dict), 2))
    for i in range(len(pixels_dict)):
        pixels[i] = np.array(pixels_dict[i])

    crossings = []
    #create line segments of consecutive points
    line_segments = []
    for i in range(len(pixels) - 1):
        curr_pixel, next_pixel = pixels[i], pixels[i + 1]     
        line_segments.append(LineString([curr_pixel, next_pixel]))

    center_point_set = {}
    for i, current_line_segment in enumerate(line_segments):
        prev_line_segments = line_segments[:max(0, i-NUM_STEPS_MIN_FOR_CROSSING)]

        if len(prev_line_segments) == 0:
            continue

        for j in range (len(prev_line_segments)):
            prev_line_seg = prev_line_segments[j]
            if(current_line_segment.intersects(prev_line_seg)):
                intersection_point = current_line_segment.intersection(prev_line_seg)
                # convert to numpy array
                center_point_tuple = [intersection_point.x, intersection_point.y]
                if tuple(center_point_tuple) in center_point_set:
                    continue
                center_point_set[tuple(center_point_tuple)] = True
                intersection_point = np.array(center_point_tuple)
                crossings.append({'index': i, 'center_pt': intersection_point})
                print(crossings[-1], j, prev_line_seg, current_line_segment)
    
    # save the crossings
    crossings_dicts = []
    for crossing in crossings:
        crossing_dict = {}
        # if crossing_dict['under_over'] == -1:
        #     crossing_dict['under_over'] = non_crossing_info[int(crossing)][1]
        crop_size = 10

        center_point = crossing['center_pt']
        crossing_top_left = np.array([int(center_point[0])-crop_size, int(center_point[1])-crop_size])
        crossing_dict['crop_img'] = np_data['img'][int(center_point[1])-crop_size:int(center_point[1])+crop_size, int(center_point[0])-crop_size:int(center_point[0])+crop_size]
        if crossing_dict['crop_img'].shape[0] < 2*crop_size or crossing_dict['crop_img'].shape[1] < 2*crop_size:
            continue

        # add all spline pixels before and after the crossing pixel that are within the crop size
        spline_pixels = []
        for i in range(crossing['index'], len(pixels)):
            if np.linalg.norm(pixels[i] - center_point, ord=np.inf) <= crop_size:
                spline_pixels.append(pixels[i] - crossing_top_left)
            else:
                break
        for i in range(crossing['index'], 0, -1):
            if np.linalg.norm(pixels[i] - center_point, ord=np.inf) <= crop_size:
                spline_pixels.insert(0, pixels[i] - crossing_top_left)
            else:
                break
        
        if len(spline_pixels) < 2:
            continue

        # get user input to determine over/under/skip with opencv

        scale = 20
        disp_img = cv2.resize(crossing_dict['crop_img'], (0,0), fx=scale, fy=scale)
        # add circles at the spline pixels
        for pixel in spline_pixels:
            cv2.circle(disp_img, (int(pixel[0]*scale), int(scale*pixel[1])), 10, (0, 0, 255), -1)

        cv2.imshow('image', disp_img)
        key = cv2.waitKey(0)
        if key == ord('u'):
            uon = 0
            num_under += 1
        elif key == ord('o'):
            uon = 1
            num_over += 1
        elif key == ord('n'):
            continue
        elif key == ord('d'):
            break
        print("received", key)

        crossing_dict['under_over'] = uon
        crossing_dict['spline_pixels'] = spline_pixels

        crossing_dict['ORIG_img_path'] = file
        crossing_dict['ORIG_pixels'] = pixels
        crossing_dict['ORIG_center_pt'] = crossing['center_pt']
        crossings_dicts.append(crossing_dict)

    print("under: ", num_under)
    print("over: ", num_over)
    print("none: ", num_none)

    # find all contiguous segments in crossing info
    for i, crossing in enumerate(crossings_dicts):
        np.save(os.path.join(out_file_path, file[:-4] + '_' + str(i) + '.npy'), crossing)