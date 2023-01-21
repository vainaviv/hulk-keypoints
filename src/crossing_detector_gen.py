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
crop_size = 16

input_file_path = '/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_2/train/'

#newest dataset w new crossing gen and filters
#train: 48132 , test: 1669
out_file_path = '/home/vainavi/hulk-keypoints/processed_sim_data/under_over_hard2_16_recenter/train'
limit = 5000 # 500 for test, for train 10000 per


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

if os.path.exists(out_file_path):
    shutil.rmtree(out_file_path)

if not os.path.exists(out_file_path):
    os.makedirs(out_file_path)

files = os.listdir(input_file_path)
under_done = False
over_done = False
num_under = 0
num_over = 0
num_none = 0
for file in files:
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


    points_3d = np.array(np_data['points_3d'])


    #?? should be 0 for under, 1 for over
    crossing_info = np.zeros((pixels.shape[0], 4), dtype=np.int32) - 1 # corresponding point idx, then second int is 1 for under, 2 for over


    #create line segments of consecutive points
    line_segments = []
    for i in range(len(pixels) - 1):
        curr_pixel, next_pixel = pixels[i], pixels[i + 1]     
        line_segments.append(LineString([curr_pixel, next_pixel]))


    for i, current_line_segment in enumerate(line_segments):
        prev_line_segments = line_segments[:max(0, i-NUM_STEPS_MIN_FOR_CROSSING)]

        if len(prev_line_segments) == 0:
            continue

        for j in range (len(prev_line_segments)):
            prev_line_seg = prev_line_segments[j]
            if(current_line_segment.intersects(prev_line_seg)):
                intersection_point = current_line_segment.intersection(prev_line_seg)
                if "POINT" in str(intersection_point):
                    intersection_point = [intersection_point.x, intersection_point.y]
                else:
                    intersection_point = [-1, -1]
                if(points_3d[i, 2] > points_3d[j, 2] and points_3d[i+1, 2] > points_3d[j+1, 2]):
                    crossing_info[i] = [j, 1, intersection_point[0], intersection_point[1]]
                    crossing_info[i + 1] = [j, 1, intersection_point[0], intersection_point[1]]
                elif (points_3d[i, 2] < points_3d[j, 2] and points_3d[i+1, 2] < points_3d[j+1, 2]):
                    crossing_info[i] = [j, 0, intersection_point[0], intersection_point[1]]
                    crossing_info[i + 1] = [j, 0, intersection_point[0], intersection_point[1]]

    crossings = []
    # add all midpoints of contiguous crossings to the crossings list
    spans = []
    span_centers = []
    centers_of_cur_span = []
    cur_span = None
    for i, crossing in enumerate(crossing_info):
        if crossing[0] != -1:
            if cur_span is None:
                cur_span = [i]
                centers_of_cur_span = [[crossing[2], crossing[3]]]
            else:
                cur_span.append(i)
                centers_of_cur_span.append([crossing[2], crossing[3]])
        else:
            if cur_span is not None:
                spans.append(cur_span)
                span_centers.append(centers_of_cur_span)
                centers_of_cur_span = []
                cur_span = None
    for i in range(len(spans)):
        span = spans[i]
        span_center = np.array(span_centers[i])
        crossings.append({"spline_index": np.median(span), "center_pixel": [np.median(span_center[:,0]), np.median(span_center[:,1])]})

    
    # save the crossings
    crossings_dicts = []
    for crossing in crossings:
        crossing_pixel = crossing['center_pixel'] #pixels[int(crossing)]
        crossing_dict = {}
        # if crossing_dict['under_over'] == -1:
        #     crossing_dict['under_over'] = non_crossing_info[int(crossing)][1]
        uon = crossing_info[int(crossing['spline_index'])][1]
        if uon == 0:
            if num_under > limit:
                under_done = True
                continue
            num_under += 1
        elif uon == 1:
            if num_over > limit:
                over_done = True
                continue
            num_over += 1
        elif uon == 2:
            if num_none > limit:
                continue
            num_none += 1

        crossing_top_left = np.array([int(crossing_pixel[0])-crop_size, int(crossing_pixel[1])-crop_size])

        crossing_dict['crop_img'] = np_data['img'][int(crossing_pixel[1])-crop_size:int(crossing_pixel[1])+crop_size, int(crossing_pixel[0])-crop_size:int(crossing_pixel[0])+crop_size]
        if crossing_dict['crop_img'].shape[0] < 2*crop_size or crossing_dict['crop_img'].shape[1] < 2*crop_size:
            continue


        # add all spline pixels before and after the crossing pixel that are within the crop size
        spline_pixels = []
        for i in range(int(crossing['spline_index']), len(pixels)):
            if np.linalg.norm(pixels[i] - crossing_pixel, ord=np.inf) <= crop_size:
                spline_pixels.append(pixels[i] - crossing_top_left)
            else:
                break
        for i in range(int(crossing['spline_index']), 0, -1):
            if np.linalg.norm(pixels[i] - crossing_pixel, ord=np.inf) <= crop_size:
                spline_pixels.insert(0, pixels[i] - crossing_top_left)
            else:
                break
        
        if len(spline_pixels) < 2:
            continue

        crossing_dict['under_over'] = uon
        crossing_dict['spline_pixels'] = spline_pixels
        # crossing_dict['crossing_pixel'] = [crop_size, crop_size]
        crossing_dict['ORIG_img_path'] = os.path.join(input_file_path, file)
        crossing_dict['ORIG_pixels'] = pixels
        crossing_dict['ORIG_center_pt'] = crossing_pixel
        crossings_dicts.append(crossing_dict)

    print("under: ", num_under)
    print("over: ", num_over)
    print("none: ", num_none)

    # find all contiguous segments in crossing info
    for i, crossing in enumerate(crossings_dicts):
        np.save(os.path.join(out_file_path, file[:-4] + '_' + str(i) + '.npy'), crossing)
