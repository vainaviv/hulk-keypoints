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

input_file_path = '/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_2/train/'

#newest dataset w new crossing gen and filters
#train: 48132 , test: 1669
out_file_path = '/home/vainavi/hulk-keypoints/processed_sim_data/under_over_centered_hard2/train'
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
    crossing_info = np.zeros((pixels.shape[0], 2), dtype=np.int32) - 1 # corresponding point idx, then second int is 1 for under, 2 for over

    ########## Where we find crossings

    #OLD WAY 
  
    # for i, point in enumerate(pixels):
    #     prev_pixels = pixels[:max(0, i-NUM_STEPS_MIN_FOR_CROSSING)]
    #     if len(prev_pixels) == 0:
    #         continue
    #     min_dist, argmin_pt = np.min(np.linalg.norm(prev_pixels - point, axis=1)), np.argmin(np.linalg.norm(prev_pixels - point, axis=1))
    #     # print("min dist: ", min_dist)
    #     if min_dist < DIST_THRESH:
    #         overcrossing = (points_3d[i, 2] >= points_3d[argmin_pt, 2])
    #         crossing_info[i] = [argmin_pt, overcrossing]

   # NEW WAY

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
                if(points_3d[i, 2] > points_3d[j, 2] and points_3d[i+1, 2] > points_3d[j+1, 2]):
                    crossing_info[i] = [j, 1]
                    crossing_info[i + 1] = [j, 1]
                elif (points_3d[i, 2] < points_3d[j, 2] and points_3d[i+1, 2] < points_3d[j+1, 2]):
                    crossing_info[i] = [j, 0]
                    crossing_info[i + 1] = [j, 0]

    crossings = []
    # add all midpoints of contiguous crossings to the crossings list
    spans = []
    cur_span = None
    for i, crossing in enumerate(crossing_info):
        if crossing[0] != -1:
            if cur_span is None:
                cur_span = [i]
            else:
                cur_span.append(i)
        else:
            if cur_span is not None:
                spans.append(cur_span)
                cur_span = None
    for span in spans:
        crossings.append(np.median(span))

    #to visualize all the crossings on the full image

    # output_vis_dir = '/home/mkparu/hulk-keypoints/processed_sim_data/crop_visualization/'
    # file_name = file[:-4] + "_with_crossings"

    # data_img = np_data['img'][:,:,:3]*255

    # print(data_img.shape)
    # clr = (255, 0, 0)
    # for pixel_idx in crossings:
    #     pixel_idx = int(pixel_idx)
    #     x = int(pixels[pixel_idx][0])
    #     y = int(pixels[pixel_idx][1])
    #     # print("x", x, "y", y)
    #     cv2.circle(data_img, (x, y), 3, clr, -1)
    # cv2.imwrite(output_vis_dir + file_name + '.png', data_img)

    ################### Get not crossings
    # num_crossings = len(crossings)
    # non_crossing_info = np.zeros((pixels.shape[0], 2), dtype=np.int32) - 1 
    # for i, point in enumerate(pixels):
    #     part1 = pixels[0:max(0, i-NUM_STEPS_MIN_FOR_CROSSING)]
    #     part2 = pixels[min(i+NUM_STEPS_MIN_FOR_CROSSING, len(pixels)):]
    #     if part1.shape[0] == 0:
    #         pixels_all = part2
    #     elif part2.shape[0] == 0:
    #         pixels_all = part1
    #     else:
    #         pixels_all = np.vstack((part1, part2))
    #     min_dist_all, argmin_pt_all = np.min(np.linalg.norm(pixels_all - point, axis=1)), np.argmin(np.linalg.norm(pixels_all - point, axis=1), axis=0)
    #     if min_dist_all > NOT_CROSSING_THRESH:
    #         non_crossing_info[i] = [argmin_pt_all, (2)] #2 = neither under or over

    # spans = []
    # cur_span = None
    # for i, crossing in enumerate(non_crossing_info):
    #     if crossing[0] != -1:
    #         if cur_span is None:
    #             cur_span = [i]
    #         else:
    #             cur_span.append(i)
    #     else:
    #         if cur_span is not None:
    #             spans.append(cur_span)
    #             cur_span = None
    # num_non_crossings = 0
    # for span in spans:
    #     if num_non_crossings > int(num_crossings/2.5):
    #         break
    #     crossings.append(np.median(span))
    #     num_non_crossings += 1
    ###################
    
    # save the crossings
    crossings_dicts = []
    for crossing in crossings:
        crossing_pixel = pixels[int(crossing)]
        crossing_dict = {}
        # if crossing_dict['under_over'] == -1:
        #     crossing_dict['under_over'] = non_crossing_info[int(crossing)][1]
        uon = crossing_info[int(crossing)][1]
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

        crossing_dict['under_over'] = uon

        crop_size = 10

        crossing_top_left = np.array([int(crossing_pixel[0])-crop_size, int(crossing_pixel[1])-crop_size])

        crossing_dict['crop_img'] = np_data['img'][int(crossing_pixel[1])-crop_size:int(crossing_pixel[1])+crop_size, int(crossing_pixel[0])-crop_size:int(crossing_pixel[0])+crop_size]
        if crossing_dict['crop_img'].shape[0] < 2*crop_size or crossing_dict['crop_img'].shape[1] < 2*crop_size:
            continue

        #filtering steps for under and over crossings - remove if crossing is out of crop or angle betweem is too small
        # if uon == 0 or uon == 1:
        #     crop_img = crossing_dict['crop_img']
        #     cable_mask = np.ones(crop_img.shape[:2])
        #     cable_mask[crop_img[:, :, 1] < 0.25] = 0

        #     crossing_pixel_other = pixels[crossing_info[int(crossing)][0]]
            
            
        #     cropped_pixel = crossing_pixel - crossing_top_left
        #     cropped_other_pixel = crossing_pixel_other - crossing_top_left

        #     #if the crossing pixel is out of crop bounds, do not include
        #     if(cropped_other_pixel[0] >= 2*crop_size or cropped_other_pixel[0] < 0  or cropped_other_pixel[1] >= 2*crop_size or cropped_other_pixel[1] < 0 ):
        #         # print("crossing is not within crop")
        #         if uon == 0:
        #             num_under -= 1
        #         else:
        #             num_over -=1
        #         num_removed += 1
        #         continue

        #     #if the angle between the line segments of the crossings is too small, do not include
        #     if(abs(calculate_cos_angle(line_segments[int(crossing)], line_segments[int(crossing_info[int(crossing)][0])])) > COS_ANGLE_THRESH):
        #         # print("angle is too close")
        #         if uon == 0:
        #             num_under -= 1
        #         else:
        #             num_over -=1
        #         num_removed += 1
        #         continue

        #     # if either this pixel or the on crossing it is light, do not include
        #     # if(cable_mask[int(cropped_pixel[0])][int(cropped_pixel[1])] == 0 or cable_mask[int(cropped_other_pixel[0])][int(cropped_other_pixel[1])] == 0):
        #     #     print("crossing is not bright enough")
        #     #     removed_crossings.append(crossing)
        #     #     continue



        # add all spline pixels before and after the crossing pixel that are within the crop size
        spline_pixels = []
        for i in range(int(crossing), len(pixels)):
            if np.linalg.norm(pixels[i] - crossing_pixel, ord=np.inf) <= crop_size:
                spline_pixels.append(pixels[i] - crossing_top_left)
            else:
                break
        for i in range(int(crossing), 0, -1):
            if np.linalg.norm(pixels[i] - crossing_pixel, ord=np.inf) <= crop_size:
                spline_pixels.insert(0, pixels[i] - crossing_top_left)
            else:
                break
        
        if len(spline_pixels) < 2:
            continue
    
        crossing_dict['spline_pixels'] = spline_pixels
        crossings_dicts.append(crossing_dict)

    print("under: ", num_under)
    print("over: ", num_over)
    print("none: ", num_none)

    # find all contiguous segments in crossing info
    for i, crossing in enumerate(crossings_dicts):
        np.save(os.path.join(out_file_path, file[:-4] + '_' + str(i) + '.npy'), crossing)
