import enum
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

NUM_STEPS_MIN_FOR_CROSSING = 10
DIST_THRESH = 0.1

input_file_path = '/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_1/train'
out_file_path = '/home/vainavi/hulk-keypoints/processed_sim_data/20crop_under_over_crossings_dataset/train'

if not os.path.exists(out_file_path):
    os.makedirs(out_file_path)

files = os.listdir(input_file_path)

for file in files:
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
    
    crossing_info = np.zeros((pixels.shape[0], 2), dtype=np.int32) - 1 # corresponding point idx, then second int is 1 for under, 2 for over

    for i, point in enumerate(pixels):
        prev_pixels = pixels[:max(0, i-NUM_STEPS_MIN_FOR_CROSSING)]
        if len(prev_pixels) == 0:
            continue
        min_dist, argmin_pt = np.min(np.linalg.norm(prev_pixels - point, axis=1)), np.argmin(np.linalg.norm(prev_pixels - point, axis=1))
        if min_dist < DIST_THRESH:
            overcrossing = (points_3d[i, 2] > points_3d[argmin_pt, 2])
            crossing_info[i] = [argmin_pt, overcrossing]

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
        crossings.append(np.mean(span))
    
    # save the crossings
    crossings_dicts = []
    for crossing in crossings:
        crossing_pixel = pixels[int(crossing)]
        crossing_dict = {}
        crossing_dict['under_over'] = crossing_info[int(crossing)][1]
        crop_size = 10 #20

        crossing_top_left = np.array([int(crossing_pixel[0])-crop_size, int(crossing_pixel[1])-crop_size])

        # og_img = np_data['img']
        # plt.imshow(og_img)
        # plt.savefig('og_img.png')
        # plt.clf()
        # raise Exception()

        crossing_dict['crop_img'] = np_data['img'][int(crossing_pixel[1])-crop_size:int(crossing_pixel[1])+crop_size, int(crossing_pixel[0])-crop_size:int(crossing_pixel[0])+crop_size]
        if crossing_dict['crop_img'].shape[0] < 2*crop_size - 1 or crossing_dict['crop_img'].shape[1] < 2*crop_size - 1:
            continue

        # add all spline pixels before and after the crossing pixel that are within the crop size
        spline_pixels = []
        for i in range(int(crossing), len(pixels)):
            if np.linalg.norm(pixels[i] - crossing_pixel, ord=np.inf) < crop_size:
                spline_pixels.append(pixels[i] - crossing_top_left)
            else:
                break
        for i in range(int(crossing), 0, -1):
            if np.linalg.norm(pixels[i] - crossing_pixel, ord=np.inf) < crop_size:
                spline_pixels.insert(0, pixels[i] - crossing_top_left)
            else:
                break

        crossing_dict['spline_pixels'] = spline_pixels
        crossings_dicts.append(crossing_dict)

    # find all contiguous segments in crossing info
    for i, crossing in enumerate(crossings_dicts):
        np.save(os.path.join(out_file_path, file[:-4] + '_' + str(i) + '.npy'), crossing)
