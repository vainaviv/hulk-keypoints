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
crop_size = 32

img_path = '/home/vainavi/hulk-keypoints/real_data/real_data_for_tracer/test'

# out_file_path = '/home/kaushiks/hulk-keypoints/processed_sim_data/under_over_REAL_centered/test'
# out_file_path2 = '/home/vainavi/hulk-keypoints/processed_sim_data/under_over_REAL_centered/test'

input_file_path = '/home/kaushiks/hulk-keypoints/processed_sim_data/under_over_REAL_centered/test'

#newest dataset w new crossing gen and filters
#train: 48132 , test: 1669
out_file_path = '/home/vainavi/hulk-keypoints/processed_sim_data/under_over_REAL_centered/test'
limit = 500 # 500 for test, for train 5000 per

if os.path.exists(out_file_path):
    shutil.rmtree(out_file_path)

if not os.path.exists(out_file_path):
    os.makedirs(out_file_path)

files = os.listdir(input_file_path)

for file in files:
    print(file)
    if not file.endswith('.npy'):
        continue

    np_data = np.load(os.path.join(input_file_path, file), allow_pickle=True).item()
    img = np.load(os.path.join(img_path, np_data['ORIG_img_path']), allow_pickle=True).item()['img']
    crossing = np_data['ORIG_center_pt']
    crossing = np.array([int(crossing[0]), int(crossing[1])])
    pixels = np_data['ORIG_pixels']

    crossing_idx = 0
    for i in range(len(pixels)):
        pix = pixels[i]
        pix = np.array([int(pix[0]), int(pix[1])])
        if (pix == crossing).all():
            print("got here")
            crossing_idx = i
            break

    print("crossing idx: ", crossing_idx)
    

    crossing_pixel = crossing
    crossing_dict = {}
    # if crossing_dict['under_over'] == -1:
    #     crossing_dict['under_over'] = non_crossing_info[int(crossing)][1]
    uon = np_data['under_over']

    crossing_top_left = np.array([int(crossing_pixel[0])-crop_size, int(crossing_pixel[1])-crop_size])

    crossing_dict['crop_img'] = img[int(crossing_pixel[1])-crop_size:int(crossing_pixel[1])+crop_size, int(crossing_pixel[0])-crop_size:int(crossing_pixel[0])+crop_size]
    if crossing_dict['crop_img'].shape[0] < 2*crop_size or crossing_dict['crop_img'].shape[1] < 2*crop_size:
        continue


    # add all spline pixels before and after the crossing pixel that are within the crop size
    spline_pixels = []
    for i in range(int(crossing_idx), len(pixels)):
        if np.linalg.norm(pixels[i] - crossing_pixel, ord=np.inf) <= crop_size:
            spline_pixels.append(pixels[i] - crossing_top_left)
        else:
            break
    for i in range(int(crossing_idx), 0, -1):
        if np.linalg.norm(pixels[i] - crossing_pixel, ord=np.inf) <= crop_size:
            spline_pixels.insert(0, pixels[i] - crossing_top_left)
        else:
            break
    
    if len(spline_pixels) < 2:
        continue

    crossing_dict['under_over'] = uon
    crossing_dict['spline_pixels'] = spline_pixels
    print("spline pixels: ", spline_pixels)

    crossing_dict['ORIG_img_path'] = os.path.join(input_file_path, file)
    crossing_dict['ORIG_pixels'] = pixels
    crossing_dict['ORIG_center_pt'] = crossing_pixel
    
    np.save(os.path.join(out_file_path, file), crossing_dict)