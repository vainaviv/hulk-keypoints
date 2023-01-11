import numpy as np
import os 
import shutil
import cv2


#How to use - 

#If you want to visualize all the crossings (stored as npy files) from a particular knot image along with their labels.

#directory of the original image (stored as npy)
input_directory_full_img = '/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_1/train/'

#directory of individual crossing images (stored as npy), that start with the name of the original image
input_directory_crossings = '/home/mkparu/hulk-keypoints/processed_sim_data/under_over_none2/train/'

#where all the output images will be generated
output_vis_dir = '/home/mkparu/hulk-keypoints/processed_sim_data/crop_visualization/'


UNDER_OVER_DICT = ['under', 'over','none']
#clearing it out if it exists
if os.path.exists(output_vis_dir):
    shutil.rmtree(output_vis_dir)

#making the tree
if not os.path.exists(output_vis_dir):
    os.makedirs(output_vis_dir)


file_name = '007156_rgb'
file_path = input_directory_full_img + file_name

loaded_data = np.load(file_path + '.npy', allow_pickle=True).item()
img = loaded_data['img'][:,:,:3]*255
cv2.imwrite(output_vis_dir + file_name + '.png', img)


for file in os.listdir(input_directory_crossings):
    if file.startswith(file_name):
        loaded_data = np.load(input_directory_crossings + file, allow_pickle=True).item()
        print(file, UNDER_OVER_DICT [loaded_data['under_over']])
        img = loaded_data['crop_img'][:, :, :3]*255
        cv2.imwrite(output_vis_dir + file[:-4] + '.png', img)    

