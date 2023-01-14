import numpy as np
import os 
import shutil
import cv2


#How to use - 

#If you want to visualize all the crossings (stored as npy files) from a particular knot image along with their labels.

#directory of the original image (stored as npy)
input_directory_full_img = '/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_1/train/'

#directory of individual crossing images (stored as npy), that start with the name of the original image
input_directory_crossings = '/home/vainavi/hulk-keypoints/processed_sim_data/pinch_point/test/'

#where all the output images will be generated
output_vis_dir = '/home/vainavi/hulk-keypoints/processed_sim_data/crop_visualization/'


#clearing it out if it exists
if os.path.exists(output_vis_dir):
    shutil.rmtree(output_vis_dir)

#making the tree
if not os.path.exists(output_vis_dir):
    os.makedirs(output_vis_dir)


# file_name = '007156_rgb'
# file_path = input_directory_full_img + file_name

# loaded_data = np.load(file_path + '.npy', allow_pickle=True).item()
# img = loaded_data['img'][:,:,:3]*255
# cv2.imwrite(output_vis_dir + file_name + '.png', img)

for file in os.listdir(input_directory_crossings):
    if file.endswith('.npy'):
        loaded_data = np.load(input_directory_crossings + file, allow_pickle=True).item()
        img = (loaded_data['crop_img'][:, :, :3]*255).copy()
        pinch_point = loaded_data['pinch_point']
        img = cv2.circle(img, (int(pinch_point[1]), int(pinch_point[0])), 2, (0, 255, 255), -1)
        cv2.imwrite(output_vis_dir + file[:-4] + '.png', img)    

