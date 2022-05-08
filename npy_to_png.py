import numpy as np
import cv2
import os

out_dir = 'more_knot_crops_png/'
for file in os.listdir('/host/hulkL_seg/train/more_knot_crops'):
    img = np.load('/host/hulkL_seg/train/more_knot_crops/' + file, allow_pickle=True).item()['img']
    cv2.imwrite(out_dir + file[:-4] + '.png', img)