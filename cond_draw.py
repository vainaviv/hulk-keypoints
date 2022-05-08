import numpy
import matplotlib.pyplot as plt
import cv2
import os

for img in os.listdir('detectron_pred_bb'):
    # allow user to click two points on image
    plt.imshow(cv2.imread(f'detectron_pred_bb/{img}'))
    plt.show()
    points = plt.ginput(2)

    print(points)