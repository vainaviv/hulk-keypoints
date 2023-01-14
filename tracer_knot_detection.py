from knot_detection import KnotDetector
import numpy as np
import cv2


test_data = np.load("/home/vainavi/hulk-keypoints/real_data/real_data_for_tracer/test/00000.npy", allow_pickle=True).item()

print(test_data.keys())

# print(test_data['pixels'])

print(test_data['img'][10][1])



output_vis_dir = '/home/mkparu/hulk-keypoints/.'
file_name = "test"

data_img = test_data["img"]
pixels = test_data['pixels']
clr = (255, 0, 0)
for x, y in pixels:
    cv2.circle(data_img, (x, y), 3, clr, -1)
cv2.imwrite(output_vis_dir + file_name + '.png', data_img)



# go pixel wise 
# generate a crop 20 by 20 around the pixel
# might need to trace a little extra  - then add the spline conditioning to the cropped image
# end as input to UON classifier
# keep tracing till you;re within 20 x 20 at current pixel - then stop and feed that crop in 

* * * * * * 
UON_CROP_SIZE = 20

pixels_so_far = []
for model_step in range(len(pixels)):
    #haven't reached model step in a trace yet
    if(model_step + 1 > len(pixels_so_far)):
        pixels_so_far.append(pixels[model_ste[]])


    pixels_so_far = pixels[:model_step + 1]
    current_pixel = pixels_so_far[model_step]
    trace_step = 0
    latest_trace_pixel = pixels_so_far[model_step]
    while abs(latest_trace_pixel[0] - current_pixel[0]) <= UON_CROP_SIZE / 2 and abs(latest_trace_pixel[1] - current_pixel[1]) <= UON_CROP_SIZE / 2:
        trace_step += 1
        latest_trace_pixel = pixels_so_far[model_step + trace_step]
        pixels_so_far.append(latest_trace_pixel)










