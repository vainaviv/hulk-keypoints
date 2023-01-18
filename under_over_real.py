import numpy as np
import shutil
import os
import cv2



vis_dir = './test_real_uo/'
if os.path.exists(vis_dir):
    shutil.rmtree(vis_dir)
os.makedirs(vis_dir)

file_name = "221_447"
data = np.load("/home/vainavi/hulk-keypoints/processed_sim_data/under_over_none2/real_test/" + file_name + ".npy", allow_pickle = True).item()

img = data["crop_img"]
spline_pixels = data["spline_pixels"]
under_over = data["under_over"]


#save img, with spline, print label

cv2.imwrite(vis_dir + file_name + '.png', img)

img_with_spline = img.copy()
for point in spline_pixels:
    x, y = point[0], point[1]
    cv2.circle(img_with_spline, (x, y), 2, (255,0, 0), -1)

cv2.imwrite(vis_dir + file_name + '_with_spline' + '.png', img_with_spline)
print("label", under_over)


# add contrast
def add_contrast_brightness(img, brightness=255,
               contrast=127):
    
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
  
    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
  
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            max = 255
        else:
            shadow = 0
            max = 255 + brightness
  
        al_pha = (max - shadow) / 255
        ga_mma = shadow
  
        # The function addWeighted calculates
        # the weighted sum of two arrays
        #output = alpha * img1 + beta * img2 + gamma
        cal = cv2.addWeighted(img, al_pha, 
                              img, 0, ga_mma)
  
    else:
        cal = img
  
    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)
  
        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(cal, Alpha, 
                              cal, 0, Gamma)
  
    return cal

img_with_contrast = add_contrast_brightness(img, contrast = 160)
cv2.imwrite(vis_dir + file_name + '_with_brightness_contrast' + '.png', img_with_contrast )



#canny edge detection
# If a pixel gradient is higher than the upper threshold, the pixel is accepted as an edge
# If a pixel gradient value is below the lower threshold, then it is rejected.
# If the pixel gradient is between the two thresholds, then it will be accepted only if it is connected to a pixel that is above the upper threshold.
# Canny recommended a upper:lower ratio between 2:1 and 3:1.

#params 
# t_lower = 50  # Lower Threshold
# t_upper = 150  # Upper threshold

t_lower = 50  # Lower Threshold
t_upper = 100  # Upper threshold

edges = cv2.Canny(img, t_lower, t_upper)
print(edges.shape)
cv2.imwrite(vis_dir + file_name + '_edges' + '.png', edges )


#canny edge detection on images with contrast
t_lower = 50  # Lower Threshold
t_upper = 150  # Upper threshold

edges_contrast = cv2.Canny(img_with_contrast, t_lower, t_upper)
cv2.imwrite(vis_dir + file_name + '_edges_with_contrast' + '.png', edges_contrast)







