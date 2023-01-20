import numpy as np
import shutil
import os
import cv2
import matplotlib.pyplot as plt

def draw_spline(spline_pixels, file_name):
    img_with_spline = img.copy()
    for point in spline_pixels:
        x, y = point[0], point[1]
        cv2.circle(img_with_spline, (x, y), 2, (255,0, 0), -1)

    cv2.imwrite(vis_dir + file_name + '_with_spline' + '.png', img_with_spline)
    print("label", under_over)

def perform_contrast(img):
    img = img.copy()

    # show a histogram of brightness values in the image
    values = img[:, :, 0].flatten()
    # plt.hist(values, bins=256, range=(0, 256), fc='k', ec='k')
    # plt.show()

    cable_mask = img[:, :, 0] > 120
    pixels_x_normalize, pixels_y_normalize = np.where(cable_mask > 0)
    pixel_vals = img[pixels_x_normalize, pixels_y_normalize, 0]
    min_px_val = np.min(pixel_vals)
    max_px_val = np.max(pixel_vals)
    # cable_norm = pixels_normalize / np.linalg.norm(pixels_normalize)
    cable_mask = np.array([cable_mask, cable_mask, cable_mask]).transpose((1,2,0))
    background = img * (1 - cable_mask)
    cable = ((img * cable_mask) - min_px_val) / (max_px_val - min_px_val)
    cable = cable * 255.0
    cable *= cable_mask
    return cable #img_contrast

# add contrast
def add_contrast_brightness(img, brightness=255,contrast=127):
    
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

if __name__ == '__main__':
    data_folder = '/home/vainavi/hulk-keypoints/processed_sim_data/under_over_none2/real_test'

    vis_dir = './test_real_uo/'
    if os.path.exists(vis_dir):
        shutil.rmtree(vis_dir)
    os.makedirs(vis_dir)

    for i, f in enumerate(np.sort(os.listdir(data_folder))):
        data_path = os.path.join(data_folder, f)
        data = np.load(data_path, allow_pickle=True).item()
        img = data["crop_img"]
        spline_pixels = data["spline_pixels"]
        under_over = data["under_over"]
        # img_with_contrast = add_contrast_brightness(img, contrast = 160)
        img_with_contrast = perform_contrast(img)
        # resize img to 512x512
        img = cv2.resize(img, (512, 512))
        img_with_contrast = cv2.resize(img_with_contrast, (512, 512))

        cv2.imwrite(vis_dir + f + '.png', img)
        cv2.imwrite(vis_dir + f + '_with_brightness_contrast' + '.png', img_with_contrast)        

