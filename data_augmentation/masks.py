import pandas as pd
import cv2
import os
from scipy.spatial import distance
import numpy as np

# Add path to image data (should contain cluttered and presorted folders)"
path_polysecure = ""

# Add path to target directory where results can be saved
target_path = ""

# Pick name for result csv file
masks_csv_name = 'masks.csv'

# Dataframe lists all images which contain one object and (hopefully) no black-transparent objects
one_lid = pd.read_csv('data/one_lid.csv')

# Add 'contours', 'objects' or 'binary_masks' to tasks
# For copy_and_past.py 'objects', 'binary_masks' and 'save_csv' is needed
# 'contours': Generating and saving images with contours
# 'objects': Generating and saving images with object and black background
# 'binary_masks': Generating and saving binary mask
# 'save_csv': Generating and saving a csv file with original image name and corresponding contour, object and binary_mask names
tasks = ['contours', 'objects', 'binary_masks', 'save_csv']

contour_names = []
object_names = []
binary_mask_names = []

image_center = (300, 400)

for index, row in one_lid.iterrows():
    if index % 100 == 0:
        print('Generating and saving masks for image ' + str(index) + ' of ' + str(len(one_lid)))
    # Get image path
    image_path = os.path.join(path_polysecure, row['name'][1:])
    # Get image
    img = cv2.imread(image_path)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Add gaussian blurr to get better mask later
    #img_gauss = cv2.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=cv2.BORDER_CONSTANT)
    img_gauss = cv2.bilateralFilter(img_gray, 9, 75, 75)


    # computes individual threshold for each image
    threshold = 20

    # computes binary mask
    ret, thresh = cv2.threshold(src=img_gauss, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY)

    # Get contours from mask
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find right contour from list of contours
    distances = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] == 0 or M["m00"] == 0:
            distances_to_center = 1000
        else:
            center_X = int(M["m10"] / M["m00"])
            center_Y = int(M["m01"] / M["m00"])
            contour_center = (center_X, center_Y)
            distances_to_center = (distance.euclidean(image_center, contour_center))
            if cv2.arcLength(contour, True) < 200:
                distances_to_center = 1000
        distances.append(distances_to_center)
        # print(cv2.arcLength(contour, True))

    i = distances.index(min(distances))

    cnt = contours[i]

    # Generating and saving images with contours
    if 'contours' in tasks:
        image = cv2.imread(image_path)
        cv2.drawContours(image, [cnt], -1, (0, 0, 255), thickness=5)
        contours_path = os.path.join(target_path, 'contours')
        if not os.path.isdir(contours_path):
            os.mkdir(contours_path)
        contour_name = "contour_" + str(index) + '.jpg'
        save_path = os.path.join(contours_path, contour_name)
        cv2.imwrite(save_path, image)
        contour_names.append(contour_name)

    # Generating and saving images with object with black background
    if 'objects' in tasks:
        image = cv2.imread(image_path)
        mask_inv = np.zeros((600, 800, 3), np.uint8)
        cv2.drawContours(mask_inv, [cnt], -1, (255, 255, 255), thickness=-2, lineType=cv2.LINE_AA)
        object_image = cv2.bitwise_and(mask_inv, image)
        objects_path = os.path.join(target_path, 'objects')
        if not os.path.isdir(objects_path):
            os.mkdir(objects_path)
        object_name = "object_" + str(index) + '.jpg'
        save_path = os.path.join(objects_path, object_name)
        cv2.imwrite(save_path, object_image)
        object_names.append(object_name)

    # Generating and saving binary mask
    if 'binary_masks' in tasks:
        mask_inv = np.zeros((600, 800), np.uint8)
        cv2.drawContours(mask_inv, [cnt], -1, (255, 255, 255), thickness=-2, lineType=cv2.LINE_AA)
        mask = cv2.bitwise_not(mask_inv)
        binary_masks_path = os.path.join(target_path, 'binary_masks')
        if not os.path.isdir(binary_masks_path):
            os.mkdir(binary_masks_path)
        binary_mask_name = "binary_mask_" + str(index) + '.jpg'
        save_path = os.path.join(binary_masks_path, binary_mask_name)
        cv2.imwrite(save_path, mask)
        binary_mask_names.append(binary_mask_name)



# Saving original image names and corresponding generated image names in csv file(s)
if 'save_csv' in tasks:
    if 'contours' in tasks:
        contour_csv_name = 'contour' + masks_csv_name
        contour_csv_path = os.path.join(target_path, contour_csv_name)
        contours_df = pd.DataFrame({'name': list(one_lid['name']), 'contour_names': contour_names})
        contours_df.to_csv(contour_csv_path)
    if 'objects' in tasks:
        object_csv_name = 'object' + masks_csv_name
        object_csv_path = os.path.join(target_path, object_csv_name)
        objects_df = pd.DataFrame({'name': list(one_lid['name']), 'object_names': object_names})
        objects_df.to_csv(object_csv_path)
    if 'binary_masks' in tasks:
        binary_mask_csv_name = 'binary' + masks_csv_name
        binary_csv_path = os.path.join(target_path, binary_mask_csv_name)
        binary_df = pd.DataFrame({'name': list(one_lid['name']), 'binary_mask_names': binary_mask_names})
        binary_df.to_csv(binary_csv_path)



