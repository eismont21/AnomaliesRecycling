import pandas as pd
import cv2

import os

import numpy as np

########################################################################################################################
# Example copy and paste
# To-do: automatic copy and paste
# To-do: Scaling, Rotation, Shifting, Color-change
# Make sure to add the same rotation/scaling/shift to both binary_mask and object_mask
# Color changes only to object_mask
# May need bounding boxes in the future to check if whole object is in image and to create edge images
########################################################################################################################

# Add path to image data (should contain cluttered and presorted folders)"
path_polysecure = ""

# Add path to masks directory where masks (generated with masks.py) are saved.
# Should contain binary masks & objects directory and binarymasks.csv & objectmasks.csv
masks_path = ""

# Add path to directory where results can be saved
target_path = ""

object_csv_path = os.path.join(masks_path, 'objectmasks.csv')
object_masks = pd.read_csv(object_csv_path)

binary_csv_path = os.path.join(masks_path, 'binarymasks.csv')
binary_masks = pd.read_csv(binary_csv_path)

#if not object_masks['name'] == binary_masks['name']:
#    print('binarymasks.csv and objectmasks.csv not compatible')

empty_trays = pd.read_csv('data/zero_lid.csv')


def get_empty_tray(index):
    empty_tray_name = empty_trays.iloc[index]['name'][1:]
    empty_tray_path = os.path.join(path_polysecure, empty_tray_name)
    empty_tray = cv2.imread(empty_tray_path)
    return empty_tray


def get_object_and_binary_masks(index):
    object_name = object_masks.iloc[index]['object_names']
    object_path = os.path.join(masks_path, 'objects', object_name)
    object_mask = cv2.imread(object_path)

    binary_name = binary_masks.iloc[index]['binary_mask_names']
    binary_path = os.path.join(masks_path, 'binary_masks', binary_name)
    binary_mask_rbg = cv2.imread(binary_path)
    binary_mask = cv2.cvtColor(binary_mask_rbg, cv2.COLOR_RGB2GRAY)
    return object_mask, binary_mask


def copy_and_paste(background, object_mask, binary_mask):
    cv2.imshow('binary_mask', binary_mask)
    bg = cv2.bitwise_or(background, background, mask=binary_mask)
    cv2.imshow('bg', bg)
    generated_image = cv2.add(bg, object_mask)
    return generated_image


empty_tray = get_empty_tray(0)
object_mask, binary_mask = get_object_and_binary_masks(35)
background = copy_and_paste(empty_tray, object_mask, binary_mask)
cv2.imshow('first_copy_and_paste', background)

object_mask, binary_mask = get_object_and_binary_masks(120)
final_image = copy_and_paste(background, object_mask, binary_mask)
cv2.imshow('final_image', final_image)

image_name = os.path.join(target_path, 'first_image.jpg')
cv2.imwrite(image_name, final_image)

cv2.waitKey()
cv2.destroyAllWindows()




