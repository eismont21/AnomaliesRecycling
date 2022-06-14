from augmentation_image import AugmentationImage
import pandas as pd
import os
import cv2

# Dataframe lists all images which contain one object and (hopefully) no black-transparent objects
one_lid = pd.read_csv('data/one_lid.csv')

# Add path to image data (should contain cluttered and presorted folders)"
path_polysecure = "C:/Users/Charlotte Goos/Documents/university/ss_22/Praktikum_CVHCI/data/POLYSECURE"
empty_trays = pd.read_csv('data/zero_lid.csv')


def get_empty_tray(index):
    empty_tray_name = empty_trays.iloc[index]['name'][1:]
    empty_tray_path = os.path.join(path_polysecure, empty_tray_name)


    empty_tray = cv2.imread(empty_tray_path)
    return empty_tray


def copy_and_paste(background, object_mask, binary_mask):
    cv2.imshow('binary_mask', binary_mask)
    print(binary_mask.shape)
    bg = cv2.bitwise_or(background, background, mask=binary_mask)
    cv2.imshow('bg', bg)
    generated_image = cv2.add(bg, object_mask)
    return generated_image


def generate_all_augm_iamges():
    augm_images = []
    for index, row in one_lid.iterrows():
        if index == 130:
            break
        image_path = os.path.join(path_polysecure, row['name'][1:])
        image = cv2.imread(image_path)
        augm_image = AugmentationImage(image)
        augm_images.append(augm_image)
    return augm_images

augm_images = generate_all_augm_iamges()
empty_tray = get_empty_tray(0)
object_mask = augm_images[35].get_object_mask()
binary_mask = augm_images[35].get_binary_mask()
print(binary_mask.shape)
background = copy_and_paste(empty_tray, object_mask, binary_mask)

cv2.imshow('first_copy_and_paste', background)

object_mask = augm_images[120].get_object_mask()
binary_mask = augm_images[120].get_binary_mask()
final_image = copy_and_paste(background, object_mask, binary_mask)
#final_image = cv2.bilateralFilter(final_image, 5, 5, 5)
cv2.imshow('final_image', final_image)
cv2.waitKey()
cv2.destroyAllWindows()

