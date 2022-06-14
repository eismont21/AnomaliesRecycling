import pandas as pd
import numpy as np
import cv2
import os
import seaborn as sns
import matplotlib.pylab as plt
from data_augmentation.augmentation_image import AugmentationImage

#STORE_DIR = "/cvhci/temp/p22g5/"
STORE_DIR = "/home/dmitrii/GitHub/AnomaliesRecycling/POLYSECURE"
HOME_DIR = "/home/p22g5/AnomaliesRecycling/"


class DataAugmentation:
    def __init__(self, data_dir, target_dir, one_lid_dir):
        self.DATA_DIR = STORE_DIR + data_dir
        self.TARGET_DIR = STORE_DIR + target_dir
        self.ONE_LID_DIR = one_lid_dir
        self.STANDARD_RESOLUTION = (600, 800)
        self.masks = None
        self.percentile_binary_mask = None

    def extract_masks(self):
        one_lid = pd.read_csv(self.ONE_LID_DIR)
        self.masks = []

        for index, row in one_lid.iterrows():
            if index % 100 == 0:
                print('Generating and saving masks for image ' + str(index) + ' of ' + str(len(one_lid)))
            # Get image path
            image_path = os.path.join(self.DATA_DIR, row['name'][1:])
            # Get image
            img = cv2.imread(image_path)
            augm_img = AugmentationImage(img)
            augm_img.calculate_contour()
            augm_img.calculate_binary_mask()
            augm_img.calculate_object_mask()
            self.masks.append(augm_img)
        return self.masks

    def get_sum_binary_mask(self, show=True):
        summed_mask = np.zeros(self.STANDARD_RESOLUTION)
        for mask in self.masks:
            binary_mask = cv2.bitwise_not(mask.binary_mask)
            summed_mask += binary_mask
        if show:
            ax = sns.heatmap(summed_mask)
            plt.show()
        return summed_mask

    def get_percentile_sum_binary_mask(self, summed_mask, percentile=95, show=True):
        self.percentile_binary_mask = np.where(summed_mask > np.percentile(summed_mask, percentile), 1, 0)
        if show:
            plt.imshow(self.percentile_binary_mask)
        return self.percentile_binary_mask





