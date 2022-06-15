import pandas as pd
import numpy as np
import cv2
import os
import seaborn as sns
import matplotlib.pylab as plt
from data_augmentation.augmentation_image import AugmentationImage
from random import randint
from skimage.util import random_noise
import csv
from pathlib import Path

STORE_DIR = "/cvhci/temp/p22g5/"
#STORE_DIR = "/home/dmitrii/GitHub/AnomaliesRecycling/POLYSECURE"
HOME_DIR = "/home/p22g5/AnomaliesRecycling/"


class DataAugmentation:
    def __init__(self, data_dir, target_dir, zero_lid_dir, one_lid_dir):
        self.DATA_DIR = STORE_DIR + data_dir
        self.TARGET_DIR = STORE_DIR + target_dir
        self.empty_trays = pd.read_csv(HOME_DIR+zero_lid_dir)
        self.one_lids = pd.read_csv(HOME_DIR+one_lid_dir)
        self.STANDARD_RESOLUTION = (600, 800)
        self.masks = None
        self.percentile_binary_mask = None

    def extract_masks(self):
        self.masks = []
        for index, row in self.one_lids.iterrows():
            if index % 100 == 0:
                print('Generating and saving masks for image ' + str(index) + ' of ' + str(len(self.one_lids)))
            image_path = os.path.join(self.DATA_DIR, row['name'][1:])
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
            binary_mask = cv2.bitwise_not(mask.get_binary_mask())
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

    def get_random_position(self):
        indexes = self.percentile_binary_mask.nonzero()
        i = randint(0, np.shape(indexes)[1])
        x, y = indexes[1][i], indexes[0][i]
        return x, y

    def get_random_background(self):
        i = randint(0, len(self.empty_trays))
        empty_tray_name = self.empty_trays.iloc[i]['name'][1:]
        empty_tray_path = os.path.join(self.DATA_DIR, empty_tray_name)
        empty_tray = cv2.imread(empty_tray_path)
        return empty_tray

    def copy_and_paste(self, label):
        background = self.get_random_background()
        if label == 0:
            return self.get_noise_img(background)
        for j in range(label):
            x, y = self.get_random_position()
            i = randint(0, len(self.masks))
            background = self.masks[i].copy_and_paste(background, x, y)
        return background

    def generate(self, classes):
        header = ['name', 'count', 'synthesized']
        data = []
        synthesize_dir = "synthesized"
        Path(os.path.join(self.DATA_DIR, synthesize_dir)).mkdir(exist_ok=True)
        for label in classes:
            for i in range(classes[label]):
                img = self.copy_and_paste(label)
                img_name = "label_" + str(label) + "_"+ "img_" + str(i)+".jpg"
                name = os.path.join(synthesize_dir, img_name)
                filename = os.path.join(self.DATA_DIR, name)
                cv2.imwrite(filename, img)
                data.append([name, label, True])
        with open(HOME_DIR + 'data/labels/synthesized/synthesized.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)

    @staticmethod
    def get_noise_img(img, mode='gaussian'):
        return np.array(255*random_noise(img, mode=mode), dtype='uint8')







