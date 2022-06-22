import pandas as pd
import numpy as np
import cv2
import os
import seaborn as sns
import matplotlib.pylab as plt
from augmentation_image import AugmentationImage
from random import randint
from skimage.util import random_noise
from pathlib import Path

STORE_DIR = "/cvhci/temp/p22g5/"
HOME_DIR = "/home/p22g5/AnomaliesRecycling/"
#STORE_DIR = "/home/dmitrii/GitHub/AnomaliesRecycling/POLYSECURE/"
#HOME_DIR = "/home/dmitrii/GitHub/AnomaliesRecycling/"


class DataAugmentation:
    def __init__(self, data_dir, zero_lid_dir, one_lid_dir):
        self.DATA_DIR = STORE_DIR + data_dir
        self.empty_trays = pd.read_csv(os.path.join(HOME_DIR, zero_lid_dir))
        self.one_lids = pd.read_csv(os.path.join(HOME_DIR, one_lid_dir))
        self.STANDARD_RESOLUTION = (600, 800)
        self.masks = None
        self.percentile_binary_mask = None
        self.inherited_tags = ["name",
                               "count",
                               "edge",
                               "different colors",
                               "one color",
                               "transparent",
                               "inside",
                               "overlapping",
                               "dark color",
                               "open lid",
                               "synthesized"]
        self.iou_tolerance = 0.9
        self.iou_bound = 0.05
        self.synthesize_dir = "synthesized"

    def extract_masks(self):
        self.masks = []
        for index, row in self.one_lids.iterrows():
            if index % 100 == 0:
                print('Generating and saving masks for image ' + str(index) + ' of ' + str(len(self.one_lids)))
            image_path = os.path.join(self.DATA_DIR, row['name'])
            img = cv2.imread(image_path)
            tags = self.one_lids.iloc[[index]]
            augm_img = AugmentationImage(img, tags)
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
        i = randint(0, len(self.empty_trays)-1)
        empty_tray_name = self.empty_trays.iloc[i]['name']
        empty_tray_path = os.path.join(self.DATA_DIR, empty_tray_name)
        empty_tray = cv2.imread(empty_tray_path)
        return empty_tray

    def copy_and_paste(self, label):
        background = self.get_random_background()
        d = [np.nan for i in range(len(self.inherited_tags))]
        df = pd.DataFrame([d], columns=self.inherited_tags)
        img_name = "label_" + str(label) + "_" + "img_" + str(0) + ".jpg"
        name = os.path.join(self.synthesize_dir, img_name)
        if label == 0:
            return self.get_noise_img(background), \
                   pd.DataFrame([[name, 0, 0, np.nan, np.nan, 0, 0, 0, 0, 0, 1]], columns=self.inherited_tags)
        bbs = []
        for j in range(label):
            while(True):
                x, y = self.get_random_position()
                i = randint(0, len(self.masks) - 1)
                bb_new = self.masks[i].get_bb(background, x, y)
                if not self._check_overlap(bbs, bb_new, self.iou_tolerance):
                    break
                else:
                    print("Overlapping! Generate new!")
            background = self.masks[i].copy_and_paste(background, x, y)
            df.iloc[0]['overlapping'] = int(self._check_overlap(bbs, bb_new, self.iou_bound))
            self.combine_tags(df, self.masks[i].tags)
            bbs.append(bb_new)
        df.iloc[0]['count'] = label
        df.iloc[0]['synthesized'] = 1
        df[['name']] = df[['name']].astype(str)
        df.at[0, 'name'] = str(name)
        tags = ["count", "edge", "transparent", "inside", "overlapping", "dark color", "open lid", "synthesized"]
        df[tags] = df[tags].astype(int)
        return background, df

    def combine_tags(self, result, new):
        for tag in self.inherited_tags:
            if tag in ["edge", "transparent", "inside", "overlapping", "dark color", "open lid"]:
                if np.isnan(result.iloc[0][tag]):
                    result.iloc[0][tag] = int(new.iloc[0][tag])
                else:
                    result.iloc[0][tag] = int(result.iloc[0][tag]) or int(new.iloc[0][tag])

    def _iou(mask1, mask2):
        mask1_area = np.count_nonzero(mask1 == 1)
        mask2_area = np.count_nonzero(mask2 == 1)
        intersection = np.count_nonzero(np.logical_and(mask1, mask2))
        iou = intersection / (mask1_area + mask2_area - intersection)
        return iou

    @staticmethod
    def _check_overlap(bbs, bb_new, tol):
        for bb in bbs:
            iou = get_iou(bb, bb_new)
            if iou > tol:
                return True
        return False

    def generate(self, classes):
        new_csv = pd.DataFrame()
        Path(os.path.join(self.DATA_DIR, self.synthesize_dir)).mkdir(exist_ok=True)
        for label in classes:
            for i in range(classes[label]):
                img, df = self.copy_and_paste(label)
                img_name = "label_" + str(label) + "_" + "img_" + str(i) + ".jpg"
                name = os.path.join(self.synthesize_dir, img_name)
                df.at[0, 'name'] = name
                new_csv = pd.concat([new_csv, df], ignore_index=True)
                filename = os.path.join(self.DATA_DIR, name)
                cv2.imwrite(filename, img)
        output_csv = HOME_DIR + 'data/labels/synthesized/'
        Path(output_csv).mkdir(exist_ok=True)
        new_csv.to_csv(output_csv + 'synthesized.csv', index=False)

    @staticmethod
    def get_noise_img(img, mode='gaussian'):
        return np.array(255*random_noise(img, mode=mode), dtype='uint8')


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou



