import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm
import random
import seaborn as sns
import matplotlib.pylab as plt
from data_augmentation.augmentation_image import AugmentationImage
from random import randint, shuffle
from skimage.util import random_noise
from pathlib import Path
from data_augmentation.coco_annotations import create_coco_json

STORE_DIR = "/cvhci/temp/p22g5/"
HOME_DIR = "/home/p22g5/AnomaliesRecycling/"
#STORE_DIR = "/home/dmitrii/GitHub/AnomaliesRecycling/POLYSECURE/"
#HOME_DIR = "/home/dmitrii/GitHub/AnomaliesRecycling/"
#STORE_DIR = "C:/Users/Charlotte Goos/Documents/university/ss_22/Praktikum_CVHCI/data/copy_and_paste"
#HOME_DIR = "C:/Users/Charlotte Goos/Documents/university/ss_22/Praktikum_CVHCI/AnomaliesRecycling/"

TEST_INDEXES = [920, 574, 33, 372, 471, 344, 272, 322, 1069, 693, 621, 1025, 541, 780, 861, 194, 167, 990, 642, 585, 552, 1000, 155, 571, 230, 303, 681, 7, 149, 1, 400, 650, 59, 834, 798, 938, 649, 795, 604, 755, 699, 284, 212, 97, 919, 712, 119, 695, 355, 1028, 378, 1049, 198, 684, 1060, 772, 548, 639, 555, 1035, 380, 749, 813, 652, 669, 778, 595, 540, 619, 884, 363, 22, 74, 914, 891, 516, 101, 764, 629, 342, 887, 377, 449, 1019, 763, 845, 66, 961, 888, 597, 370, 151, 267, 104, 1010, 917, 113, 1020, 316, 796, 521, 515, 677, 398, 568, 491, 878, 768, 468, 199, 13, 75, 704, 663, 357, 577, 15, 319, 462, 640, 453, 831, 388, 249, 765, 489, 391, 821, 305, 384, 881, 864, 857, 387, 367, 851, 123, 951, 626, 409, 88, 1040, 1023, 83, 724, 986, 636, 943, 4, 847, 21, 40, 751, 605, 824, 815, 610, 912, 218, 1061, 1070, 368, 144, 952, 3, 191, 229, 507, 707, 921, 438, 495, 508, 989, 454, 310, 326, 551, 812, 186, 958, 890, 1055, 461, 627, 576, 512, 122, 1062, 732, 1018, 996, 586, 686, 259, 379, 630, 895, 108, 915, 983, 125, 714, 691, 717, 1036, 759, 899, 257, 302, 874, 128, 725, 911, 689]


class DataAugmentation:
    """
    Class for Data Augmentation
    """
    def __init__(self, data_dir, zero_lid_dir, one_lid_dir):
        """
        Constructor for the class
        :param data_dir: directory with images
        :param zero_lid_dir: filename of csv file with images that have only one lid for generating of other synthesized images
        :param one_lid_dir: filename of csv file with images that have no lids for generating of other synthesized images
        """
        self.DATA_DIR = os.path.join(STORE_DIR, data_dir)
        self.empty_trays = pd.read_csv(os.path.join(HOME_DIR, zero_lid_dir))
        self.one_lids = pd.read_csv(os.path.join(HOME_DIR, one_lid_dir))
        #self.split_randomly(n=len(self.one_lids), p=0.2)
        self.split_randomly2(len(self.one_lids))
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
        self.iou_tolerance = 0.8
        self.iou_bound = 0.01
        self.edge_case_probability = 0.05
        self.dark_case_probability = 0.15
        self.change_color_case_probability = 0.15
        self.transparent_case_probability = 0.25
        self.rotate_case_probability = 0.5
        self.synthesize_dir = "synthesized"

    def split_randomly2(self, n):
        """
        Use this instead of :split_randomly for controlling the lids for test
        :param n:
        :return:
        """
        indexes = list(range(0, n))
        self.train_indexes = list(set(indexes) - set(TEST_INDEXES))
        self.test_indexes = TEST_INDEXES

    def split_randomly(self, n, p):
        """
        Generate random split for train and test indexes
        :param n: number of items
        :param p: percent of test indexes
        :return:
        """
        indexes = list(range(0, n))
        shuffle(indexes)
        last = int(n * (1 - p))
        self.train_indexes = indexes[:last]
        self.test_indexes = indexes[last:]

    def extract_masks(self):
        """
        Extract binary masks from images with one lid
        :return: binary masks
        """
        self.masks = []
        print('Generating and saving masks for images')
        with tqdm(total=len(self.one_lids), ncols=100) as pbar:
            for index, row in self.one_lids.iterrows():
                image_path = os.path.join(self.DATA_DIR, row['name'])
                img = cv2.imread(image_path)
                tags = self.one_lids.iloc[[index]]
                augm_img = AugmentationImage(img, tags)
                augm_img.calculate_contour()
                augm_img.calculate_binary_mask()
                augm_img.calculate_object_mask()
                self.masks.append(augm_img)
                pbar.update(1)
        return self.masks

    def get_sum_binary_mask(self, show=True):
        """
        Calculate the sum of binary masks in all images. Used for generating heatmap.
        :param show: whether to show the density heatmap
        :return: summed mask
        """
        summed_mask = np.zeros(self.STANDARD_RESOLUTION)
        for mask in self.masks:
            binary_mask = cv2.bitwise_not(mask.get_binary_mask())
            summed_mask += binary_mask
        if show:
            ax = sns.heatmap(summed_mask)
            plt.show()
        return summed_mask

    def get_percentile_sum_binary_mask(self, summed_mask, percentile=95, show=True):
        """
        Calculate the percentile binary mask from summed mask.
        :param summed_mask: summed mask
        :param percentile: percent values
        :param show: whether to show
        :return: percentile binary mask
        """
        self.percentile_binary_mask = np.where(summed_mask > np.percentile(summed_mask, percentile), 1, 0)
        if show:
            plt.imshow(self.percentile_binary_mask)
        return self.percentile_binary_mask

    def get_random_position(self):
        """
        Get random point from percentile binary mask.
        :return: (x,y) point for the center of the lid to insert
        """
        indexes = self.percentile_binary_mask.nonzero()
        i = randint(0, np.shape(indexes)[1] - 1)
        x, y = indexes[1][i], indexes[0][i]
        return x, y

    def get_random_background(self):
        """
        Get random background from zero lid images (empry tray).
        :return: background image
        """
        i = randint(0, len(self.empty_trays)-1)
        empty_tray_name = self.empty_trays.iloc[i]['name']
        empty_tray_path = os.path.join(self.DATA_DIR, empty_tray_name)
        empty_tray = cv2.imread(empty_tray_path)
        return empty_tray
    
    def get_random_object(self, pick_from="all"):
        """
        Ger random object (lid)
        :param pick_from: where to take: train, test, all
        :return: random index in the self.mask
        """
        if pick_from == "train":
            i = self.train_indexes[randint(0, len(self.train_indexes)-1)]
        elif pick_from == "test":
            i = self.test_indexes[randint(0, len(self.test_indexes)-1)]
        else:
            i = randint(0, len(self.masks) - 1)
        return i

    def copy_and_paste(self, label, rotate, change_color, make_dark, transparent, pick_from='all'):
        """
        Generate one synthesized image with copy and paste method.
        :param label: number of objects
        :param rotate: rotate the inserted object with random angle
        :param change_color: change color of inserted object
        :param make_dark: make the inserted object dark
        :param transparent: make the inserted object transparent
        :param pick_from: generate from train, test, or all
        :return: synthesized image, its tags, its binary object masks
        """
        object_binary_masks = []
        background = self.get_random_background()
        d = [np.nan for i in range(len(self.inherited_tags))]
        df = pd.DataFrame([d], columns=self.inherited_tags)
        img_name = "label_" + str(label) + "_" + "img_" + str(0) + ".jpg"
        name = os.path.join(self.synthesize_dir, img_name)
        if label == 0:
            return self.get_noise_img(background), \
                   pd.DataFrame([[name, 0, 0, np.nan, np.nan, 0, 0, 0, 0, 0, 1]], columns=self.inherited_tags), \
                   object_binary_masks
        bbs = []
        for j in range(label):
            while True:
                flag_dark = False
                if make_dark:
                    flag_dark = random.random() < self.dark_case_probability
                
                flag_color = False
                if change_color and not(flag_dark):
                    flag_color = random.random() < self.change_color_case_probability
                
                flag_rotate = False
                if rotate:
                    flag_rotate = random.random() < self.rotate_case_probability
                if flag_rotate:
                    angle = randint(1, 360)
                else:
                    angle = 0
                
                flag_edge = random.random() < self.edge_case_probability
                if flag_edge:
                    x, y = random.randint(-20, 820), random.choice(list(range(-20, 10)) + list(range(590, 620)))
                else:
                    x, y = self.get_random_position()
                
                i = self.get_random_object(pick_from)
                bb_new = self.masks[i].get_mask_dic(x, y, angle)
                flag, bbs_new = self._check_overlap(bbs, bb_new, self.iou_tolerance)
                if not(flag):
                    bbs = bbs_new.copy()
                    bbs.append(bb_new)
                    break
            
            flag_transparent = False
            if transparent and not(flag_dark):
                flag_transparent = random.random() < self.transparent_case_probability
            
            background, binary_mask = self.masks[i].copy_and_paste(background, x, y, angle, flag_color, flag_dark, flag_transparent)
            if not flag_transparent:
                for k in range(len(object_binary_masks)):
                    bin_xor = cv2.bitwise_xor(binary_mask, object_binary_masks[k])
                    new = cv2.bitwise_and(bin_xor, cv2.bitwise_not(object_binary_masks[k]))
                    object_binary_masks[k] = cv2.bitwise_not(new)
            object_binary_masks.append(binary_mask)
            if flag_transparent:
                df.iloc[0]['transparent'] = 1
            if flag_edge:
                df.iloc[0]['edge'] = 1
            if flag_dark:
                df.iloc[0]['dark color'] = 1
            self._combine_tags(df, self.masks[i].tags)
            
        for bb in bbs:
            if bb['overlapped'] > self.iou_bound:
                df.iloc[0]['overlapping'] = 1
                break

        df.iloc[0]['count'] = label
        df.iloc[0]['synthesized'] = 1
        df[['name']] = df[['name']].astype(str)
        df.at[0, 'name'] = str(name)
        tags = ["count", "edge", "transparent", "inside", "overlapping", "dark color", "open lid", "synthesized"]
        df[tags] = df[tags].astype(int)
        return background, df, object_binary_masks
    
    def _check_overlap(self, bbs, bb_new, tol):
        flag = False
        bbs_new = []
        if bb_new['overlapped'] > tol:
            return True, bbs_new
        for bb in bbs:
            mask_dic_new = {}
            mask_combined = cv2.bitwise_and(bb_new['mask'], bb['mask'])        
            n_pixel_combined = cv2.countNonZero(mask_combined)            
            if n_pixel_combined / bb['size'] + bb['overlapped'] < tol:
                mask_dic_new['size'] = bb['size']
                mask_dic_new['mask'] = cv2.bitwise_xor(bb['mask'], mask_combined)
                mask_dic_new['overlapped'] = bb['overlapped'] + n_pixel_combined / bb['size']
                bbs_new.append(mask_dic_new)
            else:
                flag = True
                break
        if flag:
            return flag, bbs
                
        return flag, bbs_new

    def _combine_tags(self, result, new):
        """
        Combine tags for the synthesized image from the original images
        :param result: resulting tags
        :param new: new tag
        :return:
        """
        for tag in self.inherited_tags:
            if tag in ["edge", "transparent", "inside", "overlapping", "dark color", "open lid"]:
                if np.isnan(result.iloc[0][tag]):
                    result.iloc[0][tag] = int(new.iloc[0][tag])
                else:
                    result.iloc[0][tag] = int(result.iloc[0][tag]) or int(new.iloc[0][tag])

    def generate(self, classes, rotate=True, change_color=False, make_dark=False, transparent=False, coco_annotation=True, data_dir_name='data'):
        """
        Generate synthesized images. Main method in the class.
        :param classes: which images to synthesize e.g. {1: 2} synthesizes 2 images with 1 object
        :param rotate: rotate the inserted object with random angle
        :param change_color: change color of inserted object
        :param make_dark: make the inserted object dark
        :param transparent: make the inserted object transparent
        :param coco_annotation: create a coco annotation labels
        :param data_dir_name: train, test or all
        :return:
        """
        new_csv = pd.DataFrame()
        synthesized_dir = os.path.join(self.DATA_DIR, self.synthesize_dir)
        annotations_dir = os.path.join(synthesized_dir, 'annotations_' + data_dir_name)
        data_dir = os.path.join(synthesized_dir, data_dir_name)
        Path(synthesized_dir).mkdir(exist_ok=True)
        Path(annotations_dir).mkdir(exist_ok=True)
        Path(data_dir).mkdir(exist_ok=True)
        print('Generate images:')
        with tqdm(total=sum(classes.values()), ncols=100) as pbar:
            for label in classes:
                for i in range(classes[label]):
                    img, df, bin_masks = self.copy_and_paste(label, rotate, change_color, make_dark, transparent, data_dir_name)
                    img_id = "label_" + str(label) + "_" + "img_" + str(i)
                    img_name = img_id + ".jpg"
                    for j in range(len(bin_masks)):
                        bin_name = os.path.join(annotations_dir, img_id + '_' + 'lid' + '_' + str(j) + '.jpg')
                        cv2.imwrite(bin_name, bin_masks[j])
                    name = os.path.join(data_dir, img_name)
                    df.at[0, 'name'] = name
                    new_csv = pd.concat([new_csv, df], ignore_index=True)
                    filename = os.path.join(self.DATA_DIR, name)
                    cv2.imwrite(filename, img)
                    pbar.update(1)
        new_csv.to_csv(os.path.join(synthesized_dir, 'synthesized_' + data_dir_name + '.csv'), index=False)
        if coco_annotation:
            print('Create coco annotation:')
            n_annotations = 0
            for key, value in classes.items():
                n_annotations += key * value
            create_coco_json(data_dir, annotations_dir, synthesized_dir, 'coco_' + data_dir_name, n_annotations)

    @staticmethod
    def get_noise_img(img, mode='gaussian'):
        """
        Add noise to the image noisy image.
        :param img: input image
        :param mode: type of noise
        :return: image with noise
        """
        return np.array(255*random_noise(img, mode=mode), dtype='uint8')
