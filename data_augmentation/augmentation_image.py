import cv2
import numpy as np
from scipy.spatial import distance


class AugmentationImage():
    #Only make augmentation images with images from the one_lid.csv file
    def __init__(self, image):
        self.image = image
        self.cnt = None
        self.binary_mask = None
        self.object_mask = None

    def calculate_contour(self):
        image_center = (self.image.shape[0] / 2, self.image.shape[1] / 2)
        image_gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        image_filtered = cv2.bilateralFilter(image_gray, 9, 75, 75)
        threshold = 20
        ret, thresh = cv2.threshold(src=image_filtered, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
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
        i = distances.index(min(distances))
        self.cnt = contours[i]
        return

    def calculate_binary_mask(self):
        if self.cnt is None:
            self.calculate_contour()
        mask_inv = np.zeros((600, 800), np.uint8)
        cv2.drawContours(mask_inv, [self.cnt], -1, (255, 255, 255), thickness=-2, lineType=cv2.LINE_AA)
        self.binary_mask = cv2.bitwise_not(mask_inv)
        return

    def calculate_object_mask(self):
        if self.cnt is None:
            self.calculate_contour()
        mask_inv = np.zeros((600, 800, 3), np.uint8)
        cv2.drawContours(mask_inv, [self.cnt], -1, (255, 255, 255), thickness=-2, lineType=cv2.LINE_AA)
        self.object_mask = cv2.bitwise_and(mask_inv, self.image)
        return

    def get_contour(self):
        if self.cnt is None:
            self.calculate_contour()
        return self.cnt

    def get_binary_mask(self):
        if self.binary_mask is None:
            self.calculate_binary_mask()
        return self.binary_mask

    def get_object_mask(self):
        if self.object_mask is None:
            self.calculate_object_mask()
        return self.object_mask

    def copy_and_paste(self, background, x_center, y_center):

        x, y, w, h = cv2.boundingRect(self.cnt) # find BB from contour
        x_offset = int(x_center - (w/2))  # left up coord
        y_offset = int(y_center - (h/2))  # left up coord
        assert x_offset > 0 and y_offset > 0, "negative coordinates"
        # crop this BB to get only the lid
        binary_mask = self.get_binary_mask()[y:y + h, x:x + w]
        fg = self.get_object_mask()[y: y + h, x:x + w]
        x_end = x_offset + binary_mask.shape[1]  # right down coord
        y_end = y_offset + binary_mask.shape[0]  # right down coord
        assert x_end < background.shape[1] and y_end < background.shape[0], "coordinates out of range"

        # tutorial start
        # small_img is cropped_object, large_img is background
        roi = background[y_offset: y_offset + h, x_offset:x_offset + w]
        bg = cv2.bitwise_or(roi, roi, mask=binary_mask)
        final_roi = cv2.add(bg, fg)
        cropped_object = final_roi
        background[y_offset: y_offset + h, x_offset: x_offset + w] = cropped_object

        return background
        #bg = cv2.bitwise_or(background, background, mask=self.binary_mask)
        #generated_image = cv2.add(bg, self.object_mask)
        #return generated_image






