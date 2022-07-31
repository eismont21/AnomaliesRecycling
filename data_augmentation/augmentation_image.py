import cv2
import numpy as np
from scipy.spatial import distance
from random import randint
import imutils
import random


class AugmentationImage:
    """
    Class for Image of Object that is copied & pasted
    """
    def __init__(self, image, tags):
        """
       Constructor for the class
       :param image: image of object
       :param tags: tags of image in
       """
        self.image = image
        self.tags = tags
        self.cnt = None
        self.binary_mask = None
        self.object_mask = None

    def calculate_contour(self):
        """
        Calculates contour of object
        """
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
        """
        Calculates binary mask of object
        """
        if self.cnt is None:
            self.calculate_contour()
        mask_inv = np.zeros((600, 800), np.uint8)
        cv2.drawContours(mask_inv, [self.cnt], -1, (255, 255, 255), thickness=-2, lineType=cv2.LINE_AA)
        self.binary_mask = cv2.bitwise_not(mask_inv)
        return

    def calculate_object_mask(self):
        """
        Calculates object mask (image of object on black background)
        """
        if self.cnt is None:
            self.calculate_contour()
        mask_inv = np.zeros((600, 800, 3), np.uint8)
        cv2.drawContours(mask_inv, [self.cnt], -1, (255, 255, 255), thickness=-2, lineType=cv2.LINE_AA)
        self.object_mask = cv2.bitwise_and(mask_inv, self.image)
        return

    def get_contour(self):
        """
        Calculates contour if necessary and returns it
        return: contour of object
        """
        if self.cnt is None:
            self.calculate_contour()
        return self.cnt

    def get_binary_mask(self):
        """
        Calculates binary mask if necessary and returns it
        return: binary mask of object
        """
        if self.binary_mask is None:
            self.calculate_binary_mask()
        return self.binary_mask

    def get_object_mask(self):
        """
        Calculates object mask (image of object on black background) if necessary and returns it
        return: object mask
        """
        if self.object_mask is None:
            self.calculate_object_mask()
        return self.object_mask

    def copy_and_paste(self, background, x_center, y_center, angle, change_color, make_dark, transparent):
        """
        Pastes object to a background
        param background: image of background
        param x_center: TODO
        param y_center: TODO
        param angle: angle with which object is rotated before insertion
        param change_color: tag which decides if the object color is changed
        param make_dark: tag which decides if object is made dark
        param transparent: tag which decides if object is made transparent
        return: new image with object pasted to background, binary mask of object with rotation and new position
        """
        x, y, w, h = cv2.boundingRect(self.cnt) # find BB from contour
        # crop this BB to get only the lid
        binary_mask, fg = self.get_rotated_object(x, y, w, h, angle, change_color)
        if make_dark:
            fg = self.reduce_brightness(fg)
        h, w = binary_mask.shape[0], binary_mask.shape[1]
        x_new_left_up = int(x_center - (w / 2))
        y_new_left_up = int(y_center - (h / 2))
        x_new_right_down = x_new_left_up + w
        y_new_right_down = y_new_left_up + h
        x_binary_mask_left_up, y_binary_mask_left_up = 0, 0
        x_binary_mask_right_down, y_binary_mask_right_down = w, h
        if x_new_left_up < 0:
            x_binary_mask_left_up = abs(x_new_left_up)
            x_new_left_up = 0
        if x_new_right_down > 800:
            x_binary_mask_right_down -= x_new_right_down - 800
            x_new_right_down = 800
        if y_new_left_up < 0:
            y_binary_mask_left_up = abs(y_new_left_up)
            y_new_left_up = 0
        if y_new_right_down > 600:
            y_binary_mask_right_down -= y_new_right_down - 600
            y_new_right_down = 600
        roi = background[y_new_left_up: y_new_right_down, x_new_left_up:x_new_right_down]
        binary_mask = binary_mask[y_binary_mask_left_up:y_binary_mask_right_down, x_binary_mask_left_up:x_binary_mask_right_down]
        fg = fg[y_binary_mask_left_up:y_binary_mask_right_down, x_binary_mask_left_up:x_binary_mask_right_down]
        bg = cv2.bitwise_or(roi, roi, mask=binary_mask)
        fg = self._get_transparent_object(binary_mask, roi, fg, transparent)
        final_roi = cv2.add(bg, fg)
        cropped_object = final_roi
        background[y_new_left_up: y_new_right_down, x_new_left_up:x_new_right_down] = cropped_object
        bin_mask = cv2.bitwise_not(np.zeros((600, 800), np.uint8))
        bin_mask[y_new_left_up: y_new_right_down, x_new_left_up:x_new_right_down] = binary_mask

        return background, bin_mask

    def _get_transparent_object(self, binary_mask, roi, fg, transparent):
        if transparent:
            mask_inv = cv2.cvtColor(cv2.bitwise_not(binary_mask), cv2.COLOR_GRAY2RGB)
            bg_fg = cv2.bitwise_and(mask_inv, roi)
            fg = cv2.addWeighted(bg_fg, 0.4, fg, 0.6, 0)
        return fg

    def get_rotated_object(self, x, y, w, h, angle, change_color):
        """
        Rotates object with given angle and changes color if tag is set
        param x: x coordinate of lower left corner of object bounding box
        param y: y coordinate of lower left corner of object bounding box
        param w: width of object bounding box
        param h: height of object bounding box
        param angle: angle with which to rotate object
        param change_color: tag which determines if color of object is changed
        """
        rotated_bin_inv = imutils.rotate_bound(cv2.bitwise_not(self.get_binary_mask()[y:y + h, x:x + w]), angle)
        rotated_bin = cv2.bitwise_not(rotated_bin_inv)
        rotated_fg = imutils.rotate_bound(self.change_object_color(change_color)[y: y + h, x:x + w], angle)
        return rotated_bin, rotated_fg

    def change_object_color(self, change_color):
        """
        Changes color of object in object mask if tag is set
        param change_color: tag which decides if color of object is changed
        return: object mask of object with changed color
        """
        if self.cnt is None:
            self.calculate_contour()
        is_change = randint(0, 1)
        if is_change == 0 or not change_color:
            return self.object_mask
        else:
            r = randint(0, 255)
            g = randint(0, 255)
            b = randint(0, 255)
            color_image = np.full(self.object_mask.shape, (r, g, b), np.uint8)
            color_object_image = cv2.addWeighted(color_image, 0.2, cv2.cvtColor(cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB), 0.8, 0)
            mask_inv = np.zeros((600, 800, 3), np.uint8)
            cv2.drawContours(mask_inv, [self.cnt], -1, (255, 255, 255), thickness=-2, lineType=cv2.LINE_AA)
            object_mask_color = cv2.bitwise_and(mask_inv, color_object_image)
            return object_mask_color

    def get_bb(self, background, x_center, y_center, angle, change_color):
        x, y, w, h = cv2.boundingRect(self.cnt)  # find BB from contour
        if not angle == 0:
            rotated, _ = self.get_rotated_object(x, y, w, h, angle, change_color=change_color)
            h, w = rotated.shape[0], rotated.shape[1]
        x_offset = int(x_center - (w / 2))  # left up coord
        y_offset = int(y_center - (h / 2))  # left up coord
        assert x_offset > 0 and y_offset > 0, "negative coordinates"
        # crop this BB to get only the lid
        binary_mask = self.get_binary_mask()[y:y + h, x:x + w]
        x_end = x_offset + binary_mask.shape[1]  # right down coord
        y_end = y_offset + binary_mask.shape[0]  # right down coord
        assert x_end < background.shape[1] and y_end < background.shape[0], "coordinates out of range"
        return {'x1': x_offset, 'x2': x_end, 'y1': y_offset, 'y2': y_end}
    
    def get_mask_dic(self, x, y, angle):
        def get_offset(x, y, countur):
            M = cv2.moments(countur)
            x_c, y_c = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            return x - x_c, y - y_c
        
        countur = self.get_contour()
        if angle != 0:
            countur = self.rotate_contour(countur, angle)
            
        mask_original = cv2.drawContours(np.zeros((600, 800), np.uint8), [countur], -1, (255, 255, 255), thickness=-2,
                                         lineType=cv2.LINE_AA)
        
        mask = cv2.drawContours(np.zeros((600, 800), np.uint8), [countur], -1, (255, 255, 255), 
                                offset=get_offset(x, y, countur), thickness=-2, lineType=cv2.LINE_AA)
        
        mask_dic = {'size': cv2.countNonZero(mask), 
                    'mask': mask, 
                    'overlapped': 1.0 - cv2.countNonZero(mask)/cv2.countNonZero(mask_original)}
        return mask_dic

    def rotate_contour(self, cnt, angle):
        def cart2pol(x, y):
            theta = np.arctan2(y, x)
            rho = np.hypot(x, y)
            return theta, rho

        def pol2cart(theta, rho):
            x = rho * np.cos(theta)
            y = rho * np.sin(theta)
            return x, y
        
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        cnt_norm = cnt - [cx, cy]
    
        coordinates = cnt_norm[:, 0, :]
        xs, ys = coordinates[:, 0], coordinates[:, 1]
        thetas, rhos = cart2pol(xs, ys)
    
        thetas = np.rad2deg(thetas)
        thetas = (thetas + angle) % 360
        thetas = np.deg2rad(thetas)
    
        xs, ys = pol2cart(thetas, rhos)
    
        cnt_norm[:, 0, 0] = xs
        cnt_norm[:, 0, 1] = ys

        cnt_rotated = cnt_norm + [cx, cy]
        cnt_rotated = cnt_rotated.astype(np.int32)

        return cnt_rotated
    
    def reduce_brightness(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        max_brightness = random.randint(8, 30)
        value = int(max_brightness - np.max(v))
        v = cv2.add(v, value)
        v[v > 255] = 255
        v[v < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR) 
        img = cv2.blur(img, (9,9))
        
        return img












