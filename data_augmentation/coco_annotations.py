import datetime
import json
import os.path
import cv2
import numpy as np
from detectron2.structures import BoxMode
import pycocotools.mask

INFO = {
    'description': 'Synthesized Coco Dataset',
    'url': '',
    'version': '1.0',
    'year': 2022,
    'contributor': 'CVHCIPraktikum',
    'date_created': datetime.datetime.utcnow().isoformat(' ')
}
LICENSES = [
    {
        'url': ' ',
        'id': 1,
        'name': ' '
    }
]
CATEGORIES = [
    {
        'id': 1,
        'name': 'Lid',
        'supercategory': 'object'
    }
]

coco_output = {
    'info': INFO,
    'licenses': LICENSES,
    'categories': CATEGORIES,
    'images': [],
    'annotations': []
}


def create_coco_json(image_dir, annotation_dir, root_dir, anno_filename):
    image_id = 0
    annotation_id = 0
    for filename in [file for file in os.listdir(image_dir) if file.endswith('.jpg')]:
        jpg_image = os.path.join(image_dir, filename)
        image_name = os.path.splitext(filename)[0]
        image_info = {
            'id': image_id,
            'file_name': filename,
            'width': 800,
            'height': 600
        }
        coco_output['images'].append(image_info)
        for annotation_name in [annotation for annotation in os.listdir(annotation_dir) if annotation.endswith('jpg') and image_name in annotation]:
            jpg_annotation = os.path.join(annotation_dir, annotation_name)
            annotation_image = cv2.imread(jpg_annotation, flags=cv2.IMREAD_GRAYSCALE)
            annotation_image = cv2.bitwise_not(annotation_image)
            #annotation_image.clip(max=1)
            #rle = pycocotools.mask.encode(np.asfortranarray(annotation_image))
            #rle_dict = dict()
            #rle_dict["size"] = list(rle['size'])
            #rle_dict["counts"] = rle['counts'].decode("utf-8")
            #area = pycocotools.mask.area(rle)
            #x, y, w, h = cv2.boundingRect(annotation_image)
            cnt, hierachy = cv2.findContours(annotation_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(cnt, key=cv2.contourArea, reverse=True)
            contours = contours[0]
            x, y, w, h = cv2.boundingRect(contours)
            area = cv2.contourArea(contours)
            poly = []
            for pos in contours:
                poly.append(float(pos[0][0]))
                poly.append(float(pos[0][1]))

            annotation_info = {
                'id': annotation_id,
                'image_id': image_id,
                'category_id': 1,
                'segmentation': [poly],
                'area': int(area),
                'bbox': [int(x), int(y), int(w), int(h)],
                'bbox_mode': BoxMode.XYWH_ABS,
                'iscrowd': 0
            }

            coco_output['annotations'].append(annotation_info)
            annotation_id += 1
            #print("annotaion_id = ", annotation_id)
        image_id += 1
    json_string = json.dumps(coco_output, indent=4)
    json_path = os.path.join(root_dir, anno_filename + '.json')
    json_file = open(json_path, 'w')
    json_file.write(json_string)
    json_file.close()







