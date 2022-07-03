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
    'year': int(2022),
    'contributor': 'CVHCIPraktikum',
    'date_created': datetime.datetime.utcnow().isoformat(' ')
}
LICENSES = [
    {
        'url': ' ',
        'id': int(1),
        'name': ' '
    }
]
CATEGORIES = [
    {
        'id': int(1),
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


def create_coco_json(image_dir, annotation_dir, root_dir):
    image_id = 0
    annotation_id = 0
    for filename in [file for file in os.listdir(image_dir) if file.endswith('.jpg')]:
        jpg_image = os.path.join(image_dir, filename)
        image_name = os.path.splitext(filename)[0]
        image_info = {
            'image_id': image_id,
            'file_name': jpg_image,
            'width': int(800),
            'height': int(600)
        }
        coco_output['images'].append(image_info)
        for annotation_name in [annotation for annotation in os.listdir(annotation_dir) if annotation.endswith('jpg') and image_name in annotation]:
            jpg_annotation = os.path.join(annotation_dir, annotation_name)
            annotation_image = cv2.imread(jpg_annotation, flags=cv2.IMREAD_GRAYSCALE)
            annotation_image = cv2.bitwise_not(annotation_image)
            rle = pycocotools.mask.encode(np.asfortranarray(annotation_image))
            area = pycocotools.mask.area(rle)
            [x, y, w, h] = cv2.boundingRect(annotation_image)

            annotation_info = {
                'id': annotation_id,
                'image_id': image_id,
                'category_id': int(1),
                'segmentation': rle,
                'area': area,
                'bbox': [x, y, w, h],
                'bbox_mode': BoxMode.XYWH_ABS,
                'iscrowd': int(0)
            }

            coco_output['annotations'].append(annotation_info)
            annotation_id += 1
        image_id += 1
    json_string = json.dumps(str(coco_output))
    json_path = os.path.join(root_dir, 'coco.json')
    json_file = open(json_path, 'w')
    json_file.write(json_string)
    json_file.close()







