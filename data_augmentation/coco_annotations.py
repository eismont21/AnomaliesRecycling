import datetime
import json
import os.path
import cv2
import numpy as np
from tqdm import tqdm
from detectron2.structures import BoxMode
from scipy.interpolate import splprep, splev

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


def create_coco_json(image_dir, annotation_dir, root_dir, anno_filename, n_annotations):
    image_id = 0
    annotation_id = 0
    with tqdm(total=n_annotations, ncols=100) as pbar:
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
                cnt, _ = cv2.findContours(annotation_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(cnt, key=cv2.contourArea, reverse=True)
                try:
                    contour = contours[0]
                except IndexError:
                    continue
                x, y = contour.T
                x = x.tolist()[0]
                y = y.tolist()[0]
                tck, u = splprep([x, y], s=1.0)
                u_new = np.linspace(u.min(), u.max(), 25)
                x_new, y_new = splev(u_new, tck, der=0)
                res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
                contour = np.asarray(res_array, dtype=np.int32)
    
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
    
                poly = []
                for pos in contour:
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
                pbar.update(1)
                #print("annotaion_id = ", annotation_id)
            image_id += 1
    json_string = json.dumps(coco_output, indent=4)
    json_path = os.path.join(root_dir, anno_filename + '.json')
    if os.path.isfile(json_path):
        os.remove(json_path)
    json_file = open(json_path, 'w')
    json_file.write(json_string)
    json_file.close()






