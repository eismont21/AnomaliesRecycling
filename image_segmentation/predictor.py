from detectron2.utils.visualizer import ColorMode
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # specify which GPU(s) to be used
from detectron2.engine import DefaultPredictor
import cv2
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
from detectron2.data import MetadataCatalog


from image_segmentation.train_net import setup
import pandas as pd
from detectron2.utils.logger import setup_logger
setup_logger()

IM_ROOT_DIR = "/cvhci/temp/p22g5/data/"


def predict_and_evaluate(weights_path, test_labels, thresh_test=0.5):
    cfg = setup()
    print("WEIGHTS = ", weights_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh_test
    predictor = DefaultPredictor(cfg)
    results = []
    for i, row in test_labels.iterrows():
        filename = row['name']
        label = row['count']
        im = cv2.imread(os.path.join(IM_ROOT_DIR, filename))
        outputs = predictor(im)
        prediction = len(outputs['instances'])
        new_row = {'name': filename, 'count': label, 'prediction': prediction}
        results.append(new_row)
    return pd.DataFrame.from_records(results)


def predict_and_visualize(weights_path, thresh_test, results, show_label=1):
    cfg = setup()
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh_test
    predictor = DefaultPredictor(cfg)

    for i, row in results.iterrows():
        filename = row['name']
        label = row['count']
        im = cv2.imread(os.path.join(IM_ROOT_DIR, filename))
        outputs = predictor(im)
        prediction = len(outputs['instances'])
        if label == show_label:
            print("LABEL = ", label, "PREDICTED = ", prediction, "FILENAME = ", filename)
            metadata = MetadataCatalog.get("polysecure_dataset_test")
            metadata.thing_classes=['Lid']
            v = Visualizer(im[:, :, ::-1],
                           metadata = metadata,
                           scale=0.8,
                           instance_mode=ColorMode.IMAGE_BW)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            plt.figure(figsize = (14, 10))
            plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
            plt.grid(False)
            plt.show()

