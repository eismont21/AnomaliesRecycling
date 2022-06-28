from detectron2.utils.visualizer import ColorMode
import os
from detectron2.engine import DefaultPredictor
import cv2
from detectron2.utils.visualizer import Visualizer
import random
import matplotlib.pyplot as plt
from detectron2.data import MetadataCatalog, DatasetCatalog

from train_net import setup, ROOT_DIR

from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import default_argument_parser



def predict():
    cfg = setup(args)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.DATASETS.TEST = ("microcontroller_test", )
    predictor = DefaultPredictor(cfg)


    dataset_dicts = get_polysecure_dicts('Microcontroller Segmentation/test')
    for d in random.sample(dataset_dicts, 3):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                       scale=0.8,
                       instance_mode=ColorMode.IMAGE_BW)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.figure(figsize = (14, 10))
        plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        plt.show()


def get_polysecure_dicts(directory):
    return {}


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--im_name", type=str, help="Visualize the prediction for the image", default="samplevalid")
    args = parser.parse_args()
    print("Command Line Args:", args)
    predict()
