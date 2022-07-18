"""
This script is based on the training script in detectron2/tools.
This starts the detectron2 model training.
"""
from datetime import datetime
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # specify which GPU(s) to be used

from detectron2.engine import launch, default_argument_parser
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog
from image_segmentation.PolysecureTrainer import PolysecureTrainer
from image_classification.constants import Constants

from detectron2.checkpoint import DetectionCheckpointer

DATASET_TRAIN_NAME = "polysecure_dataset_train"
DATASET_TEST_NAME = "polysecure_dataset_test"
CONFIG = os.path.join(Constants.PROJECT_DIR.value, "image_segmentation", "configs", "mask_rcnn_R_50_FPN_3x.yaml")

TRAIN_ANNO = os.path.join(Constants.SYNTHESIZE_DIR.value, "coco_train.json")
TEST_ANNO = os.path.join(Constants.SYNTHESIZE_DIR.value, "coco_test.json")

TRAIN_IMAGES = os.path.join(Constants.SYNTHESIZE_DIR.value, "train")
TEST_IMAGES = os.path.join(Constants.SYNTHESIZE_DIR.value, "test")


def setup(args=None):
    DatasetCatalog.clear()
    cfg = get_cfg()
    register_coco_instances(DATASET_TRAIN_NAME, {}, TRAIN_ANNO, TRAIN_IMAGES)
    register_coco_instances(DATASET_TEST_NAME, {}, TEST_ANNO, TEST_IMAGES)
    add_standard_config(cfg)
    time = datetime.now()
    folder_name = time.strftime('%Y-%m-%d_%H-%M-%S/')
    cfg.OUTPUT_DIR = os.path.join(Constants.STORE_DIR.value, "segmentation_experiments", folder_name)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def do_train(args, cfg, trainer):
    trainer.resume_or_load(resume=False)
    #time = datetime.now()
    #folder_name = time.strftime('%Y-%m-%d_%H-%M-%S/')
    #cfg.OUTPUT_DIR = os.path.join(Constants.STORE_DIR.value, "segmentation_experiments", folder_name)
    print("do_train")
    print("model weights: " + cfg.MODEL.WEIGHTS)
    print("output directory: " + cfg.OUTPUT_DIR)
    trainer.train()


def do_test(args, cfg):
    model = PolysecureTrainer.build_model(cfg)
    print("do_test2")
    print("model weights: " + cfg.MODEL.WEIGHTS)
    print("output directory: " + cfg.OUTPUT_DIR)
    #DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS)
    
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(Constants.HOME_DIR.value + "output_segmentation/maskrcnn/model_final.pth")
    res = PolysecureTrainer.test(cfg, model)
    return res


def add_standard_config(cfg):
    cfg.merge_from_file(CONFIG)
    #cfg.OUTPUT_DIR = Constants.HOME_DIR.value + "output_segmentation/maskrcnn/"
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1


def main(args):
    cfg = setup(args)
    trainer = PolysecureTrainer(cfg)
    if args.eval_only:
        cfg.MODEL.WEIGHTS = Constants.HOME_DIR.value + "MaskRCNN_inf.pth"
        #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        print("model weights: " + cfg.MODEL.WEIGHTS)
        do_test(args, cfg)
    else:
        print("Model training->")
        #cfg.MODEL.WEIGHTS = Constants.HOME_DIR.value + "output/quantized_retinanet_apot/model_0007999.pth" # for starting with pretrained weights
        do_train(args, cfg, trainer)


if __name__ == "__main__":
    parser = default_argument_parser()
    #parser.add_argument("--visualize", type=bool, help="Visualize the prediction of the model", default=False)
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        1,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
