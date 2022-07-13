"""
This script is based on the training script in detectron2/tools.
This starts the detectron2 model training.
"""
ROOT_DIR = "/home/p22g5/AnomaliesRecycling/"
STORE_DIR = "/cvhci/temp/p22g5/"
IM_ROOT_DIR = "/cvhci/temp/p22g5/data/synthesized/"
import torch
from datetime import datetime
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # specify which GPU(s) to be used

from detectron2.engine import launch, default_argument_parser
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, inference_on_dataset
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog
from image_segmentation.PolysecureTrainer import PolysecureTrainer

from detectron2.checkpoint import DetectionCheckpointer

DATASET_TRAIN_NAME = "polysecure_dataset_train"
DATASET_TEST_NAME = "polysecure_dataset_test"
RETINANET_CONFIG = ROOT_DIR + "image_segmentation/configs/mask_rcnn_R_50_FPN_3x.yaml"

TRAIN_ANNO = IM_ROOT_DIR + "coco_train.json"
TEST_ANNO = IM_ROOT_DIR + "coco_test.json"

TRAIN_IMAGES = IM_ROOT_DIR + "train"
TEST_IMAGES = IM_ROOT_DIR + "test"


def setup(args=None):
    DatasetCatalog.clear()
    cfg = get_cfg()
    register_coco_instances(DATASET_TRAIN_NAME, {}, TRAIN_ANNO, TRAIN_IMAGES)
    register_coco_instances(DATASET_TEST_NAME, {}, TEST_ANNO, TEST_IMAGES)
    add_standard_config(cfg)
    time = datetime.now()
    folder_name = time.strftime('%Y-%m-%d_%H-%M-%S/')
    cfg.OUTPUT_DIR = os.path.join(STORE_DIR, "segmentation_experiments", folder_name)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def do_train(args, cfg, trainer):
    trainer.resume_or_load(resume=False)
    #time = datetime.now()
    #folder_name = time.strftime('%Y-%m-%d_%H-%M-%S/')
    #cfg.OUTPUT_DIR = os.path.join(STORE_DIR, "segmentation_experiments", folder_name)
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
    
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(ROOT_DIR + "output_segmentation/maskrcnn/model_final.pth")
    res = PolysecureTrainer.test(cfg, model)
    return res


def add_standard_config(cfg):
    cfg.merge_from_file(RETINANET_CONFIG)
    #cfg.OUTPUT_DIR = ROOT_DIR + "output_segmentation/maskrcnn/"
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1


def main(args):
    cfg = setup(args)
    trainer = PolysecureTrainer(cfg)
    if args.eval_only:
        cfg.MODEL.WEIGHTS = ROOT_DIR + "MaskRCNN_inf.pth"
        #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        print("model weights: " + cfg.MODEL.WEIGHTS)
        do_test(args, cfg)
    else:
        print("Model training->")
        #cfg.MODEL.WEIGHTS = ROOT_DIR + "output/quantized_retinanet_apot/model_0007999.pth" # for starting with pretrained weights
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
