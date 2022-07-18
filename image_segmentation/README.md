# Instance Segmentation 
Start detectron2 training from ***/AnomaliesRecycling/*** 
```bash
python image_segmentation/train_net.py
```

Models are saved into *<Constants.STORE_DIR>/segmentation_experiments/*.

Adjust the network configuration in ***/configs/***.

Change the optimizer and evaluator in ***PolysecureTrainer.py***

Default model: Mask R-CNN with FPN and ResNet50 backbone. 
SGD optimizer. Annealing LR=0.001 in steps (3000, 4000). 
For more info, see [detectron2](https://github.com/facebookresearch/detectron2)