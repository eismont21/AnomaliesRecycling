# AnomaliesRecycling
## Description
The project "Anomaly Detection for the Singulation of Plastic Wastes in 
Polymer Recycling" is done in scope of the practical course Computer Vision 
for Human-Computer Interaction at Karlsruhe Institute of Technology.

The main purpose of the project is counting the number of plastic lids 
in trays. Two main approaches are investigated: image classification and 
instance segmentation. The performance of both ones is boosted by 
synthesized data generated using data augmentation, 
especially copy and paste method.

## Installation
Install dependencies using anaconda.   
```bash
conda env create -f environment.yml
```
The GPU unit is required. 

## Start
To overview the functionality of the project, see [Jupyter Notebooks](notebooks/README.md). 
The project consists of three main modules:
1. [Data Augmentation](data_augmentation/README.md) to synthesize new data
2. [Image Classification](image_classification/README.md) to solve the problem with classification approach
3. [Image Segmentation](image_segmentation/README.md) to solve the problem with instance segmentation approach.

Before start, note that
1. The execution GPU can be defined in ***image_segmentation/train_net.py*** and ***image_classification/transfer_learning.py***
2. For convinience it is recommended to define store and project directory in ***image_classification/constants.py***

