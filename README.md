# Anomaly Detection in Polymer Recycling

## Description

The "Anomaly Detection for the Singulation of Plastic Wastes in Polymer Recycling" project is part
of [the practical course Computer Vision for Human-Computer Interaction at Karlsruhe Institute of Technology](https://cvhci.anthropomatik.kit.edu/600_2231.php).
It focuses on counting the number of plastic lids in trays and identifying singulation anomalies, addressing a crucial
need in polymer recycling processes.

## Objective

The project's objective is to develop robust deep learning models that can accurately count and detect anomalies in
plastic lid singulation. This is crucial for improving the sorting and recycling of plastics, a significant
environmental concern.

## Dataset and Data Augmentation

The dataset comprises images of black trays with 0 to 5 plastic lids. Due to class imbalance and limited annotations for
instance segmentation, the team innovated a copy & paste data augmentation technique. This method enhanced the dataset
with synthetic images, using techniques like object rotation, color changes, and transparency, to improve model training
and performance.

## Methodology

Two primary approaches were explored:

1. **Classification**: Using a ResNet18 model trained on both original and synthetic datasets, achieving remarkable
   accuracy in identifying the number of lids.
2. **Instance Segmentation**: Employing Mask R-CNN, although it showed lower performance than classification, it
   provided valuable insights and interpretability.

## Results and Future Directions

The classification approach, especially with synthetic data, showed the best performance. Future work could explore
combining classification and instance segmentation to enhance both accuracy and interpretability.

The qualitative and quantitative results can be found in the [presentation](materials/AnomaliesRecycling_final_pres.pdf)
and [report](materials/AnomaliesRecycling_report.pdf) in the **materials** folder.

## Installation

Install dependencies using Anaconda.

```bash
conda env create -f environment.yml
```

A GPU unit is required.

## Getting Started

For an overview of the project's functionality, see [Jupyter Notebooks](notebooks/README.md). The project includes three
main modules:

1. [Data Augmentation](data_augmentation/README.md) for synthesizing new data.
2. [Image Classification](image_classification/README.md) using the classification approach.
3. [Image Segmentation](image_segmentation/README.md) using instance segmentation.

Before starting, configure the GPU settings in **image_segmentation/train_net.py** and
**image_classification/transfer_learning.py**. It is recommended to define the store and project directory in
**image_classification/constants.py**.