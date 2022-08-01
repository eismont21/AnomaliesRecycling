# Image Classification

Original dataset images should be saved into *<Constants.DATA_DIR>/*.

If you want to use synthetic data for training, you have to generate it first using ***data_augmentation*** module.

Models are saved into *<Constants.STORE_DIR>/classification_experiments/*.

The basic example of training a model is shown in the notebook *notebooks/Classification.ipynb*.

### Default Settings:

Model is ResNet18. Batch sampler is stratified. Cross entropy loss uses weights.