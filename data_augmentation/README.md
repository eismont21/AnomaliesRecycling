# Copy & Paste Data Augmentation

Original dataset images should be saved into *<Constants.DATA_DIR>/*.

Synthetic images will be saved into *<Constants.DATA_DIR>/synthesized/*.

You will need a csv file listing all background images (see */data/zero_lid.csv* for reference).

You will need a csv file listing all images with exactly one object (see */data/one_lid.csv* for reference). 
Please note that our method does not work with very dark objects/low contrast between background and object.

Start Copy & Paste Data Augmentation by creating DataAugmentation object, with the paths to the two csv files, in ***/data_augmentation/***.
    
    from data_augmentation.data_aufmentation import DataAugmentation
    data_augmentation_object = DataAugmentaion(zero_lid_dir, one_lid_dir)

Then run *generate* method on this DataAugmentation object. You can switch on different tags:

  - ***noise_background:*** add noiso to the images with label 0 (switched on by default)
  - ***rotate:*** rotate some inserted objects with random angle (switched on by default)
  - ***change_color:*** change color of some inserted objects
  - ***make_edge:*** allows creating object in the edges
  - ***make_dark:*** make some inserted objects dark
  - ***make_transparent:*** make some inserted objects transparent
  - ***coco_annotation:*** create a coco annotation json file with annotations for instance segmentation

For example the following code generates a new synthetic data set with 6 classes, 10 images in each class and all tags switched on:
````
dataAugm.generate(
    classes={0: 10, 1: 10, 2: 10, 3: 10, 4: 10, 5: 10}, 
    noise_background=True, 
    rotate=True, 
    change_color=True, 
    make_edge=True, 
    make_dark=True, 
    make_transparent=True, 
    coco_annotation=True,  
    data_dir_name='train'
    )
````