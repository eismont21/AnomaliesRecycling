import os
import pandas as pd
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
import torch
from image_classification.constants import Constants


class RecyclingDataset(Dataset):
    """
    Custom class for dataset.
    """

    def __init__(self, csv_file, img_dir, transform=None, sos="", synthetic=False):
        """
        Initialize the dataset.
        :param csv_file: path to csv file with images paths and labels
        :param img_dir: path to directory with images
        :param transform: transform to apply to images
        :param sos: indicator of using SOS dataset
        :param synthetic: indicator of using synthetic dataset
        """
        df = pd.read_csv(csv_file)
        if "test" in csv_file and sos != "":
            df["count"] = df["count"].apply(lambda x: 4 if x == 5 else x)
        if "train" in csv_file:
            if synthetic:
                df_synthesized = pd.read_csv(Constants.SYNTHESIZE_DIR.value + '/synthesized_train.csv')
                df_synthesized['name'] = df_synthesized['name'].apply(lambda x: x[x.find('synthesized'):])
                df = df[~df['name'].str.contains('synthesized')]
                df = pd.concat([df, df_synthesized], ignore_index=True).fillna(0)
                df['synthesized'] = df['synthesized'].astype(int)
            else:
                df = df[~df["name"].str.contains("synthesized")]
        self.img_labels = df
        self.img_dir = img_dir
        self.transform = transform
        self.classes = self.img_labels["count"].unique()
        self.classes.sort()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Get item from dataset.
        :param idx: index of item
        :return: image, label, path and features
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path, ImageReadMode.RGB)
        label = self.img_labels.iloc[idx, 1]
        features = dict(self.img_labels.iloc[idx, 2:])
        if self.transform:
            image = self.transform(image)
        sample = {
            "image": image,
            "label": label,
            "img_path": img_path,
            "features": features,
        }
        return sample
