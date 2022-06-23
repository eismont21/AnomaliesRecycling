import os
import pandas as pd
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
import torch

# git fix
class RecyclingDataset(Dataset):
    """
    Custom class for custom dataset
    """
    def __init__(self, csv_file, img_dir, transform=None):
        #if 'train' in csv_file:
        #    df = pd.read_csv(csv_file)
        #    df = df[~df['name'].str.contains("harder")]
        #    self.img_labels = df
        #else:
        #    self.img_labels = pd.read_csv(csv_file)
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.classes = self.img_labels['count'].unique()
        self.classes.sort()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path, ImageReadMode.RGB)
        label = self.img_labels.iloc[idx, 1]
        features = dict(self.img_labels.iloc[idx, 2:])
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label': label, 'img_path': img_path, 'features': features}
        return sample
