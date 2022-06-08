import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset


class RecyclingDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.classes = self.img_labels['count'].unique()
        self.classes.sort()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        features = dict(self.img_labels.iloc[idx, 2:])
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label': label, 'features': features}
        return sample