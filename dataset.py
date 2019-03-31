import numpy as np
import pandas as pd
import cv2
import os
from torch.utils.data import Dataset


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class CsvDataset(Dataset):

    def __init__(self,
                 csv_file,
                 root,
                 root_external,
                 transform,
                 mode='train',
                 image_key='file_name',
                 label_key='category_id',
                 ):
        df = pd.read_csv(csv_file, nrows=None)
        
        self.mode = mode
        self.images = df[image_key].values
        if mode == 'train':
            self.labels = df[label_key].values

        self.transform = transform
        
        if "is_inat" in df.columns:
            self.is_externals = df["is_inat"].values
        else:
            self.is_externals = []

        self.root = root
        self.root_external = root_external

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        
        if len(self.is_externals):
            is_external = self.is_externals[idx]
            if is_external:
                image = os.path.join(self.root_external, image)
            else:
                image = os.path.join(self.root, image)
        else:
            image = os.path.join(self.root, image)

        image = load_image(image)

        if self.mode == 'train':
            label = self.labels[idx]
        else:
            label = 0

        if self.transform:
            image = self.transform(image=image)['image']
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "images": image,
            "targets": label
        }
