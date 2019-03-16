import numpy as np
import pandas as pd
import cv2
import os
from torch.utils.data import Dataset

def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class IntelSceneDataset(Dataset):

    def __init__(self, csv_file, root, transform, mode='train'):
        df = pd.read_csv(csv_file)
        self.images = df['image_name'].values
        self.mode = mode
        if mode == 'train':
            self.labels = df['label'].values

        self.transform = transform

        self.root = root

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
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
            "features": image,
            "targets": label
        }
