import numpy as np
import os
import cv2
import pandas as pd
from torch.utils.data import Dataset

NUM_CLASSES = 1103


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class CsvDataset(Dataset):

    def __init__(self,
                 csv_file,
                 root,
                 transform,
                 mode='train',
                 image_key='file_name',
                 label_key='category_id',
                 ):
        print("\n" + csv_file)
        df = pd.read_csv(csv_file, nrows=None)
        print(df.head())

        self.mode = mode
        if mode == 'train':
            self.labels = df[label_key].values

        self.images = df[image_key].values
        self.transform = transform

        self.root = root

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = os.path.join(self.root, image + '.png')

        image = load_image(image)

        if self.mode == 'train':
            label = self.labels[idx]
            label = [int(l) for l in label.split(' ')]
            label_arr = np.zeros(NUM_CLASSES).astype(np.float32)
            for l in label:
                label_arr[l] = 1
        else:
            label_arr = np.zeros(NUM_CLASSES).astype(np.float32)

        if self.transform:
            image = self.transform(image=image)['image']
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "images": image,
            "targets": label_arr
        }