import numpy as np
import pandas as pd
import cv2
import os
from torch.utils.data import Dataset

NUM_CLASSES = 1103


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def random_sampling(df, label_key, ignore_label=0, ratio=0.75):
    all_labels = df[label_key].unique()
    up_df = []
    for label in all_labels:
        label_df = df[df[label_key] == label]
        if label != ignore_label:
            upsampling_df = label_df.sample(int(len(label_df) * ratio), replace=False)
            label_df = pd.concat([label_df, upsampling_df], axis=0)

        up_df.append(label_df)

    up_df = pd.concat(up_df, axis=0)
    return up_df


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
        if mode == 'train':
            self.labels = df[label_key].values

        self.images = df[image_key].values
        self.transform = transform
        
        self.root = root
        self.root_external = root_external

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = os.path.join(self.root, image + '.png')

        image = load_image(image)

        if self.mode == 'train':
            label = self.labels[idx]
            label = [int(l) for l in label.split(' ')]
            label_arr = np.zeros((NUM_CLASSES, )).astype(np.float32)
            for l in label:
                label_arr[l] = 1
        else:
            label_arr = np.zeros((NUM_CLASSES, )).astype(np.float32)

        if self.transform:
            image = self.transform(image=image)['image']
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "images": image,
            "targets": label_arr
        }
