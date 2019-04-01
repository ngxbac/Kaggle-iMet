import numpy as np
import pandas as pd
import cv2
import os
from torch.utils.data import Dataset


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
            # if 'train' in csv_file:
            #     df = random_sampling(df, label_key=label_key, ratio=0.75)
            #     print("Upsampling {}".format(csv_file))
            self.labels = df[label_key].values

        self.images = df[image_key].values
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
