import numpy as np
import pandas as pd
import cv2
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from albumentations import *


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


class TwoHeadDataset(Dataset):

    def __init__(self,
                 csv_file,
                 root,
                 transform,
                 mode='train',
                 image_key='file_name',
                 label_key='category_id',
                 ):
        df = pd.read_csv(csv_file, nrows=None)

        self.mode = mode
        if mode == 'train':
            self.labels = df[label_key].values

            # Softmax label
            cls = np.load("./data/class.npy")
            le = LabelEncoder()
            le.classes_ = cls
            self.labels_softmax = le.transform(self.labels)
            np.save("softmax_label.npy", le.classes_)

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
            label_sigmoid = np.zeros(NUM_CLASSES).astype(np.float32)
            for l in label:
                label_sigmoid[l] = 1

            label_softmax = self.labels_softmax[idx]
        else:
            label_sigmoid = np.zeros(NUM_CLASSES).astype(np.float32)
            label_softmax = -1

        if self.transform:
            image = self.transform(image=image)['image']
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "images": image,
            "sigmoid_label": label_sigmoid,
            "softmax_label": label_softmax
        }


from collections import namedtuple
import json
from os.path import exists, join


Dataset = namedtuple('Dataset', ['model_hash', 'classes', 'mean', 'std',
                                 'eigval', 'eigvec', 'name'])

imagenet = Dataset(name='imagenet',
                   classes=1000,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225],
                   eigval=[55.46, 4.794, 1.148],
                   eigvec=[[-0.5675, 0.7192, 0.4009],
                           [-0.5808, -0.0045, -0.8140],
                           [-0.5836, -0.6948, 0.4203]],
                   model_hash={'dla34': 'ba72cf86',
                               'dla46_c': '2bfd52c3',
                               'dla46x_c': 'd761bae7',
                               'dla60x_c': 'b870c45c',
                               'dla60': '24839fc4',
                               'dla60x': 'd15cacda',
                               'dla102': 'd94d9790',
                               'dla102x': 'ad62be81',
                               'dla102x2': '262837b6',
                               'dla169': '0914e092'})


def get_data(data_name):
    try:
        return globals()[data_name]
    except KeyError:
        return None


def load_dataset_info(data_dir, data_name='new_data'):
    info_path = join(data_dir, 'info.json')
    if not exists(info_path):
        return None
    info = json.load(open(info_path, 'r'))
    assert 'mean' in info and 'std' in info, \
        'mean and std are required for a dataset'
    data = Dataset(name=data_name, classes=0,
                   mean=None,
                   std=None,
                   eigval=None,
                   eigvec=None,
                   model_hash=dict())
    return data._replace(**info)
