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
        print(csv_file)
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
                 label_file,
                 root,
                 transform,
                 mode='train',
                 image_key='file_name',
                 label_key='category_id',
                 ):
        df = pd.read_csv(csv_file, nrows=None)
        label_df = pd.read_csv(label_file)

        label_df['att'] = label_df['attribute_name'].apply(lambda x: 'culture' if 'culture' in x else 'tag')
        attribute_names = label_df['attribute_name'].values

        self.unique_cultures = label_df[label_df['att'] == 'culture']['attribute_name'].unique()
        self.unique_tag = label_df[label_df['att'] == 'tag']['attribute_name'].unique()

        self.le_culture = LabelEncoder()
        self.le_culture.fit(self.unique_cultures)
        np.save("./data/culture_class.npy", self.le_culture.classes_)
        self.n_culture_cls = len(self.le_culture.classes_)

        self.le_tag = LabelEncoder()
        self.le_tag.fit(self.unique_tag)
        np.save("./data/tag_class.npy", self.le_tag.classes_)
        self.n_tag_cls = len(self.le_tag.classes_)

        self.mode = mode
        if mode == 'train':
            self.labels = df[label_key].values

            self.culture_labels = []
            self.tag_labels = []
            for label in self.labels:
                label = [int(l) for l in label.split(' ')]
                culture_label = []
                tag_label = []

                for l in label:
                    attribute_name = attribute_names[l]
                    if 'culture' in attribute_name:
                        culture_label.append(attribute_name)
                    elif 'tag' in attribute_name:
                        tag_label.append(attribute_name)
                    else:
                        raise ValueError

                self.culture_labels.append(culture_label)
                self.tag_labels.append(tag_label)

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
            culture_label = self.culture_labels[idx]
            tag_label = self.tag_labels[idx]

            culture_label_sigmoid = np.zeros(self.n_culture_cls).astype(np.float32)
            for label in culture_label:
                label = self.le_culture.transform([label])[0]
                culture_label_sigmoid[label] = 1

            tag_label_sigmoid = np.zeros(self.n_tag_cls).astype(np.float32)
            for label in tag_label:
                label = self.le_tag.transform([label])[0]
                tag_label_sigmoid[label] = 1
        else:
            culture_label_sigmoid = np.zeros(self.n_culture_cls).astype(np.float32)
            tag_label_sigmoid = np.zeros(self.n_tag_cls).astype(np.float32)

        if self.transform:
            image = self.transform(image=image)['image']
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "images": image,
            "culture_labels": culture_label_sigmoid,
            "tag_labels": tag_label_sigmoid
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
