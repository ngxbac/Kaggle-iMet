from albumentations import *


def train_aug(image_size=224):
    return Compose([
        Resize(image_size, image_size),
        CLAHE(clip_limit=2),
#         ToGray(),
        RandomBrightnessContrast(),
        Rotate(limit=5),
        HorizontalFlip(),
        Normalize(),
    ], p=1)


def valid_aug(image_size=224):
    return Compose([
        Resize(image_size, image_size),
        Normalize(),
    ], p=1)


def infer_aug(image_size=224):
    return Compose([
        RandomCrop(),
        Resize(image_size, image_size),
        Normalize(),
    ], p=1)


def infer_tta_aug(image_size=224):
    tta_simple = [
        Compose([
            Resize(image_size, image_size),
            Normalize(p=1),
        ], p=1),
        Compose([
            Resize(image_size, image_size),
            HorizontalFlip(p=1.0),
            Normalize(p=1),
        ], p=1),
    ]

    return tta_simple
