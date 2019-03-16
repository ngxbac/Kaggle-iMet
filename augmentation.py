from albumentations import *


def train_aug(image_size=224):
    return Compose([
        Resize(image_size, image_size),
        Rotate(limit=10),
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
    # for i in range(5):
    #     tta_simple.append(
    #         Compose([
    #             infer_aug_five_crops(i, image_size, p),
    #             Normalize(p=1)
    #         ], p=p)
    #     )
    #
    #     tta_simple.append(
    #         Compose([
    #             HorizontalFlip(p=1.0),
    #             infer_aug_five_crops(i, image_size, p),
    #             Normalize(p=1)
    #         ], p=p)
    #     )

    return tta_simple
