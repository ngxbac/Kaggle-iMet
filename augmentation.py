from albumentations import *


def train_aug(image_size=224):
    return Compose([
        Resize(image_size, image_size),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
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
        # Compose([
        #     Resize(image_size, image_size),
        #     HorizontalFlip(p=1.0),
        #     Normalize(p=1),
        # ], p=1),
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
