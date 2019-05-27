from collections import OrderedDict
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
from catalyst.dl.experiments import ConfigExperiment
from dataset import *
from augmentation import train_aug, valid_aug, infer_tta_aug


class Experiment(ConfigExperiment):
    def _postprocess_model_for_stage(self, stage: str, model: nn.Module):

        import warnings
        warnings.filterwarnings("ignore")

        random.seed(2411)
        np.random.seed(2411)
        torch.manual_seed(2411)

        model_ = model
        if isinstance(model, torch.nn.DataParallel):
            model_ = model_.module

        if "stage1" in stage:
            print("Stage1")
            model_.freeze_base()
        elif "stage2" in stage:
            print("Stage2")
            model_.unfreeze_base()
        elif "infer" in stage:
            print("Inference stage ... ")
            if hasattr(model, 'is_infer'):
                model.is_infer = True

        return model_

    @staticmethod
    def get_transforms(stage: str = None, mode: str = None):
        if mode == 'train':
            return train_aug()
        elif mode == 'valid':
            return valid_aug()
        elif mode == 'infer':
            return infer_tta_aug()

    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()

        """
        image_key: 'id'
        label_key: 'attribute_ids'
        """

        image_size = kwargs.get("image_size", 320)
        train_csv = kwargs.get('train_csv', None)
        valid_csv = kwargs.get('valid_csv', None)
        root = kwargs.get('root', None)

        if train_csv:
            transform = train_aug(image_size)
            train_set = CsvDataset(
                csv_file=train_csv,
                root=root,
                transform=transform,
                mode='train',
                image_key='id',
                label_key='attribute_ids',
            )
            datasets["train"] = train_set

        if valid_csv:
            transform = valid_aug(image_size)
            valid_set = CsvDataset(
                csv_file=valid_csv,
                root=root,
                transform=transform,
                mode='train',
                image_key='id',
                label_key='attribute_ids',
            )
            datasets["valid"] = valid_set

        return datasets
