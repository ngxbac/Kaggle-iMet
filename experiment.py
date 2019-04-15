from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from catalyst.dl.experiments import ConfigExperiment
import dataset as Dataset
from augmentation import train_aug, valid_aug, infer_tta_aug


class Experiment(ConfigExperiment):
    def _postprocess_model_for_stage(self, stage: str, model: nn.Module):

        import warnings
        warnings.filterwarnings("ignore")

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

        dataset = kwargs.get('dataset', None)
        assert dataset is not None

        dataset_name = dataset.get("dataset_name", None)
        assert dataset_name is not None
        dataset_func = getattr(Dataset, dataset_name)

        image_size = kwargs.get("image_size", 224)

        train = dataset.get('train', None)
        valid = dataset.get('valid', None)
        infer = dataset.get('infer', None)

        if train:
            transform = train_aug(image_size)
            train.update({
                "transform": transform
            })
            train_set = dataset_func(**train)
            datasets["train"] = train_set

        if valid:
            transform = valid_aug(image_size)
            valid.update({
                "transform": transform
            })
            valid_set = dataset_func(**valid)
            datasets["valid"] = valid_set

        if infer:
            transform = valid_aug(image_size)
            infer.update({
                "transform": transform
            })
            infer_set = dataset_func(**infer)
            datasets["infer"] = infer_set

        return datasets
