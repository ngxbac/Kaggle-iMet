from collections import OrderedDict
import torch
import torch.nn as nn
from catalyst.dl.experiments import ConfigExperiment
from dataset import IntelSceneDataset
from augmentation import train_aug, valid_aug, infer_tta_aug


class Experiment(ConfigExperiment):
    def _postprocess_model_for_stage(self, stage: str, model: nn.Module):
        model_ = model
        if isinstance(model, torch.nn.DataParallel):
            model_ = model_.module

        if stage == "stage1":
            print("Stage1")
            model_.freeze_base()
        elif stage == 'stage2':
            print("Stage2")
            model_.unfreeze_base()

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

        train_csv = kwargs.get("train_csv", None)
        valid_csv = kwargs.get("valid_csv", None)
        infer_csv = kwargs.get("infer_csv", None)
        root = kwargs.get("root", None)

        if train_csv:
            trainset = IntelSceneDataset(
                csv_file=train_csv,
                root=root,
                transform=Experiment.get_transforms(stage=stage, mode='train'),
                mode='train'
            )
            datasets["train"] = trainset

        if valid_csv:
            validset = IntelSceneDataset(
                csv_file=valid_csv,
                root=root,
                transform=Experiment.get_transforms(stage=stage, mode='valid'),
                mode='train'
            )
            datasets["valid"] = validset

        if infer_csv:
            inferset = IntelSceneDataset(
                csv_file=infer_csv,
                root=root,
                transform=Experiment.get_transforms(stage=stage, mode='valid'),
                mode='infer'
            )
            datasets["infer"] = inferset

        return datasets