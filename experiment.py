from collections import OrderedDict
import torch
import torch.nn as nn
from catalyst.dl.experiments import ConfigExperiment
from dataset import CsvDataset
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

        image_key = kwargs.get("image_key", 'image')
        label_key = kwargs.get("label_key", 'label')

        train_csv = kwargs.get("train_csv", None)
        valid_csv = kwargs.get("valid_csv", None)
        infer_csv = kwargs.get("infer_csv", None)
        image_size = kwargs.get("image_size", 224)
        test_tta = kwargs.get("use_tta", False)
        root = kwargs.get("root", None)
        root_external = kwargs.get("root_external", None)

        if train_csv:
            trainset = CsvDataset(
                csv_file=train_csv,
                root=root,
                root_external=root_external,
                transform=train_aug(image_size),
                image_key=image_key,
                label_key=label_key,
                mode='train'
            )
            datasets["train"] = trainset

        if valid_csv:
            validset = CsvDataset(
                csv_file=valid_csv,
                root=root,
                root_external=root_external,
                transform=valid_aug(image_size),
                image_key=image_key,
                label_key=label_key,
                mode='train'
            )
            datasets["valid"] = validset

        if infer_csv:
            transforms = infer_tta_aug(image_size)
            if not test_tta:
                inferset = CsvDataset(
                    csv_file=infer_csv,
                    root=root,
                    root_external=root_external,
                    transform=transforms[0],
                    image_key=image_key,
                    label_key=label_key,
                    mode='infer'
                )
                datasets["infer"] = inferset
            else:
                for i, transform in enumerate(transforms):
                    inferset = CsvDataset(
                        csv_file=infer_csv,
                        root=root,
                        root_external=root_external,
                        transform=transform,
                        image_key=image_key,
                        label_key=label_key,
                        mode='infer'
                    )
                    datasets[f"infer_{i}"] = inferset

        return datasets