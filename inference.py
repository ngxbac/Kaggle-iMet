#!/usr/bin/env python

from collections import OrderedDict  # noqa F401
from torch.utils.data import DataLoader
# from catalyst.dl.experiments import SupervisedRunner
from runner import ModelRunner
from catalyst.dl.callbacks import InferCallback, CheckpointCallback
import json
import yaml
import models
from augmentation import infer_tta_aug
from dataset import CsvDataset
import glob


if __name__ == "__main__":
    for model_name in ["resnet34"]:
        for fold in [0]: #[0, 1, 2, 3, 4]
            log_dir = f"/media/ngxbac/DATA/logs_iwildcam/{model_name}/fold_{fold}/"
            with open(f"{log_dir}/config.json") as f:
                config = json.load(f)

            with open("./configs/inference.yml") as f:
                infer_config = yaml.load(f)

            model_function = getattr(models, config['model_params']['model'])
            model = model_function(config['model_params']['params'])

            infer_csv = infer_config['stages']['infer']['data_params']['infer_csv']
            root = infer_config['stages']['infer']['data_params']['root']
            image_size = infer_config['stages']['infer']['data_params']['image_size']
            image_key = infer_config['stages']['infer']['data_params']['image_key']
            label_key = infer_config['stages']['infer']['data_params']['label_key']
            use_tta = infer_config['stages']['infer']['data_params']['use_tta']

            loaders = OrderedDict()
            if infer_csv:
                transforms = infer_tta_aug(image_size)
                if use_tta:
                    for i, transform in enumerate(transforms):
                        inferset = CsvDataset(
                            csv_file=infer_csv,
                            root=root,
                            image_key=image_key,
                            label_key=label_key,
                            transform=transform,
                            mode='infer'
                        )

                        infer_loader = DataLoader(
                            dataset=inferset,
                            num_workers=4,
                            shuffle=False,
                            batch_size=32
                        )

                        loaders[f'infer_{i}'] = infer_loader

                else:
                    inferset = CsvDataset(
                        csv_file=infer_csv,
                        root=root,
                        image_key=image_key,
                        label_key=label_key,
                        transform=transforms[0],
                        mode='infer'
                    )

                    infer_loader = DataLoader(
                        dataset=inferset,
                        num_workers=4,
                        shuffle=False,
                        batch_size=32
                    )

                    loaders[f'infer'] = infer_loader

            all_checkpoints = glob.glob(f"{log_dir}/checkpoints/stage2.*.pth")

            for i, checkpoint in enumerate(all_checkpoints):
                callbacks = [
                    CheckpointCallback(resume=checkpoint),
                    InferCallback(out_dir=log_dir, out_prefix="/predict_swa/predictions." + "{suffix}" + f".{i}.npy")
                ]

                runner = ModelRunner()
                runner.infer(
                    model,
                    loaders,
                    callbacks,
                    verbose=True,
                )
