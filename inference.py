#!/usr/bin/env python

import torch
from collections import OrderedDict  # noqa F401
from torch.utils.data import DataLoader
from catalyst.dl.experiments import SupervisedRunner
from catalyst.dl.callbacks import InferCallback, CheckpointCallback
import json
import yaml
from model import Finetune
from experiment import Experiment
from dataset import IntelSceneDataset


if __name__ == "__main__":
    for fold in range(5):
        log_dir = f"/media/ngxbac/DATA/logs_datahack/intel-scene/fold_{fold}"
        with open(f"{log_dir}/config.json") as f:
            config = json.load(f)

        with open("inference.yml") as f:
            infer_config = yaml.load(f)

        model = Finetune()

        infer_csv = infer_config['data_params']['infer_csv']
        root = infer_config['data_params']['root']

        loaders = OrderedDict()
        if infer_csv:
            transforms = Experiment.get_transforms(stage='infer', mode='infer')
            for i, transform in enumerate(transforms):
                inferset = IntelSceneDataset(
                    csv_file=infer_csv,
                    root=root,
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

        callbacks = [
            CheckpointCallback(resume=f"{log_dir}/checkpoints/best.pth"),
            InferCallback(out_dir=log_dir, out_prefix="/predict_seresnet50_2tta/dataset.predictions.{suffix}infer.npy")
        ]

        runner = SupervisedRunner()
        runner.infer(
            model,
            loaders,
            callbacks,
            verbose=True,
        )
