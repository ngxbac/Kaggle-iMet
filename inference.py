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
import glob


checkpoints = [
    [2, 4, 5],
    [1, 2, 6],
    [1, 3, 5],
    [1, 0, 4],
    [1, 0, 4]
]

if __name__ == "__main__":
    for model_name in ["densenet121", "inception_v3", "resnet50", "resnet34", "resnet18", "se_resnet50"]:
        for fold in range(5):
            log_dir = f"/media/ngxbac/DATA/logs_datahack/intel-scene/{model_name}_{fold}"
            with open(f"{log_dir}/config.json") as f:
                config = json.load(f)

            with open("inference.yml") as f:
                infer_config = yaml.load(f)

            model = Finetune(**config['model_params']['params'])

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
                    
            all_checkpoints = glob.glob(f"{log_dir}/checkpoints/stage2.*.pth")

            for i, checkpoint in enumerate(all_checkpoints):
                callbacks = [
                    CheckpointCallback(resume=checkpoint),
                    InferCallback(out_dir=log_dir, out_prefix="/predict_swa_2/predictions." + "{suffix}" + f".{i}.npy")
                ]

                runner = SupervisedRunner()
                runner.infer(
                    model,
                    loaders,
                    callbacks,
                    verbose=True,
                )
