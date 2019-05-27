import numpy as np
import torch
from typing import Mapping, Any
from catalyst.dl.experiments import Runner


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class ModelRunner(Runner):

    def __init__(self):
        super(ModelRunner, self).__init__()
        self.input_key = "images"
        self.output_key = "logits"
        self.input_target_key = "targets"

    def _run_batch(self, batch):
        self.state.step += self.state.batch_size
        batch = self._batch2device(batch, self.device)

        # Mixup data
        if self.state.loader_name == 'train':
            mixed_x, y_a, y_b, lam = mixup_data(batch['images'], batch['targets'], alpha=0.4)
            batch['images'] = mixed_x
            batch['targets_a'] = y_a
            batch['targets_b'] = y_b
            batch['lam'] = lam

        self.state.input = batch
        self.state.output = self.predict_batch(batch)

    def _run_loader(self, loader):
        self.state.batch_size = loader.batch_size
        self.state.step = (
            self.state.step
            or self.state.epoch * len(loader) * self.state.batch_size
        )
        # @TODO: remove time usage, use it under the hood
        self.state.timer.reset()

        self.state.timer.start("_timers/batch_time")
        self.state.timer.start("_timers/data_time")

        for i, batch in enumerate(loader):
            batch = self._batch2device(batch, self.device)
            self.state.timer.stop("_timers/data_time")

            self._run_event("batch_start")

            self.state.timer.start("_timers/model_time")
            self._run_batch(batch)
            self.state.timer.stop("_timers/model_time")

            self.state.timer.stop("_timers/batch_time")
            self._run_event("batch_end")

            self.state.timer.reset()

            if self._check_run and i >= 3:
                break

            self.state.timer.start("_timers/batch_time")
            self.state.timer.start("_timers/data_time")

    def predict_batch(self, batch: Mapping[str, Any]):
        output = self.model(batch["images"])
        return {
            "logits": output
        }