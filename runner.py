from typing import Mapping, Any
from catalyst.dl.experiments import SupervisedRunner
from models import *


class ModelRunner(SupervisedRunner):

    def __init__(self):
        super(ModelRunner, self).__init__()
        self.input_key = "images"
        self.output_key = "logits"
        self.input_target_key = "targets"

    def predict_batch(self, batch: Mapping[str, Any]):
        if 'softmax_label' in batch.keys():
            sigmoid_logits, softmax_logits = self.model(batch["images"])
            output = {
                "sigmoid_logits": sigmoid_logits,
                "softmax_logits": softmax_logits
            }
        else:
            logits = self.model(batch["images"])
            output = {
                "logits": logits,
            }

        return output
