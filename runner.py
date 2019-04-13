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
        if 'culture_labels' in batch.keys():
            """
            TwoHead
            """
            culture_logits, tag_logits = self.model(batch["images"])
            output = {
                "culture_logits": culture_logits,
                "tag_logits": tag_logits
            }
        else:
            logits = self.model(batch["images"])
            output = {
                "logits": logits,
            }

        return output
