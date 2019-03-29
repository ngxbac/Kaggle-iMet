from typing import Mapping, Any
from catalyst.dl.experiments import SupervisedRunner


class ModelRunner(SupervisedRunner):

    def __init__(self):
        super(ModelRunner, self).__init__()
        self.input_key = "images"
        self.output_key = "logits"
        self.input_target_key = "targets"
