# flake8: noqa
from runner import ModelRunner as Runner
from experiment import Experiment
from catalyst.contrib.registry import Registry
from models.finetune import *
from callbacks import F1Callback

# Register model
Registry.model(finetune)
Registry.model(finetune_embedding)

# Register callback
Registry.callback(F1Callback)