
# flake8: noqa
from catalyst.dl import registry

from .experiment import Experiment
from .runner import ModelRunner as Runner
from .callbacks import *
from .models import *
from .losses import *

registry.Model(Finetune)

registry.Callback(F1Callback)
registry.Callback(FbetaCallback)
registry.Callback(MixupLossCallback)
registry.Callback(IterCheckpointCallback)


# Register loss
registry.Criterion(FocalLoss)
registry.Criterion(FbetaLoss)
registry.Criterion(BCEAndFbeta)
registry.Criterion(BCEFbetaFocalLoss)