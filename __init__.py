# flake8: noqa
from runner import ModelRunner as Runner
from experiment import Experiment
from catalyst.contrib.registry import Registry
from models import *
from callbacks import *
from losses import *

# Register model
Registry.model(finetune)
Registry.model(finetune_cbam)
Registry.model(inception_v3_deepsupervision)
Registry.model(finetune_fishnet)
Registry.model(finetune_dla)

# Register callback
Registry.callback(F1Callback)
Registry.callback(FbetaCallback)
Registry.callback(LossDeepSupervisionCallback)
Registry.callback(FbetaDeepSupervisionCallback)
Registry.callback(FbetaTwoHeadsCallback)
Registry.callback(LossTwoHeadCallback)

# Register loss
Registry.criterion(FocalLoss)
Registry.criterion(FbetaLoss)
Registry.criterion(BCEAndFbeta)
Registry.criterion(BCEFbetaFocalLoss)
Registry.criterion(TwoHeadsLoss)
