# flake8: noqa
from runner import ModelRunner as Runner
from experiment import Experiment
from catalyst.dl import registry
from models import *
from callbacks import *
from losses import *

# Register model
registry.Model(finetune)
registry.Model(finetune_cbam)
registry.Model(inception_v3_deepsupervision)
registry.Model(finetune_fishnet)
registry.Model(finetune_dla)

# Register callback
registry.Callback(F1Callback)
registry.Callback(FbetaCallback)
registry.Callback(LossDeepSupervisionCallback)
registry.Callback(FbetaDeepSupervisionCallback)
registry.Callback(FbetaTwoHeadsCallback)
registry.Callback(LossTwoHeadCallback)

# Register loss
registry.Criterion(FocalLoss)
registry.Criterion(FbetaLoss)
registry.Criterion(BCEAndFbeta)
registry.Criterion(BCEFbetaFocalLoss)
registry.Criterion(TwoHeadsLoss)

