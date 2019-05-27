import torch.nn as nn

from models.senet import *
from models.inceptionresnetv2 import *
from models.xception import *
from models.densenet import *


def get_senet_features(original_model):
    return nn.Sequential(
        original_model.layer0,
        original_model.layer1,
        original_model.layer2,
        original_model.layer3,
        original_model.layer4,
    )


def get_inceptionresnetv2_features(original_model):
    return nn.Sequential(*list(original_model.children())[:-2])


def get_xception_features(original_model):
    return nn.Sequential(*list(original_model.children())[:-1])


def get_nasnet5large_features(original_model):
    features = nn.Module()
    for name, module in list(original_model.named_children())[:-3]:
        features.add_module(name, module)
    return features


def get_densenet_features(original_model):
    return nn.Sequential(*original_model.features, nn.ReLU(inplace=True))


class Finetune(nn.Module):
    def __init__(
        self,
        arch="se_resnet50",
        n_class=6,
        **kwargs
    ):
        super(Finetune, self).__init__()
        self.arch = arch
        if arch == 'se_resnext50_32x4d':
            self.model = se_resnext50_32x4d(num_classes=1000, pretrained='imagenet')
            self.model._features = get_senet_features(self.model)
            self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
            in_features = self.model.last_linear.in_features
            self.model.last_linear = nn.Linear(
                in_features=in_features, out_features=n_class
            )
        elif arch == 'inceptionresnetv2':
            self.model = inceptionresnetv2(num_classes=1000, pretrained='imagenet')
            self.model._features = get_inceptionresnetv2_features(self.model)
            self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
            in_features = self.model.last_linear.in_features
            self.model.last_linear = nn.Linear(
                in_features=in_features, out_features=n_class
            )
        elif arch == 'xception':
            self.model = xception(num_classes=1000, pretrained='imagenet')
            self.model._features = get_xception_features(self.model)
            self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
            in_features = self.model.last_linear.in_features
            self.model.last_linear = nn.Linear(
                in_features=in_features, out_features=n_class
            )
        elif arch == 'densenet121':
            self.model = densenet121(num_classes=1000, pretrained=True)
            self.model._features = get_densenet_features(self.model)
            self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(
                in_features=in_features, out_features=n_class
            )
        else:
            raise ValueError("No model founded !!!")

    def freeze_base(self):
        for param in self.model._features.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        for param in self.model._features.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
