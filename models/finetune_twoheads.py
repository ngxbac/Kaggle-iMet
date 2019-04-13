import torch
import torch.nn.functional as F
import torch.nn as nn
from cnn_finetune import make_model
from .cbam_cam import ResidualNet as AttentionResnet
import models.fishnet as fishnet
import models.dla as dla


class Finetune2Heads(nn.Module):
    def __init__(
        self,
        arch="se_resnet50",
        n_culture=398,
        n_tag=705,
        pretrained=True,
        image_size=256,
        **kwargs
    ):
        super(Finetune2Heads, self).__init__()
        self.model = make_model(
            model_name=arch,
            num_classes=1000,
            pretrained=pretrained,
            input_size=(image_size, image_size),
        )

        in_features = self.model._classifier.in_features
        self.head_culture = nn.Linear(
            in_features=in_features, out_features=n_culture
        )

        self.head_tag = nn.Linear(
            in_features=in_features, out_features=n_tag
        )

    def freeze_base(self):
        for param in self.model._features.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        for param in self.model._features.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model._features(x)
        x = self.model.pool(x)
        x = x.view(x.size(0), -1)

        culture_logits = self.head_culture(x)
        tag_logits = self.head_tag(x)
        return culture_logits, tag_logits


def finetune_2heads(params):
    return Finetune2Heads(**params)


# def finetune_cbam(params):
#     return FinetuneCBAM(**params)
#
#
# def finetune_fishnet(params):
#     """
#     Finetune fishmodel
#     """
#     return FinetuneFishNet(**params)
#
#
# def finetune_dla(params):
#     """
#     Finetune fishmodel
#     """
#     return FinetuneDLA(**params)
