import torch
import torch.nn as nn
from cnn_finetune import make_model
from .cbam_cam import ResidualNet as AttentionResnet


class Finetune(nn.Module):
    def __init__(
        self,
        arch="se_resnet50",
        n_class=6,
        pretrained=True,
        image_size=256,
        **kwargs
    ):
        super(Finetune, self).__init__()
        self.model = make_model(
            model_name=arch,
            num_classes=n_class,
            pretrained=pretrained,
            input_size=(image_size, image_size),
        )

        self.head_sigmoid = nn.Linear(
            self.model._classifier.in_features, n_class
        )

        self.head_softmax = nn.Linear(
            self.model._classifier.in_features, 50238
        )

    def freeze_base(self):
        for param in self.model._features.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        for param in self.model._features.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.model._features(x)
        features = self.model.pool(features)
        features = features.view(features.size(0), -1)
        x_sigmoid = self.head_sigmoid(features)
        x_softmax = self.head_softmax(features)
        return x_sigmoid, x_softmax


class FinetuneCBAM(nn.Module):
    def __init__(
        self,
        arch="se_resnet50",
        n_class=6,
        pretrained=True,
        image_size=256,
        **kwargs
    ):
        super(FinetuneCBAM, self).__init__()
        self.model = AttentionResnet(
            pretrained=pretrained,
            network_type='ImageNet',
            depth=50,
            num_classes=1000,
            att_type='CBAM'
        )

        self.fc = nn.Linear(
            self.model.fc.in_features,
            n_class
        )

    def freeze_base(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model.features(x)
        return self.fc(x)


def finetune(params):
    return Finetune(**params)


def finetune_embedding(params):
    return FinetuneEmbedding(**params)


def finetune_cbam(params):
    return FinetuneCBAM(**params)