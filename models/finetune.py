import torch
import torch.nn.functional as F
import torch.nn as nn
from cnn_finetune import make_model
from .cbam_cam import ResidualNet as AttentionResnet
import models.fishnet as fishnet
import models.dla as dla


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

    def freeze_base(self):
        for param in self.model._features.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        for param in self.model._features.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


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


class FinetuneFishNet(nn.Module):
    def __init__(self,
                 arch='fishnet99',
                 pretrained=None,
                 n_class=7
                 ):
        super(FinetuneFishNet, self).__init__()

        self.model = getattr(fishnet, arch)(pretrained=pretrained, n_class=n_class)
        self.model_name = arch
        self.extract_feature = False

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.conv2(x)
        x = self.model.conv3(x)
        x = self.model.pool1(x)
        score, score_feat = self.model.fish(x)
        # 1*1 output
        out = score.view(x.size(0), -1)
        if self.extract_feature:
            score_feat = F.adaptive_avg_pool2d(score_feat, 1)
            score_feat = score_feat.view(x.size(0), -1)
            return score_feat
        else:
            return out

    def freeze_base(self):
        pass

    def unfreeze_base(self):
        pass


class FinetuneDLA(nn.Module):
    def __init__(self,
                 arch='dla34',
                 pretrained='imagenet',
                 n_class=7
                 ):
        super(FinetuneDLA, self).__init__()

        self.model = getattr(dla, arch)(pretrained=pretrained, n_class=n_class)
        self.model_name = arch
        self.extract_feature = False

    def forward(self, x):
        return self.model(x)

    def freeze_base(self):
        for param in self.model.parameters():
            param.requires_grad = False

        # we want to freeze the fc layer
        self.model.fc.weight.requires_grad = True
        self.model.fc.requires_grad = True

    def unfreeze_base(self):
        for param in self.model.parameters():
            param.requires_grad = True


def finetune(params):
    return Finetune(**params)


def finetune_cbam(params):
    return FinetuneCBAM(**params)


def finetune_fishnet(params):
    """
    Finetune fishmodel
    """
    return FinetuneFishNet(**params)


def finetune_dla(params):
    """
    Finetune fishmodel
    """
    return FinetuneDLA(**params)
