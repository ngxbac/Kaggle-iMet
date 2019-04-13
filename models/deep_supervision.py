import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from catalyst.contrib.modules.common import Flatten


class Inception_v3_DeepSupervision(nn.Module):
    def __init__(
        self,
        arch="se_resnet50",
        n_class=6,
        pretrained=True,
        image_size=256,
        **kwargs
    ):
        super(Inception_v3_DeepSupervision, self).__init__()
        self.model = models.inception_v3(
            pretrained=pretrained
        )

        self.model.aux_logits = False

        self.fc = nn.Sequential(
            nn.Conv2d(2048, 32, kernel_size=1),
            nn.ReLU(),
            Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, n_class)
        )

        self.deepsuper_2 = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            Flatten(),
            nn.BatchNorm1d(288),
            nn.Linear(288, n_class)
        )

        self.deepsuper_4 = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            Flatten(),
            nn.BatchNorm1d(768),
            nn.Linear(768, n_class)
        )

        self.deepsuper_6 = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            Flatten(),
            nn.BatchNorm1d(768),
            nn.Linear(768, n_class)
        )

        self.deepsuper_8 = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            Flatten(),
            nn.BatchNorm1d(1280),
            nn.Linear(1280, n_class)
        )

        self.deepsuper_10 = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            Flatten(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, n_class)
        )

        self.deepsuper_mid = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            Flatten(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, n_class)
        )

        self.is_infer = False

    def freeze_base(self):
        # pass
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        # pass
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        if self.model.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.model.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.model.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.model.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.model.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.model.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.model.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.model.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.model.Mixed_5d(x)

        x_mix_2 = self.deepsuper_2(x)

        # 35 x 35 x 288
        x = self.model.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6b(x)

        x_mix_4 = self.deepsuper_4(x)

        # 17 x 17 x 768
        x = self.model.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6d(x)

        x_mix_6 = self.deepsuper_6(x)

        # 17 x 17 x 768
        x = self.model.Mixed_6e(x)
        # 17 x 17 x 768
        if self.model.training and self.model.aux_logits:
            aux = self.model.AuxLogits(x)
        # 17 x 17 x 768
        x = self.model.Mixed_7a(x)

        x_mix_8 = self.deepsuper_8(x)

        # 8 x 8 x 1280
        x = self.model.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.model.Mixed_7c(x)

        x_mix_10 = self.deepsuper_10(x)

        x_middle = self.deepsuper_mid(x)

        # # 8 x 8 x 2048
        # x = F.avg_pool2d(x, kernel_size=8)
        # # 1 x 1 x 2048
        # x = F.dropout(x, training=self.model.training)
        # # 1 x 1 x 2048
        # x = x.view(x.size(0), -1)
        # # 2048
        x_final = self.fc(x)
        # 1000 (num_classes)
        if self.model.training and self.model.aux_logits:
            return x, aux

        if self.is_infer:
            x_mix_2 = F.sigmoid(x_mix_2)
            x_mix_4 = F.sigmoid(x_mix_4)
            x_mix_6 = F.sigmoid(x_mix_6)
            x_mix_8 = F.sigmoid(x_mix_8)
            x_mix_10 = F.sigmoid(x_mix_10)
            x_middle = F.sigmoid(x_middle)
            x_final = F.sigmoid(x_final)

            return (x_mix_2 + x_mix_4 + x_mix_6 + x_mix_8 + x_mix_10 + x_final + x_middle) / 7

        return x_mix_2, x_mix_4, x_mix_6, x_mix_8, x_mix_10, x_middle, x_final


def inception_v3_deepsupervision(params):
    return Inception_v3_DeepSupervision(**params)