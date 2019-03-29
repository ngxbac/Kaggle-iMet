import torch
import torch.nn as nn
from cnn_finetune import make_model


class Finetune(nn.Module):
    def __init__(
        self,
        arch="se_resnet50",
        n_class=6,
        pretrained=True,
        image_size=256,
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
        # import pdb
        # pdb.set_trace()
        return self.model(x)


def finetune(params):
    return Finetune(**params)