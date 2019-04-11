import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()
    
    
class FbetaLoss(nn.Module):
    def __init__(self, beta=2):
        super(FbetaLoss, self).__init__()
        self.small_value = 1e-6
        self.beta = beta

    def forward(self, logits, labels):
        beta = self.beta
        batch_size = logits.size()[0]
        p = F.sigmoid(logits)
        l = labels
        num_pos = torch.sum(p, 1) + self.small_value
        num_pos_hat = torch.sum(l, 1) + self.small_value
        tp = torch.sum(l * p, 1)
        precise = tp / num_pos
        recall = tp / num_pos_hat
        fs = (1 + beta * beta) * precise * recall / (beta * beta * precise + recall + self.small_value)
        loss = fs.sum() / batch_size
        return 1 - loss
    
    
class BCEAndFbeta(nn.Module):
    def __init__(self, bce_weight=0.5):
        super(BCEAndFbeta, self).__init__()
        self.bce_weight = bce_weight
        self.f1_loss = FbetaLoss()
        self.bce_loss = F.binary_cross_entropy_with_logits

    def forward(self, logits, labels):
        f1 = self.f1_loss(logits, labels)
        bce = self.bce_loss(logits, labels)
        return self.bce_weight * bce + (1 - self.bce_weight) * f1


class BCEFbetaFocalLoss(nn.Module):
    def __init__(self):
        super(BCEFbetaFocalLoss, self).__init__()
        self.f1_loss = FbetaLoss()
        self.bce_loss = F.binary_cross_entropy_with_logits
        self.focal_loss = FocalLoss()

    def forward(self, logits, labels):
        f1 = self.f1_loss(logits, labels)
        bce = self.bce_loss(logits, labels)
        focal = self.focal_loss(logits, labels)
        return 0.45 * bce + 0.45 * f1 + 0.1 * focal


class TwoHeadsLoss(nn.Module):
    """
    Loss for two heads
    """
    def __init__(self):
        super(TwoHeadsLoss, self).__init__()
        self.sigmoid_loss = BCEFbetaFocalLoss()
        self.softmax_loss = nn.CrossEntropyLoss()

    def forward(self, logit_sigmoid, logit_softmax, target_sigmoid, target_softmax):
        sigmoid_loss = self.sigmoid_loss(logit_sigmoid, target_sigmoid)
        softmax_loss = self.softmax_loss(logit_softmax, target_softmax)

        return 0.5 * sigmoid_loss + 0.5 * softmax_loss

