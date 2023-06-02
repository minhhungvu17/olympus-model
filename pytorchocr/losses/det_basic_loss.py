from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BalanceLoss(nn.Module):
    def __init__(
        self,
        balance_loss=True,
        main_loss_type="DiceLoss",
        negative_ratio=3,
        return_origin=False,
        eps=1e-6,
        **kwargs
    ) -> None:
        super(BalanceLoss, self).__init__()
        self.balance_loss = balance_loss
        self.main_loss_type = main_loss_type
        self.negative_ratio = negative_ratio
        self.return_origin = return_origin
        self.eps = eps

        if main_loss_type == "CrossEntropy":
            self.loss = nn.CrossEntropyLoss()
        elif main_loss_type == "Euclidean":
            self.loss = nn.MSELoss()
        elif main_loss_type == "DiceLoss":
            self.loss = DiceLoss(eps=self.eps)
        elif main_loss_type == "BCELoss":
            self.loss = nn.BCELoss(reduction=None)
        elif main_loss_type == "MaskL1Loss":
            self.loss = MaskL1Loss(eps=self.eps)
        else:
            loss_type = [
                "CrossEntropy",
                "DiceLoss",
                "Euclidean",
                "BCELoss",
                "MaskL1Loss",
            ]
            raise Exception(
                "main_loss_type in BalanceLoss() can only be one of {}".format(
                    loss_type
                )
            )

    def forward(self, pred, gt, mask=None):
        positive = gt * mask
        negative = (1 - gt) * mask
        positive_count = int(positive.sum())
        negative_count = int(min(negative.sum(), positive_count * self.negative_ratio))

        loss = self.loss(pred, gt, mask)

        if not self.balance_loss:
            return loss

        positive_loss = positive * loss
        negative_loss = negative * loss
        negative_loss = torch.reshape(negative_loss, shape=[-1])
        if negative_count > 0:
            sort_loss = negative_loss.sort(descending=True)
            negative_loss = sort_loss[:negative_count]
            balance_loss = (positive_loss.sum() + negative_loss.sum()) / (
                positive_count + negative_count + self.eps
            )
        else:
            balance_loss = positive_loss.sum() / (positive_count + self.eps)

        if self.return_origin:
            return balance_loss, loss

        return balance_loss


class BCELoss(nn.Module):
    def __init__(self, reduction="mean") -> None:
        super(BCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, label, mask=None, weight=None, name=None):
        loss = F.binary_cross_entropy(input, label, reduction=self.reduction)
        return loss


class MaskL1Loss(nn.Module):
    def __init__(self, eps=1e-6) -> None:
        super(MaskL1Loss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt, mask):
        loss = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        loss = torch.mean(loss)
        return loss


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6) -> None:
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt, mask, weights=None):
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask

        intersection = torch.sum(pred * gt * mask)
        union = torch.sum(pred * mask) + torch.sum(gt * mask) + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss
