from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from .det_basic_loss import BalanceLoss, MaskL1Loss, DiceLoss


class DBLoss(nn.Module):
    def __init__(
        self,
        balance_loss=True,
        main_loss_type="DiceLoss",
        alpha=5,
        beta=10,
        ohem_ratio=3,
        eps=1e-6,
        **kwargs
    ) -> None:
        super(DBLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss(eps=eps)
        self.bce_loss = BalanceLoss(
            balance_loss=balance_loss,
            main_loss_type=main_loss_type,
            negative_ratio=ohem_ratio,
        )

    def forward(self, predicts, labels):
        predict_maps = predicts["maps"]
        (
            label_threshold_map,
            label_threshold_mask,
            label_shrink_map,
            label_shrink_mask,
        ) = labels[1:]
        shrink_maps = predict_maps[:, 0, :, :]
        threshold_maps = predict_maps[:, 1, :, :]
        binary_maps = predict_maps[:, 2, :, :]

        loss_shrink_maps = self.bce_loss(
            shrink_maps, label_shrink_map, label_shrink_mask
        )
        loss_threshold_maps = self.l1_loss(
            threshold_maps, label_threshold_map, label_threshold_mask
        )
        loss_binary_maps = self.dice_loss(
            binary_maps, label_shrink_map, label_shrink_mask
        )

        loss_shrink_maps = self.alpha * loss_shrink_maps
        loss_threshold_maps = self.beta * loss_threshold_maps

        loss_all = loss_shrink_maps + loss_threshold_maps + loss_binary_maps

        losses = {
            "loss": loss_all,
            "loss_shrink_maps": loss_shrink_maps,
            "loss_threshold_maps": loss_threshold_maps,
            "loss_binary_maps": loss_binary_maps,
        }
        return losses
