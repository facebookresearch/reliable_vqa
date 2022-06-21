# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmf.common.registry import registry


@registry.register_loss("correct_pred")
class CorrectnessPredictionLoss(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.target_type = params["target_type"]
        self.t = params.get("acc_threshold", 0.5)

        assert self.target_type in ["threshold", "max_ind", "regress_bce", "regress_mse", "regress_l1"]

        if self.target_type == "regress_bce":
            self.loss_func = nn.BCELoss(reduction="mean")
        elif self.target_type == "regress_mse":
            self.loss_func = nn.MSELoss()
        elif self.target_type == "regress_l1":
            self.loss_func = nn.L1Loss()
        else:
            self.loss_func = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, 1.0]))

    def _masked_unk_softmax(self, x, dim, mask_idx):
        """
        Copied from VQAAccuracy.
        """
        x1 = F.softmax(x, dim=dim)
        x1[:, mask_idx] = 0
        x1_sum = torch.sum(x1, dim=1, keepdim=True)
        y = x1 / x1_sum
        return y

    def forward(self, sample_list, model_output):
        """
        Compute binary correctness prediction loss.
        Requires:
            scores --> model logits over answers
            targets --> ground truth accuracies for each answer
            confidences --> confidence of correctness prediction (binary from 2-class softmax)
        """
        logits = model_output["scores"]
        # for three branch movie+mcan model
        if logits.dim() == 3:
            logits = logits[:, 0]

        targets = sample_list["targets"]

        normalized_logits = self._masked_unk_softmax(logits, 1, 0)
        pred_inds = torch.argmax(normalized_logits, dim=1)

        if self.target_type == "max_ind":
            tgt_inds = torch.argmax(targets, dim=1)
            correctness = (pred_inds == tgt_inds).to(dtype=torch.long)
        else:
            one_hots = targets.new_zeros(*targets.size())
            one_hots.scatter_(1, pred_inds.view(-1, 1), 1)
            tgt_scores = torch.sum(one_hots * targets, dim=-1)
            if "regress" in self.target_type:
                tgt_scores = tgt_scores.unsqueeze(1)
                correctness = torch.cat([1. - tgt_scores, tgt_scores], dim=-1)
            else:
                correctness = (tgt_scores >= self.t).to(dtype=torch.long)

        confidences = model_output["confidences"]  # normalized confidences, B x 2 if not regression
        
        if self.target_type == "regress_bce":
            return self.loss_func(confidences, correctness) * correctness.size(1)
        else:
            return self.loss_func(confidences, correctness)
