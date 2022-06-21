# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
from torch import Tensor

from mmf.common.registry import registry
from mmf.models.vilbert import ViLBERTForClassification, ViLBERT

from reliable_vqa.models.selective_predictors import SelectivePredictor


class CustomViLBERTForClassification(ViLBERTForClassification):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: Tensor,
        image_feature: Tensor,
        image_location: Tensor,
        token_type_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        image_attention_mask: Optional[Tensor] = None,
        masked_lm_labels: Optional[Tensor] = None,
        image_label: Optional[Tensor] = None,
        image_target: Optional[Tensor] = None,
        next_sentence_label: Optional[Tensor] = None,
        output_all_attention_masks: bool = False,
    ) -> Dict[str, Tensor]:

        (
            sequence_output_t,
            sequence_output_v,
            pooled_output_t,
            pooled_output_v,
            attention_weights,
            _encoded_layers_t_output,
            _encoded_layers_v_output,
        ) = self.bert(
            input_ids,
            image_feature,
            image_location,
            token_type_ids,
            attention_mask,
            image_attention_mask,
            output_all_encoded_layers=False,
            output_all_attention_masks=output_all_attention_masks,
        )

        output = {}

        if not torch.jit.is_scripting() and output_all_attention_masks:
            output["attention_weights"] = attention_weights

        if self.fusion_method == "sum":
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.fusion_method == "mul":
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        else:
            raise AssertionError

        if self.training_head_type == "nlvr2":
            pooled_output = pooled_output.view(-1, pooled_output.size(1) * 2)

        output["sequence_output_t"] = sequence_output_t
        output["sequence_output_v"] = sequence_output_v
        output["pooled_output"] = pooled_output

        logits = self.classifier(pooled_output)
        reshaped_logits = logits.contiguous().view(-1, self.num_labels)
        output["scores"] = reshaped_logits

        return output


@registry.register_model("select_vilbert")
class SelectViLBERT(ViLBERT):
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def config_path(cls):
        return "configs/experiments/vilbert/vqa2/select_pred.yaml"

    def build(self):
        self.model = CustomViLBERTForClassification(self.config)

        if self.config.get("freeze_base", False):
            for p in self.model.bert.parameters():
                p.requires_grad = False

        self._init_selector()

        if self.config.get("freeze_vqa", False):
            for p in self.model.classifier.parameters():
                p.requires_grad = False

    def _init_selector(self):
        config_attr = "selector"
        select_type = self.config[config_attr].type
        feat_size = self.config["bi_hidden_size"]
        num_choices = self.config.num_labels
        self.selector = SelectivePredictor(
            select_type,
            feat_size=feat_size,
            num_answers=num_choices,
            **self.config[config_attr].params
        )

    def get_optimizer_parameters(self, config):
        if not self.config.get("freeze_vqa", False):
            params = super().get_optimizer_parameters(config)
            params.append(
                {"params": self.selector.parameters()}
            )
        else:
            params = [{"params": self.selector.parameters()}]

        if hasattr(self.config.selector, "sel_lr"):
            params[-1]["lr"] = self.config.selector.sel_lr

        return params

    def forward(self, sample_list):
        params = self.get_image_and_text_features(sample_list)
        # pretraining labels
        params["masked_lm_labels"] = getattr(sample_list, "lm_label_ids", None)
        # is_random_next = getattr(sample_list, "is_correct", None)
        # TODO(aps): Fix on dataset side
        # params["is_random_next"] = None

        # Prepare Mask
        if params["image_feature"] is not None and params["image_dim"] is not None:
            image_mask = torch.arange(
                params["image_feature"].size(-2), device=params["image_feature"].device
            ).expand(*params["image_feature"].size()[:-1])
            if len(params["image_dim"].size()) < len(image_mask.size()):
                params["image_dim"] = params["image_dim"].unsqueeze(-1)
                assert len(params["image_dim"].size()) == len(image_mask.size())
            image_mask = image_mask < params["image_dim"]
            params["image_attention_mask"] = image_mask.long()
        else:
            params["image_attention_mask"] = None
        params.pop("image_dim")

        output_dict = self.model(
            params["input_ids"],
            params["image_feature"],
            params["image_location"],
            params["token_type_ids"],
            params["attention_mask"],
            params["image_attention_mask"],
            params["masked_lm_labels"],
            params["image_label"],
            params["image_target"],
        )

        if self.config.training_head_type == "pretraining":
            loss_key = "{}/{}".format(
                sample_list.dataset_name, sample_list.dataset_type
            )
            output_dict["losses"] = {}
            output_dict["losses"][loss_key + "/masked_lm_loss"] = output_dict.pop(
                "masked_lm_loss"
            )
            output_dict["losses"][loss_key + "/masked_img_loss"] = output_dict.pop(
                "masked_img_loss"
            )
            # if params["is_random_next"] is not None:
            #     output_dict["losses"][loss_key + "/next_sentence_loss"]
            #       = output_dict.pop("next_sentence_loss")

        text_embedding_total = output_dict.pop("sequence_output_t")
        image_embedding_total = output_dict.pop("sequence_output_v")
        pooled_output = output_dict.pop("pooled_output")
        
        selector_output = self.selector(
            output_dict["scores"].detach(),
            image_embedding_total.detach(),
            text_embedding_total.detach(),
            pooled_output.detach(),
        )

        output_dict.update(selector_output)

        return output_dict
