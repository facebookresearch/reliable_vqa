# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from mmf.common.registry import registry
from mmf.utils.torchscript import getattr_torchscriptable
from mmf.models.visual_bert import VisualBERT

from reliable_vqa.models.selective_predictors import SelectivePredictor


@registry.register_model("select_visual_bert")
class SelectVisualBERT(VisualBERT):
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def config_path(cls):
        return "configs/experiments/visual_bert/vqa2/select_pred.yaml"

    def build(self):
        super().build()
        self._init_selector()

        # See https://github.com/g-luo/news_clippings_training/blob/master/models/utils.py#L37
        # See https://github.com/facebookresearch/mmf/blob/b0b2af7ff21eac648598ebd94d09628127ffc4b5/mmf/models/visual_bert.py#L417
        # Freezing the "trunk" (i.e., multimodal encoder) weights is already handled.
        if self.config.get("freeze_vqa", False):
            for p in self.model.classifier.parameters():
                p.requires_grad = False

    def _init_selector(self):
        config_attr = "selector"
        select_type = self.config[config_attr].type
        feat_size = self.config["hidden_size"]
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
        if torch.jit.is_scripting():
            assert (
                "image_feature_0" in sample_list
            ), "Key 'image_feature_0' is required in TorchScript model"

        # input_mask is B x T, where T is text length and entries are {0, 1}
        # image_mask is B x (num_choices x) x I, where I is number of image
        #      regions, num_choices is for NLVR or other tasks requiring more
        #      than one image as input. Entries in {0, 1}.

        sample_list = self.update_sample_list_based_on_head(sample_list)
        sample_list = self.add_custom_params(sample_list)
        sample_list = self.flatten_for_bert(sample_list)
        sample_list = self.add_post_flatten_params(sample_list)

        output_dict = self.model(
            sample_list["input_ids"],
            sample_list["input_mask"],
            sample_list["attention_mask"],
            sample_list["token_type_ids"],
            sample_list["visual_embeddings"],
            sample_list["visual_embeddings_type"],
            getattr_torchscriptable(sample_list, "image_text_alignment", None),
            getattr_torchscriptable(sample_list, "masked_lm_labels", None),
        )

        if self.training_head_type == "pretraining":
            if not torch.jit.is_scripting():
                loss_key = "{}/{}".format(
                    sample_list["dataset_name"], sample_list["dataset_type"]
                )
                output_dict["losses"] = {}
                output_dict["losses"][loss_key + "/masked_lm_loss"] = output_dict.pop(
                    "masked_lm_loss"
                )
            else:
                raise RuntimeError("Pretraining head can't be used in script mode.")

        sequence_output = output_dict["sequence_output"]
        pooled_output = output_dict["pooled_output"]

        text_mask_len = sample_list["input_mask"].size(-1)

        text_embedding_total = sequence_output[:, :text_mask_len, :]
        image_embedding_total = sequence_output[:, text_mask_len:, :]

        selector_output = self.selector(
            output_dict["scores"].detach(),
            image_embedding_total.detach(),
            text_embedding_total.detach(),
            pooled_output.detach(),
        )

        output_dict.update(selector_output)

        if not self.config["return_hidden_states"]:
            output_dict.pop("sequence_output")
            output_dict.pop("pooled_output")

        return output_dict
