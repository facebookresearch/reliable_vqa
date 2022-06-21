# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from mmf.common.registry import registry
from mmf.utils.general import filter_grads
from mmf.models.movie_mcan import MoVieMcan

from reliable_vqa.models.selective_predictors import SelectivePredictor


@registry.register_model("select_movie_mcan")
class SelectMoVieMcan(MoVieMcan):
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def config_path(cls):
        return "configs/experiments/movie_mcan/vqa2/select_pred.yaml"

    def build(self):
        super().build()
        self._init_selector()

        # See https://github.com/g-luo/news_clippings_training/blob/master/models/utils.py#L37
        # See https://github.com/facebookresearch/mmf/blob/b0b2af7ff21eac648598ebd94d09628127ffc4b5/mmf/models/visual_bert.py#L417
        # Freezing the "trunk" (i.e., multimodal encoder) weights is already handled.
        if self.config.get("freeze_vqa", False):
            for p in self.word_embedding.parameters():
                p.requires_grad = False

            for p in self.image_feature_embeddings_list.sga.parameters():
                p.requires_grad = False

            for p in self.image_feature_embeddings_list.sga_pool.parameters():
                p.requires_grad = False

            for p in self.image_feature_embeddings_list.cbn.parameters():
                p.requires_grad = False

            for p in self.text_embeddings.parameters():
                p.requires_grad = False

            for p in self.image_text_multi_modal_combine_layer.parameters():
                p.requires_grad = False

            for p in self.classifier.parameters():
                p.requires_grad = False
            
            for p in self.image_feature_encoders.parameters():
                p.requires_grad = False

    def _init_selector(self):
        config_attr = "selector"
        select_type = self.config[config_attr].type
        feat_size = self.image_text_multi_modal_combine_layer.out_dim
        self.config[config_attr].params["qi_feat_size"] = self.image_text_multi_modal_combine_layer.out_dim
        num_choices = registry.get(self._datasets[0] + "_num_final_outputs")
        self.selector = SelectivePredictor(
            select_type,
            feat_size=feat_size,
            num_answers=num_choices,
            **self.config[config_attr].params
        )

    def get_optimizer_parameters(self, config):
        if self.config.get("freeze_vqa", False):
            params = [{"params": filter_grads(self.selector.parameters())}]
        else:
            combine_layer = self.image_text_multi_modal_combine_layer
            params = [
                {"params": filter_grads(self.word_embedding.parameters())},
                {
                    "params": filter_grads(
                        self.image_feature_embeddings_list.sga.parameters()
                    )
                },
                {
                    "params": filter_grads(
                        self.image_feature_embeddings_list.sga_pool.parameters()
                    )
                },
                {
                    "params": filter_grads(
                        self.image_feature_embeddings_list.cbn.parameters()
                    ),
                    "lr": (
                        config.optimizer.params.lr * config.training.encoder_lr_multiply
                    ),
                },
                {"params": filter_grads(self.text_embeddings.parameters())},
                {"params": filter_grads(combine_layer.parameters())},
                {"params": filter_grads(self.classifier.parameters())},
                {"params": filter_grads(self.image_feature_encoders.parameters())},
                {"params": filter_grads(self.selector.parameters())},
            ]

        if hasattr(self.config.selector, "sel_lr"):
            params[-1]["lr"] = self.config.selector.sel_lr

        return params

    def forward(self, sample_list):
        sample_list.text_mask = sample_list.text.eq(0)
        sample_list.text = self.word_embedding(sample_list.text)
        text_embedding_total, text_embedding_vec = self.process_text_embedding(
            sample_list
        )

        feature_sga, feature_cbn = self.process_feature_embedding(
            "image", sample_list, text_embedding_total, text_embedding_vec[:, 0]
        )

        joint_embedding = self.combine_embeddings(
            ["image", "text"], [feature_sga, feature_cbn, text_embedding_vec[:, 1]]
        )

        model_output = {"scores": self.calculate_logits(joint_embedding)}

        logits = model_output["scores"]

        # for three branch movie+mcan model
        if model_output["scores"].dim() == 3:
            joint_embedding = joint_embedding[:, 0]
            logits = logits[:, 0]

        image_embedding_vec = torch.cat([feature_sga, feature_cbn], dim=-1)

        if sample_list.image_feature_0.dim() == 4:
            uni_image_feats = sample_list.image_feature_0
            batch_size, feat_dim, _, _ = sample_list.image_feature_0.size()
            uni_image_feats = uni_image_feats.view(batch_size, feat_dim, -1).contiguous()
            uni_image_feats = uni_image_feats.permute(0, 2, 1)

        selector_output = self.selector(
            logits.detach(),
            image_embedding_vec.detach(),
            text_embedding_vec[:, 1].detach(),
            joint_embedding.detach(),
            uni_text_feats=text_embedding_vec[:, 1].detach(),
            uni_image_feats=uni_image_feats.detach()
        )

        model_output.update(selector_output)

        return model_output
