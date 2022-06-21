# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmf.common.registry import registry
from mmf.models.pythia import Pythia

from reliable_vqa.models.selective_predictors import SelectivePredictor


@registry.register_model("select_pythia")
class SelectPythia(Pythia):
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def config_path(cls):
        return "configs/experiments/pythia/vqa2/select_pred.yaml"

    def build(self):
        super().build()
        self._init_selector()

        # See https://github.com/g-luo/news_clippings_training/blob/master/models/utils.py#L37
        # See https://github.com/facebookresearch/mmf/blob/b0b2af7ff21eac648598ebd94d09628127ffc4b5/mmf/models/visual_bert.py#L417
        if self.config.get("freeze_vqa", False):
            for p in self.word_embedding.parameters():
                p.requires_grad = False

            for p in self.image_feature_embeddings_list.parameters():
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
        feat_size = self._get_classifier_input_dim()
        self.config[config_attr].params["qi_feat_size"] = feat_size
        num_choices = registry.get(self._datasets[0] + "_num_final_outputs")
        self.selector = SelectivePredictor(
            select_type,
            feat_size=feat_size,
            num_answers=num_choices,
            **self.config[config_attr].params
        )

    def get_optimizer_parameters(self, config):
        if not self.config.get("freeze_vqa", False):
            combine_layer = self.image_text_multi_modal_combine_layer
            params = [
                {"params": self.word_embedding.parameters()},
                {"params": self.image_feature_embeddings_list.parameters()},
                {"params": self.text_embeddings.parameters()},
                {"params": combine_layer.parameters()},
                {"params": self.classifier.parameters()},
                {
                    "params": self.image_feature_encoders.parameters(),
                    "lr": (config.optimizer.params.lr * 0.1),
                },
                {"params": self.selector.parameters()},
            ]
        else:
            params = [{"params": self.selector.parameters()}]

        if hasattr(self.config.selector, "sel_lr"):
            params[-1]["lr"] = self.config.selector.sel_lr

        return params

    def forward(self, sample_list):
        sample_list.text = self.word_embedding(sample_list.text)
        text_embedding_total = self.process_text_embedding(sample_list)

        image_embedding_total, _ = self.process_feature_embedding(
            "image", sample_list, text_embedding_total
        )

        if self.inter_model is not None:
            image_embedding_total = self.inter_model(image_embedding_total)

        joint_embedding = self.combine_embeddings(
            ["image", "text"], [image_embedding_total, text_embedding_total]
        )

        model_output = {"scores": self.calculate_logits(joint_embedding)}

        selector_output = self.selector(
            model_output["scores"].detach(),
            image_embedding_total.detach(),
            text_embedding_total.detach(),
            joint_embedding.detach(),
        )

        model_output.update(selector_output)

        return model_output
