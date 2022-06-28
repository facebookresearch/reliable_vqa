# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_unk_softmax(x, dim, mask_idx):
    """
    Copied from VQAAccuracy.
    """
    x1 = F.softmax(x, dim=dim)
    x1[:, mask_idx] = 0
    x1_sum = torch.sum(x1, dim=1, keepdim=True)
    y = x1 / x1_sum
    return y


class SelectivePredictor(nn.Module):
    def __init__(self, selector_type, *args, **kwargs):
        super().__init__()

        if selector_type == "combo_embeddings_logit":
            self.module = ComboEmbeddingsFullLogitSelectivePredictor(*args, **kwargs)
        elif selector_type == "calibration":
            self.module = Calibration(*args, **kwargs)
        else:
            raise NotImplementedError("Unknown selector type: {}".format(selector_type))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class ComboEmbeddingsFullLogitSelectivePredictor(nn.Module):
    def __init__(self, feat_size, **kwargs):
        super(ComboEmbeddingsFullLogitSelectivePredictor, self).__init__()

        ans_embed_size = kwargs["answer_hidden_size"]
        n_answers = kwargs["num_answers"]

        self.use_softmax = kwargs.get("use_softmax", False)
        self.use_qi_embed = kwargs.get("use_qi_embed", False)

        image_feat_size  = kwargs["image_feat_size"]
        text_feat_size   = kwargs["text_feat_size"]
        image_embed_size = kwargs["image_hidden_size"]
        text_embed_size  = kwargs["text_hidden_size"]

        if self.use_qi_embed:
            qi_feat_size   = kwargs["qi_feat_size"]
            qi_embed_size = kwargs["qi_hidden_size"]
            input_size = image_embed_size + text_embed_size + ans_embed_size + qi_embed_size
        else:
            input_size = image_embed_size + text_embed_size + ans_embed_size

        self.pool_image_feats = kwargs.get("pool_image_embedding", False)
        self.pool_text_feats = kwargs.get("pool_text_embedding", False)
        self.pool_image_dim = kwargs.get("pool_image_dim", 1)
        self.pool_text_dim = kwargs.get("pool_text_dim", 1)
        self.pool_type = kwargs.get("pool_type", None)
        if self.pool_image_feats or self.pool_text_feats:
            assert self.pool_type is not None

        use_batchnorm = kwargs.get("use_batchnorm", False)
        if use_batchnorm:
            self.selective_predictor = nn.Sequential(
                nn.Linear(input_size, kwargs["hidden_1"]),
                nn.Dropout(p=kwargs["dropout"]),
                nn.BatchNorm1d(kwargs["hidden_1"]),
                nn.ReLU(),
                nn.Linear(kwargs["hidden_1"], kwargs["hidden_2"]),
                nn.Dropout(p=kwargs["dropout"]),
                nn.BatchNorm1d(kwargs["hidden_2"]),
                nn.ReLU(),
                nn.Linear(kwargs["hidden_2"], 2),
                nn.Softmax(dim=-1),
            )
        else:
            self.selective_predictor = nn.Sequential(
                nn.Linear(input_size, kwargs["hidden_1"]),
                nn.Dropout(p=kwargs["dropout"]),
                nn.ReLU(),
                nn.Linear(kwargs["hidden_1"], kwargs["hidden_2"]),
                nn.Dropout(p=kwargs["dropout"]),
                nn.ReLU(),
                nn.Linear(kwargs["hidden_2"], 2),
                nn.Softmax(dim=-1),
            )

        if self.use_softmax:
            self.s_embed = nn.Sequential(
                nn.Softmax(dim=-1), nn.Linear(n_answers, ans_embed_size), nn.ReLU()
            )
        else:
            self.s_embed = nn.Sequential(
                nn.ReLU(), nn.Linear(n_answers, ans_embed_size), nn.ReLU()
            )

        # Text & image embedding layers
        self.init_embedding_layers(
            kwargs.get("double_embedding_layers", False),
            use_batchnorm,
            image_feat_size,
            image_embed_size,
            text_feat_size,
            text_embed_size
        )

        # Fused text+image feature layer
        if self.use_qi_embed:
            self.qi_embed = nn.Sequential(
                  nn.ReLU(), nn.Linear(qi_feat_size, qi_embed_size), nn.ReLU()
            )

    def init_embedding_layers(
            self,
            double_embedding_layers,
            use_batchnorm,
            image_feat_size,
            image_embed_size,
            text_feat_size,
            text_embed_size
    ):
        if double_embedding_layers:
            if use_batchnorm:
                self.image_embed = nn.Sequential(
                    nn.ReLU(), nn.BatchNorm1d(image_feat_size),
                    nn.Linear(image_feat_size, image_embed_size), nn.ReLU(),
                    nn.Linear(image_embed_size, image_embed_size), nn.ReLU()
                )
                self.text_embed = nn.Sequential(
                    nn.ReLU(), nn.BatchNorm1d(text_feat_size),
                    nn.Linear(text_feat_size, text_embed_size), nn.ReLU(),
                    nn.Linear(text_embed_size, text_embed_size), nn.ReLU(),
                )
            else:
                self.image_embed = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(image_feat_size, image_embed_size), nn.ReLU(),
                    nn.Linear(image_embed_size, image_embed_size), nn.ReLU()
                )
                self.text_embed = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(text_feat_size, text_embed_size), nn.ReLU(),
                    nn.Linear(text_embed_size, text_embed_size), nn.ReLU(),
                )
        else:
            if use_batchnorm:
                self.image_embed = nn.Sequential(
                    nn.ReLU(), nn.BatchNorm1d(image_feat_size),
                    nn.Linear(image_feat_size, image_embed_size), nn.ReLU()
                )
                self.text_embed = nn.Sequential(
                    nn.ReLU(), nn.BatchNorm1d(text_feat_size),
                    nn.Linear(text_feat_size, text_embed_size), nn.ReLU()
                )
            else:
                self.image_embed = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(image_feat_size, image_embed_size), nn.ReLU()
                )
                self.text_embed = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(text_feat_size, text_embed_size), nn.ReLU()
                )

    def pool_features(self, features, pool_dim, pool_type):
        if pool_type == 'max':
            pooled_features = features.max(pool_dim).values
        elif pool_type == 'mean':
            pooled_features = features.mean(pool_dim)
        else:
            raise Exception(f'Pool type {pool_type} not recognized.')
        return pooled_features

    def forward(self, answer_logits, image_feats, text_feats, qi_embed, **kwargs):
        if self.pool_image_feats:
            image_feats = self.pool_features(
                image_feats, self.pool_image_dim, self.pool_type
            )
        if self.pool_text_feats:
            text_feats = self.pool_features(
                text_feats, self.pool_text_dim, self.pool_type
            )

        image_emb =  self.image_embed(image_feats)
        text_emb =  self.text_embed(text_feats)
        answer_emb = self.s_embed(answer_logits)

        if self.use_qi_embed:
            qi_emb =  self.qi_embed(qi_embed)
            input_feat = torch.cat([image_emb, text_emb, qi_emb, answer_emb], -1)
        else:
            input_feat = torch.cat([image_emb, text_emb, answer_emb], -1)

        return {"confidences": self.selective_predictor(input_feat)}


class Calibration(nn.Module):
    def __init__(self, feat_size, **kwargs):
        super(Calibration, self).__init__()
        n_answers = kwargs['num_answers']
        self.weight = torch.nn.Parameter(torch.ones(n_answers))
        self.bias = torch.nn.Parameter(torch.zeros(n_answers))

    def forward(self, answer_logits, image_feats, text_feats, qi_embed, **kwargs):
        calibrated_logits = self.weight * answer_logits + self.bias
        return {'scores': calibrated_logits}
