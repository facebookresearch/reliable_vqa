# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmf.common.registry import registry
from mmf.datasets.builders.vqa2.builder import VQA2Builder

from reliable_vqa.datasets.vqa2 import VQA2DatasetExtended


@registry.register_builder("vqa2_extended")
class VQA2DatasetExtendedBuilder(VQA2Builder):
    def __init__(
        self,
        dataset_name="vqa2_extended",
        dataset_class=VQA2DatasetExtended,
        *args,
        **kwargs,
    ):
        super().__init__(dataset_name=dataset_name, dataset_class=dataset_class, *args, **kwargs)
        self.dataset_class = VQA2DatasetExtended

    @classmethod
    def config_path(cls):
        return "configs/datasets/vqa2_extended.yaml"

    # Note from mmf.datasets.builders.vqa2.builder.VQA2Builder:
    # TODO: Deprecate this method and move configuration updates directly to processors
    def update_registry_for_model(self, config):
        super().update_registry_for_model(config)
        if hasattr(self.dataset, "text_processor"):
            registry.register(
                self.dataset_name + "_text_processor",
                self.dataset.text_processor
            )
