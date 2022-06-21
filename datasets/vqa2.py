# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
from typing import Dict

from mmf.datasets.builders.vqa2.dataset import VQA2Dataset


class VQA2DatasetExtended(VQA2Dataset):
    def __init__(self, config: Dict, dataset_type: str, index, *args, **kwargs):
        super().__init__(config, dataset_type, index, *args, name='vqa2_extended', **kwargs)
        self.add_multiple_choice = config.get('add_multiple_choice', False)

        self.save_logits = config.get("save_logits", False)
        self.save_logit_dir = None

        if self.save_logits:
            self.save_logit_dir = config["save_logit_dir"]

        self.save_confs = config.get("save_confs", False)
        self.save_conf_dir = None

        if self.save_confs:
            self.save_conf_dir = config["save_conf_dir"]

    def add_answer_info(self, sample_info, sample):
        if "answers" in sample_info:
            answers = sample_info["answers"]
            answer_processor_arg = {"answers": answers}

            if self.use_ocr:
                answer_processor_arg["tokens"] = sample_info["ocr_tokens"]
            processed_soft_copy_answers = self.answer_processor(answer_processor_arg)

            # sample.answers = processed_soft_copy_answers["answers"]
            sample.answers_indices = processed_soft_copy_answers["answers_indices"]
            sample.targets = processed_soft_copy_answers["answers_scores"]

            if self.add_multiple_choice:
                mc = self.answer_processor({'answers': [sample_info['multiple_choice_answer']]})
                sample.mc_indices = \
                    torch.Tensor([mc['answers_indices'][0].item()]).long()

        return sample

    def _masked_unk_softmax(self, x, dim, mask_idx):
        """
        Copied function from VQAAccuracy.
        """
        x1 = torch.nn.functional.softmax(x, dim=dim)
        x1[:, mask_idx] = 0
        x1_sum = torch.sum(x1, dim=1, keepdim=True)
        y = x1 / x1_sum
        return y

    def _get_answers_and_confidences(self, output):
        output = self._masked_unk_softmax(output, 1, 0)
        conf, ans_inds = output.max(dim=1)

        return conf, ans_inds

    def format_for_prediction(self, report):
        confidences, answers = self._get_answers_and_confidences(report.scores)

        if report.get("confidences", None) is not None:
            confidences = report["confidences"]

            if confidences.dim() == 2:
                confidences = confidences[:, 1]

        predictions = []
        answer_space_size = self.answer_processor.get_true_vocab_size()

        for idx, question_id in enumerate(report.question_id):
            answer_id = answers[idx].item()

            if answer_id >= answer_space_size:
                answer_id -= answer_space_size
                answer = report.context_tokens[idx][answer_id]
                if answer == self.context_processor.PAD_TOKEN:
                    answer = "unanswerable"
            else:
                answer = self.answer_processor.idx2word(answer_id)

            gt_answers = []
            for gt_idx in report.answers_indices[idx]:
                gt_ans = self.answer_processor.idx2word(gt_idx.item())
                gt_answers.append(gt_ans)

            pred_result = {
                "question_id": question_id.item(),
                "answer": answer,
                "confidence": confidences[idx].item(),
                "gt_answers": gt_answers,
            }

            if self.add_multiple_choice:
                pred_result['mc_answer_id'] = \
                    report.mc_indices[idx].item()

                pred_result['answer_id'] = answer_id

            predictions.append(pred_result)

            if self.save_logits:
                logit_path = os.path.join(self.save_logit_dir, "{}.pth".format(pred_result["question_id"]))
                with open(logit_path, "wb") as outf:
                    torch.save(report.scores[idx].cpu(), outf)


            if self.save_confs:
                conf_path = os.path.join(self.save_conf_dir, "{}.pth".format(pred_result["question_id"]))
                with open(conf_path, "wb") as outf:
                    torch.save(confidences[idx].cpu(), outf)

        return predictions
