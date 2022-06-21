# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import auc
import pandas as pd
from collections import OrderedDict

from mmf.common.registry import registry
from mmf.modules.metrics import BaseMetric
from mmf.utils.distributed import broadcast_tensor, is_master
from mmf.utils.general import get_current_device
from mmf.datasets.processors.processors import EvalAIAnswerProcessor


NULL = np.inf


class ClassificationVQAAccuracyEvaluator:
    """
    Evaluator to compute VQA accuracies. Largely similar to MMF VQAEvalAIAccuracy:
    https://github.com/facebookresearch/mmf/blob/582c7195cbf1eb948436b66c1e9e4bb2e5652a27/mmf/modules/metrics.py#L405
    """

    def __init__(self):
        self.evalai_answer_processor = EvalAIAnswerProcessor()

    def eval_pred_list(self, all_pred_answers, all_gt_answers, *args, **kwargs):
        accuracy = []

        for idx, answer in enumerate(all_pred_answers):
            answer = self.evalai_answer_processor(answer)

            gt_answers = [self.evalai_answer_processor(x) for x in all_gt_answers[idx]]
            gt_answers = list(enumerate(gt_answers))

            gt_acc = []
            for gt_answer in gt_answers:
                other_answers = [item for item in gt_answers if item != gt_answer]
                matching_answers = [item for item in other_answers if item[1] == answer]
                acc = min(1, float(len(matching_answers)) / 3)
                gt_acc.append(acc)
            avgGTAcc = float(sum(gt_acc)) / len(gt_acc)
            accuracy.append(avgGTAcc)

        return accuracy


@registry.register_metric("risk_coverage")
class RiskCoverage(BaseMetric):
    """
    Computes risk-coverage metrics, where the decision to answer or abstain on each sample
    is based on either (1) thresholding model uncertainty estimates, or (2) using the best
    possible selection function (abstaining on incorrect answers). Coverage is defined as
    the fraction of samples that a selective model chooses to answer, whereas risk is the
    error on those answered questions.

    RiskCoverage.compute calculates the following:
    1. Coverage at exact levels of risk, for a selection function based on predicted
       uncertainty estimates as well as the optimal selection function.

    2. Area under the risk-coverage curve (AUC), again for both predicted uncertainties and
       the optimal selection function.

    3. If the input `precomputed_threshold_file` is None, computes the best possible thresholds
       for each of the risk tolerances, and saves to a CSV file. Typically, this is done on a
       validation set, and the thresholds can then be used for the test set eval. If not None,
       the CSV file is loaded, and risk-coverage metrics are computed using the saved thresholds.

    Initialization of this metric has the following arguments:

    - `risk_tolerances`:
          List of floats in the range [0,1]; specifies levels of risk to compute coverage at
    - `save_dir`:
          Directory to save threshold CSV file
    - `precomputed_threshold_file`:
          If not None, path to CSV file with columns "RiskTolerance" and "Threshold" (with the
          corresponding thresholds).
    """
    def __init__(self, risk_tolerances, save_dir, precomputed_threshold_file=None, **kwargs):
        super().__init__("risk_coverage")
        self.required_params = ["scores", "targets", "confidences", "__prediction_report__"]
        self.precomputed_threshold_file = precomputed_threshold_file
        self.risk_tolerances = risk_tolerances
        self.save_dir = save_dir

        self.acc_evaluator = ClassificationVQAAccuracyEvaluator()

    def _broadcast_result(self, result_dict):
        """
        Utility allowing for distributed computation of the metrics.
        """
        broad_result = {}
        for k, t in result_dict.items():
            broad_result[k] = broadcast_tensor(t, src=0)
        return broad_result

    def _get_accuracies(self, pred_answers, gt_answers, *args, **kwargs):
        return self.acc_evaluator.eval_pred_list(pred_answers, gt_answers, *args, **kwargs)

    def _get_risk_coverage(self, scores, confidences, best=False):
        """
        Compute risk and coverage for all points based on model confidences or best possible.
        """
        assert len(scores) == len(confidences), \
            "{} != {}".format(len(scores), len(confidences))

        sorted_confs, sorted_scores = \
            zip(*sorted([tup for tup in zip(confidences, scores)], key=lambda x: -x[int(best)]))

        total_questions = len(sorted_scores)

        risks = []
        coverages = []
        covered = 0.
        cum_score = 0.

        for i in range(total_questions):
            score = sorted_scores[i]

            covered += 1
            cum_score += score
            risks.append(1 - (cum_score / covered))
            coverages.append(covered / total_questions)

        auc_score = auc(coverages, risks)

        return risks, coverages, auc_score, sorted_scores, sorted_confs

    def _get_coverage_at_risk(self, risks, coverages, risk_tolerance, sorted_confs):
        """
        Compute maximum coverage at specific risk (C@R).
        """
        index = len(risks)
        while index > 0 and risks[index - 1] >= risk_tolerance:
            index -= 1
        index -= 1

        if index > -1 and index < (len(risks) - 1) and sorted_confs[index] == sorted_confs[index + 1]:
            while index > -1 and (sorted_confs[index] == sorted_confs[index + 1] or risks[index] >= risk_tolerance):
                index -= 1

        cov = coverages[index] if index > -1 else 0.
        threshold = sorted_confs[index] if index > -1 else 0.
        return cov, threshold, index

    def _get_risk_coverage_from_precomputed_threshold(self, threshold, sorted_confs, risks):
        """
        Compute risk-coverage with threshold from validation set.
        """
        sorted_confs = np.array(sorted_confs)
        answered = np.where(sorted_confs > threshold)[0]

        cov = len(answered) / len(sorted_confs)
        risk = risks[answered[-1]]  # Last index where conf > threshold
        return risk, cov

    def _calc_rc_result(self, scores, confidences, device, best):
        """
        Execute metric calculations, returning dictionary of scores.
        """
        (
            risks, coverages, auc_score, sorted_scores, sorted_confs
        ) = self._get_risk_coverage(scores, confidences, best=best)

        results = OrderedDict()

        if best:
            prefix = "best_"
        else:
            prefix = ""

        key = prefix + "auc"
        results[key] = torch.tensor(auc_score, dtype=torch.float, device=device)

        thresholds = []
        save_thresholds = not best and self.precomputed_threshold_file is None
        for rt in self.risk_tolerances:
            key = prefix + "cov@{}".format(str(rt))
            cov, threshold, _ = self._get_coverage_at_risk(risks, coverages, rt, sorted_confs=sorted_confs)
            results[key] = torch.tensor(cov, dtype=torch.float, device=device)

            if save_thresholds:
                thresholds.append((rt, threshold))

        if save_thresholds:
            csv_path = 'thresholds_at_risk_tolerances.csv'
            csv_path = os.path.join(self.save_dir, csv_path)
            print(f'Saving thresholds at specified risk tolerances to {csv_path}')
            with open(csv_path, "w") as f:
                headers = "RiskTolerance,Threshold"
                f.write(headers + "\n")
                for rt, t in thresholds:
                    f.write("{},{}\n".format(rt,t))

        if not best and self.precomputed_threshold_file:
            # Calculate coverage using precomputed thresholds
            threshold_data = pd.read_csv(self.precomputed_threshold_file)
            risk_tolerances = threshold_data['RiskTolerance'].values
            thresholds = threshold_data['Threshold'].values
            for rt, threshold in zip(risk_tolerances, thresholds):
                risk, cov = self._get_risk_coverage_from_precomputed_threshold(
                    threshold, sorted_confs, risks
                )
                precomp_kv = [
                    ("cov_precomputed_threshold@{}".format(str(rt)), cov),
                    ("risk_precomputed_threshold@{}".format(str(rt)), risk),
                ]
                for key, val in precomp_kv:
                    results[key] = torch.tensor(val, dtype=torch.float, device=device)

        return results

    def calculate(
        self, sample_list, model_output, execute_on_master_only=True, *args, **kwargs
    ):
        device = get_current_device()
        keys = []
        keys.append("auc")
        for rt in self.risk_tolerances:
            keys.append("cov@{}".format(str(rt)))

        if self.precomputed_threshold_file:
            threshold_data = pd.read_csv(self.precomputed_threshold_file)
            risk_tolerances = threshold_data['RiskTolerance'].values
            for rt in risk_tolerances:
                keys.append("cov_precomputed_threshold@{}".format(str(rt)))
                keys.append("actual_risk_precomputed_threshold@{}".format(str(rt)))

        keys.append("best_auc")
        for rt in self.risk_tolerances:
            keys.append("best_cov@{}".format(str(rt)))

        if execute_on_master_only and not is_master():
            # dummy to be overridden in broadcasting
            results = OrderedDict()
            for key in keys:
                # dummy to be overridden in broadcasting
                results[key] = torch.tensor(NULL, dtype=torch.float, device=device)
        else:
            output = []
            expected = []
            confidences = []

            for entry in model_output.prediction_report:
                output.append(entry["answer"])
                expected.append(entry["gt_answers"])
                confidences.append(entry["confidence"])

            acc_scores = self._get_accuracies(output, expected)

            results = self._calc_rc_result(acc_scores, confidences, device, best=False)
            results.update(self._calc_rc_result(acc_scores, confidences, device, best=True))

        if execute_on_master_only:
             results = self._broadcast_result(results)
        return results


@registry.register_metric("effective_reliability")
class EffectiveReliability(BaseMetric):
    """
    Computes the Effective Reliability metric (Phi_c), calculated as follows,
    with variables:
      - x: input
      - g: selection function, with output in {0,1}. g(x)=0 indicates abstention on x,
           whereas g(x)=1 indicates an answer is given for x.
      - Acc: accuracy function; in this case, VQA Accuracy
      - c: cost value

    Phi_c(x) =   Acc(x),   if g(x) = 1 and Acc(x) > 0
                 -c,       if g(x) = 1 and Acc(x) = 0
                 0,        if g(x) = 0.

    The final Phi_c is a summation over Phi_c(x) for each sample x.

    EffectiveReliability.compute calculates the following:
    1. If the input `precomputed_cost_threshold_file` is None, computes the best possible
       thresholds for each of the cost values, and saves to a CSV file. Typically, this is
       done on a validation set, and the thresholds can then be used for the test set eval.
       If not None, the CSV file is loaded, and Phi_c is computed using the saved thresholds.

    2. Best possible Phi_c for each of the input cost values, as well as the
       corresponding risk and coverage (i.e., using the best possible g, which
       abstains only on samples where Acc(x) = 0).

    3. Phi_c for each of the input cost values, where g(x) is always 1 (as is the case for
       models which do not have the option to abstain).

    Initialization of this metric has the following arguments:
    - `cost_values`:
          List of numerical cost values `c` to use for separate Phi_c calculations
    - `save_dir`:
          Directory to save CSV file to
    - `precomputed_cost_threshold_file`:
          If not None, path to CSV file with columns "Cost" (with cost values) and
          "Threshold" (with the corresponding thresholds).
    """
    def __init__(self, cost_values, save_dir, precomputed_cost_threshold_file=None, **kwargs):
        super().__init__("effective_reliability")
        self.required_params = ["scores", "targets", "confidences", "__prediction_report__"]
        self.acc_evaluator = ClassificationVQAAccuracyEvaluator()
        self.cost_values = cost_values
        self.precomputed_cost_threshold_file = precomputed_cost_threshold_file
        self.save_dir = save_dir

    def _get_accuracies(self, pred_answers, gt_answers, *args, **kwargs):
        return self.acc_evaluator.eval_pred_list(pred_answers, gt_answers, *args, **kwargs)

    def _broadcast_result(self, result_dict):
        broad_result = {}
        for k, t in result_dict.items():
            broad_result[k] = broadcast_tensor(t, src=0)
        return broad_result

    def _get_sorted_costs(self, sorted_scores):
        """
        Return dictionary mapping a cost value c to an array sorted_costs
        of the same length as sorted_scores, where each entry sorted_costs[i]
        contains the value phi_c[i], computed assuming sample i was NOT abstained on
        (i.e., g(x_i) = 1).
        """
        cost2sorted_costs = {}
        for c in self.cost_values:
            sorted_costs = []
            for s in sorted_scores:
                if s == 0:
                    sorted_costs.append(-c)
                else:
                    sorted_costs.append(s)
            cost2sorted_costs[c] = sorted_costs
        return cost2sorted_costs

    def _calc_best_possible_phi(self, sorted_costs):
        """
        Given an array with phi_c values computed without abstention,
        calculate the best possible phi_c (where g(x) = 0 iff. Acc(x) = 0,
        for each x).

        Compute the corresponding risk and coverage as well.
        """
        total_questions = len(sorted_costs)

        # Add up all positive entries of sorted_costs
        sorted_costs = np.array(sorted_costs)
        max_phi = sorted_costs[sorted_costs > 0].sum()
        best_possible_phi = max_phi / total_questions

        # Coverage
        num_answered = (sorted_costs > 0).sum()
        best_coverage = num_answered / total_questions

        # Risk
        # max_phi is the sum of Acc(x) scores on samples where Acc(x) > 0.
        # A perfect model gets Acc(x) = 1 each time, which equals num_answered,
        # giving a risk of 0.
        best_risk = 1 - (max_phi / num_answered)

        return best_possible_phi, best_coverage, best_risk

    def _calc_cost_threshold(self, sorted_confs, sorted_costs):
        """
        Given a list of model confidences and corresponding phi_c cost values
        computed without abstention, return the confidence threshold for abstention
        which maximizes phi_c.
        """
        all_phis = []
        for i in range(len(sorted_confs)):
            phi = sum(sorted_costs[:i])
            all_phis.append(phi)
        all_phis = np.array(all_phis)
        threshold_index = np.argmax(all_phis)
        threshold = sorted_confs[threshold_index]
        return threshold

    def _calc_phi_from_precomputed_threshold(
            self, c, threshold, sorted_confs, sorted_scores
    ):
        """
        Given a cost value c, threshold on model confidence,
        model confidences and associated accuracy scores,
        return phi_c and corresponding coverage and risk.
        """
        cum_score = 0.
        acc_score = 0.
        num_answered = 0
        total_questions = len(sorted_confs)
        for i in range(total_questions):
            if sorted_confs[i] > threshold:
                # Choose to answer
                acc_score += sorted_scores[i]
                num_answered += 1
                if sorted_scores[i] == 0:
                    cum_score -= c
                else:
                    cum_score += sorted_scores[i]
            else:
                # Choose to abstain; no updates to the score.
                pass
        phi = cum_score / total_questions
        coverage = num_answered / total_questions
        risk = 1 - (acc_score / num_answered)
        return phi, coverage, risk

    def _calc_phi_no_abstention(self, c, sorted_scores):
        """
        Given an array of accuracy scores, compute phi_c where the
        model must answer each question (i.e., g(x) = 1 for all samples x).
      	"""
        cum_score = 0
        for s in sorted_scores:
            # No option to abstain
            if s == 0:
                cum_score -= c
            else:
                cum_score += s
        phi_no_abstention = cum_score / len(sorted_scores)
        return phi_no_abstention

    def _calc_er_result(self, scores, confidences, device, best):
        assert len(scores) == len(confidences), \
            "{} != {}".format(len(scores), len(confidences))
        sorted_confs, sorted_scores = \
            zip(*sorted(
                [tup for tup in zip(confidences, scores)],
                key=lambda x: -x[int(best)]
            ))

        results = {}
        cost2sorted_costs = self._get_sorted_costs(sorted_scores)

        if best:
            # Compute best possible effective reliability score
            for c in self.cost_values:
                best_possible_phi, best_coverage, best_risk = \
                    self._calc_best_possible_phi(cost2sorted_costs[c])
                for key, val in [
                        (f'best_phi_c@{c}', best_possible_phi),
                        (f'best_cov_phi_c@{c}', best_coverage),
                        (f'best_risk_phi_c@{c}', best_risk)
                ]:
                    results[key] = torch.tensor(val, dtype=torch.float, device=device)
        else:
            if self.precomputed_cost_threshold_file is None:
                # Compute cost thresholds and save to a new file
                thresholds = []
                for c in self.cost_values:
                    threshold = self._calc_cost_threshold(
                        sorted_confs,
                        cost2sorted_costs[c],
                    )
                    thresholds.append((c, threshold))
                csv_path = 'thresholds_at_costs.csv'
                csv_path = os.path.join(self.save_dir, csv_path)
                print(f'Saving effective reliability cost threshold info to {csv_path}')
                with open(csv_path, "w") as f:
                    headers = "Cost,Threshold"
                    f.write(headers + "\n")
                    for c, t in thresholds:
                        f.write("{},{}\n".format(c,t))
            else:
                # Load precomputed cost thresholds, and compute
                # effective reliability score
                threshold_data = pd.read_csv(self.precomputed_cost_threshold_file)
                costs = threshold_data['Cost'].values
                thresholds = threshold_data['Threshold'].values
                for c, threshold in zip(costs, thresholds):
                    phi, cov, risk = \
                        self._calc_phi_from_precomputed_threshold(
                            c, threshold, sorted_confs, sorted_scores
                        )
                    for key, val in [
                            (f'phi_c@{c}', phi),
                            (f'cov_phi_c@{c}', cov),
                            (f'risk_phi_c@{c}', risk)
                    ]:
                        results[key] = torch.tensor(
                            val, dtype=torch.float, device=device
                        )

                    # Compute effective reliability without abstention option
                    phi_no_abstention = self._calc_phi_no_abstention(c, sorted_scores)
                    key = f'no_abstention_phi_c@{c}'
                    results[key] = torch.tensor(
                        phi_no_abstention, dtype=torch.float, device=device
                    )

        return results

    def calculate(
        self, sample_list, model_output, execute_on_master_only=True, *args, **kwargs
    ):
        device = get_current_device()
        keys = []

        if self.precomputed_cost_threshold_file:
            threshold_data = pd.read_csv(self.precomputed_cost_threshold_file)
            costs = threshold_data['Cost'].values
            for c in costs:
                keys.append(f'phi_c@{c}')
                keys.append(f'cov_phi_c@{c}')
                keys.append(f'risk_phi_c@{c}')
                keys.append(f'no_abstention_phi_c@{c}')

        for c in self.cost_values:
            keys.append(f'best_phi_c@{c}')
            keys.append(f'best_cov_phi_c@{c}')
            keys.append(f'best_risk_phi_c@{c}')

        if execute_on_master_only and not is_master():
            results = OrderedDict()
            for key in keys:
                # dummy to be overridden in broadcasting
                results[key] = torch.tensor(NULL, dtype=torch.float, device=device)
        else:
            output = []
            expected = []
            confidences = []

            for entry in model_output.prediction_report:
                output.append(entry["answer"])
                expected.append(entry["gt_answers"])
                confidences.append(entry["confidence"])

            acc_scores = self._get_accuracies(output, expected)

            results = self._calc_er_result(acc_scores, confidences, device, best=False)
            results.update(self._calc_er_result(acc_scores, confidences, device, best=True))

        if execute_on_master_only:
             results = self._broadcast_result(results)
        return results


@registry.register_metric("ece")
class ECE(BaseMetric):
    """
    Expected Calibration Error. See [1] for more details.

    ECE calculation credit to Geoff Pleiss' implementation at:
    https://github.com/gpleiss/temperature_scaling

    Chuan Guo*, Geoff Pleiss*, Yu Sun*, Kilian Q. Weinberger.
    On Calibration of Modern Neural Networks. ICML 2017.
    """
    def __init__(self, n_bins, multiple_choice_eval=True, **kwargs):
        super().__init__("ece")
        self.required_params = ["scores", "targets", "confidences", "mc_indices", "__prediction_report__"]

        self.mc = multiple_choice_eval
        if not self.mc:
            # VQA Accuracy
            self.acc_evaluator = ClassificationVQAAccuracyEvaluator()

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def _broadcast_result(self, result_dict):
        broad_result = {}
        for k, t in result_dict.items():
            broad_result[k] = broadcast_tensor(t, src=0)
        return broad_result

    def _get_accuracies(self, pred_answers, gt_answers, *args, **kwargs):
        if self.mc:
            acc = (torch.Tensor(pred_answers) == torch.Tensor(gt_answers)).float()
        else:
            acc =  self.acc_evaluator.eval_pred_list(
                pred_answers, gt_answers, *args, **kwargs
            )
        return acc

    def _calc_ece_result(self, accuracies, confidences, device):
        accuracies = torch.Tensor(accuracies)
        confidences = torch.Tensor(confidences)
        ece = torch.zeros(1, device=device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        ece = ece.item()
        results = {}
        key = 'ece'
        results[key] = torch.tensor(ece, dtype=torch.float, device=device)
        return results

    def calculate(
        self, sample_list, model_output, execute_on_master_only=True, *args, **kwargs
    ):
        device = get_current_device()

        if execute_on_master_only and not is_master():
            # dummy to be overridden in broadcasting
            results = {}
            results['ece'] = torch.tensor(NULL, dtype=torch.float, device=device)
        else:
            output = []
            expected = []
            confidences = []

            for entry in model_output.prediction_report:
                if self.mc:
                    output.append(entry["answer_id"])
                    expected.append(entry["mc_answer_id"])
                else:
                    output.append(entry["answer"])
                    expected.append(entry["gt_answers"])
                confidences.append(entry["confidence"])

            acc_scores = self._get_accuracies(output, expected)
            results = self._calc_ece_result(acc_scores, confidences, device)

        if execute_on_master_only:
             results = self._broadcast_result(results)
        return results
