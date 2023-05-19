# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.realpath(os.path.dirname(__file__)))

from reliable_vqa_eval import ReliabilityEval
from vqa import VQA


def load_json(fname):
    with open(fname, "r") as f:
        data = json.load(f)
    return data


def load_data(
    ques_file, ann_file, res_file, threshold_res_file, risk_tolerances, costs
):
    questions = load_json(ques_file)
    annotations = load_json(ann_file)

    ann_vqa = VQA(annotations=annotations, questions=questions)
    all_qids = ann_vqa.getQuesIds()

    vqa_eval = ReliabilityEval(
        all_qids, risk_tolerances=risk_tolerances, costs=costs, n=2
    )
    res_vqa = ann_vqa.loadRes(VQA(), res_file)

    if threshold_res_file is not None:
        threshold_res_vqa = ann_vqa.loadRes(VQA(), threshold_res_file)
    else:
        threshold_res_vqa = None

    return ann_vqa, res_vqa, threshold_res_vqa, vqa_eval


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run reliable VQA evaluations.")
    parser.add_argument(
        "-q", "--questions", required=True, help="Path to question json file"
    )
    parser.add_argument(
        "-a", "--annotations", required=True, help="Path to annotation json file"
    )
    parser.add_argument(
        "-p", "--predictions", required=True, help="Path to prediction json file"
    )
    parser.add_argument(
        "-t",
        "--threshold_predictions",
        default=None,
        help="Path to prediction json file to choose effective reliability thresholds",
    )
    parser.add_argument(
        "-r",
        "--risk_tols",
        nargs="*",
        type=float,
        default=[0.01, 0.05, 0.1, 0.2],
        help="Risk tolerances for risk-coverage metrics",
    )
    parser.add_argument(
        "-c",
        "--costs",
        nargs="*",
        type=float,
        default=[1, 10, 100],
        help="Cost values for effective reliability metrics",
    )

    return parser.parse_args()


def main(args):
    full_ques_file = args.questions
    full_ann_file = args.annotations
    result_file = args.predictions
    threshold_result_file = args.threshold_predictions
    risk_tols = args.risk_tols
    costs = args.costs

    gt_data, pred_data, threshold_pred_data, evaluator, = load_data(
        full_ques_file,
        full_ann_file,
        result_file,
        threshold_result_file,
        risk_tols,
        costs,
    )

    qids = set(pred_data.getQuesIds())

    if threshold_pred_data is not None:
        threshold_qids = set(threshold_pred_data.getQuesIds())
    else:
        threshold_qids = None

    evaluator.evaluate(
        gt_data,
        pred_data,
        threshold_pred_data,
        quesIds=qids,
        thresholdQuesIds=threshold_qids,
    )

    print(json.dumps(evaluator.accuracy, sort_keys=True, indent=4))


if __name__ == "__main__":
    args = parse_arguments()

    print("\n\n----------")
    print("Arguments:")
    argvar_list = [arg for arg in vars(args)]
    for arg in argvar_list:
        print("\t{}: {}".format(arg, getattr(args, arg)))
    print("----------\n")

    main(args)
