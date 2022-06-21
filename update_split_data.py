# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import argparse
import numpy as np


def load_npy(path):
    with open(path, "rb") as f:
        db = np.load(f, allow_pickle=True)
    return db


def save_npy(path, data):
    np.save(path, data)


def get_feature_path(data_split, image_id):
    return os.path.join(data_split, "{}.pth".format(image_id))


def get_image_name(data_split, image_id):
    return os.path.join(data_split, "COCO_{}_{:012d}".format(data_split, image_id))


def update_image_data(data_split, data):
    for d in data:
        if "image_id" in d:
            d["feature_path"] = get_feature_path(data_split, d["image_id"])
            d["image_name"] = get_image_name(data_split, d["image_id"])

    return data


def parse_arguments():
    parser = argparse.ArgumentParser(description="Update CLIP-ViL annotations for CLIP features")
    parser.add_argument("--input_dir", required=True, help="Path to directory with .npy annotation files")
    parser.add_argument("--output_dir", required=True, help="Output directory for udpated .npy annotation files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    anno_dir = args.input_dir
    updated_dir = args.output_dir

    if not os.path.exists(updated_dir):
        os.makedirs(updated_dir)

    split_files = glob.glob(os.path.join(anno_dir, "*.npy"))

    split_name = "val2014"

    for filepath in split_files:
        curr_data = load_npy(filepath)
        curr_data = update_image_data(split_name, curr_data)

        updated_path = os.path.join(updated_dir, os.path.basename(filepath))

        assert not os.path.exists(updated_path)

        save_npy(updated_path, curr_data)
