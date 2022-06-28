# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import random
import numpy as np

from collections import defaultdict


random.seed(1234)


def load_npy(path):
    with open(path, "rb") as f:
        db = np.load(f, allow_pickle=True)
    return db


def save_npy(path, data):
    np.save(path, data)


def get_imageid2questions(data):
    imageid2ques = defaultdict(list)
    meta_data = []
    for d in data:
        if "image_id" not in d:
            meta_data.append(d)
        else:
            imageid2ques[d["image_id"]].append(d)
    return dict(imageid2ques), meta_data


def randomly_split_images(imageid2ques, ratios, sanity_check=False):
    assert sum(ratios.values()) == 1.

    num_images = len(imageid2ques)

    image_ids = list(imageid2ques.keys())
    random.shuffle(image_ids)

    image_splits = {}

    i = 0

    for split, r in ratios.items():
        n = int(round(r * num_images))
        image_splits[split] = image_ids[i: i + n]
        i += n

    if sanity_check:
        # Check that the number of images covered in the splits
        # matches the total number of images. Then, check that
        # all images in each split are unique and do not overlap
        # with any of the other splits.

        split_lens = {k: len(v) for k, v in image_splits.items()}

        total_lens = sum(split_lens.values())

        assert total_lens == num_images, \
            "expected: {}, got: {} ({})".format(num_images, image_splits, split_lens)

        for split_name, image_ids in image_splits.items():
            assert len(set(image_ids)) == len(image_ids)

            img_split_len = len(image_ids)
            img_split_percent = img_split_len / num_images

            print("# images -- {}: {} ({:.3f})".format(split_name, img_split_len, img_split_percent))

            for other_split, other_ids in image_splits.items():
                if other_split != split_name:
                    assert len(set(image_ids) & set(other_ids)) == 0

    return image_splits


def divide_data_by_split(imageid2ques, image_splits, meta_data):
    for split_name, image_ids in image_splits.items():
        split_data = []
        for imid in image_ids:
            split_data += imageid2ques[imid]
        yield split_name, np.array(meta_data + split_data, dtype=object)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Split val2014 data into dev, val, and test sets.")
    parser.add_argument("--input_file", required=True, help="Path to .npy annotation file")
    parser.add_argument("--output_dir", required=True, help="Output directory for split .npy annotation files")
    parser.add_argument("--percentages", nargs=3, default=[0.4, 0.1, 0.5], help="\% for dev, val, test (respectively)")
    return parser.parse_args()


def main(args):
    ratios = dict(zip(["dev", "val", "test"], args.percentages))

    data_path = args.input_file
    output_dir = args.output_dir
    output_fname_form = "imdb_val2014-{}.npy"

    data = load_npy(data_path)

    imageid2ques, meta_data = get_imageid2questions(data)

    image_splits = randomly_split_images(imageid2ques, ratios, sanity_check=True)

    total_len = len(data) - len(meta_data)
    split_total_len = 0
    
    for split_name, split_data in divide_data_by_split(imageid2ques, image_splits, meta_data):
        out_fname = output_fname_form.format(split_name)
        out_path = os.path.join(output_dir, out_fname)

        assert not os.path.exists(out_path)

        split_len = len(split_data) - len(meta_data)  # don't count metadata
        split_percent = split_len / total_len
        split_total_len += split_len

        print("# questions -- {}: {} ({:.3f})".format(split_name, split_len, split_percent))

        save_npy(out_path, split_data)
    
    print("total # questions = {}".format(total_len))
    print("split total # questions = {}".format(split_total_len))

    assert total_len == split_total_len


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
