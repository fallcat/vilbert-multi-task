# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import os
import pickle

import lmdb
import numpy as np
import tqdm
from itertools import product


MAP_SIZE = 1099511627776


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lmdb_file1", default=None, type=str, help="Path to original lmdb file"
    )
    parser.add_argument(
        "--lmdb_file2", default=None, type=str, help="Path to generated LMDB file that has grid features"
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    id_list = []
    env1 = lmdb.open(args.lmdb_file1, map_size=MAP_SIZE)
    env2 = lmdb.open(args.lmdb_file2, map_size=MAP_SIZE)

    with env1.begin(write=False) as txn1:
        with env2.begin(write=True) as txn2:
            image_ids = pickle.loads(txn1.get("keys".encode()))
            for img_id in image_ids:
                id_list.append(img_id)
                item = pickle.loads(txn1.get(img_id))
                image_id = item["image_id"]
                image_h = int(item["image_h"])
                image_w = int(item["image_w"])
                features = item["features"].reshape(-1, 2048)
                num_boxes = int(item["num_boxes"])
                boxes = item["boxes"].reshape(-1, 4)
                # x1, y1, width, height
                width = int(image_w / 10)
                height = int(image_h / 10)
                xs = range(0, image_w, width)
                ys = range(0, image_h, height)
                product_xy = product(xs, ys)
                product_xy = np.array([[item[0], item[1], width, height] for item in product_xy])
                print("product_xy", product_xy)
                txn2.put(img_id, pickle.dumps(item))
                print("boxes", boxes.shape)
                print("image_h", image_h)
                print("image_w", image_w)
                exit()
            # for infile in tqdm.tqdm(infiles):
            #     reader = np.load(infile, allow_pickle=True)
            #     item = {}
            #     item["image_id"] = reader.item().get("image_id")
            #     img_id = str(item["image_id"]).encode()
            #     id_list.append(img_id)
            #     item["image_h"] = reader.item().get("image_height")
            #     item["image_w"] = reader.item().get("image_width")
            #     item["num_boxes"] = reader.item().get("num_boxes")
            #     item["boxes"] = reader.item().get("bbox")
            #     item["features"] = reader.item().get("features")
            #     txn2.put(img_id, pickle.dumps(item))
            txn2.put("keys".encode(), pickle.dumps(id_list))
