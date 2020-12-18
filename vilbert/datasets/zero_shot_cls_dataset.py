import json
from typing import Any, Dict, List
import random
import os

import torch
from torch.utils.data import Dataset
import numpy as np
import _pickle as cPickle

from pytorch_transformers.tokenization_bert import BertTokenizer
from ._image_features_reader import ImageFeaturesH5Reader
import jsonlines
import sys
import pdb

from collections import defaultdict

def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _load_annotationsVal(annotations_jsonpath, task):

    with jsonlines.open(annotations_jsonpath) as reader:

        # Build an index which maps image id with a list of caption annotations.
        image_entries = {}
        caption_entries = []
        caption_entries_by_class = defaultdict(list)
        image_classes = {}

        for annotation in reader:
            if task == "ZeroShotCUB" or task == "RetrievalCUB":
                image_id = annotation["id"]

            image_entries[image_id] = 1
            image_classes[image_id] = annotation["class_label"]

            for sentences in annotation["sentences"]:
                caption_entries.append({"caption": sentences, "image_id": image_id})
                if sentences not in caption_entries_by_class[annotation["class_label"]]:
                    caption_entries_by_class[annotation["class_label"]].append(sentences)

    caption_entries_unique = []
    caption_classes = []
    for c in caption_entries_by_class:
        caption_entries_unique.extend([{"caption": sentences, "class_label": c} for sentences in caption_entries_by_class[c]])
        caption_classes.extend([c] * len(caption_entries_by_class[c]))

    image_entries = [*image_entries]

    return image_entries, caption_entries, caption_entries_unique, caption_classes, image_classes


class ZeroShotClsDatasetVal(Dataset):
    def __init__(
        self,
        task: str,
        dataroot: str,
        annotations_jsonpath: str,
        split: str,
        image_features_reader: ImageFeaturesH5Reader,
        gt_image_features_reader: ImageFeaturesH5Reader,
        tokenizer: BertTokenizer,
        bert_model,
        clean_datasets,
        padding_index: int = 0,
        max_seq_length: int = 20,
        max_region_num: int = 101,
    ):
        # All the keys in `self._entries` would be present in `self._image_features_reader`
        self._image_entries, self._caption_entries, self._caption_entries_unique, self._caption_classes, self._image_classes = \
            _load_annotationsVal(
            annotations_jsonpath, task
        )
        print(f'num images: {len(self._image_entries)}')
        print(f'num captions: {len(self._caption_entries_unique)}')
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer

        self._split = split
        self._padding_index = padding_index
        self._max_region_num = max_region_num
        self._max_seq_length = max_seq_length
        self.num_labels = 1

        # cache file path data/cache/train_ques
        # cap_cache_path = "data/cocoRetreival/cache/val_cap.pkl"
        # if not os.path.exists(cap_cache_path):
        self.tokenize()
        self.tensorize()
        # cPickle.dump(self._entries, open(cap_cache_path, 'wb'))
        # else:
        # print('loading entries from %s' %(cap_cache_path))
        # self._entries = cPickle.load(open(cap_cache_path, "rb"))
        #
        self.features_all = np.zeros((len(self._image_entries), self._max_region_num, 2048))
        self.spatials_all = np.zeros((len(self._image_entries), self._max_region_num, 5))
        self.image_mask_all = np.zeros((len(self._image_entries), self._max_region_num))

        #print("self._image_entries", len(self._image_entries))
        for i, image_id in enumerate(self._image_entries):
            features, num_boxes, boxes, _ = self._image_features_reader[image_id]

            mix_num_boxes = min(int(num_boxes), self._max_region_num)
            mix_boxes_pad = np.zeros((self._max_region_num, 5))
            mix_features_pad = np.zeros((self._max_region_num, 2048))

            image_mask = [1] * (int(mix_num_boxes))
            while len(image_mask) < self._max_region_num:
                image_mask.append(0)

            mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
            mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

            self.features_all[i] = mix_features_pad
            self.image_mask_all[i] = np.array(image_mask)
            self.spatials_all[i] = mix_boxes_pad

            sys.stdout.write("%d/%d\r" % (i, len(self._image_entries)))
            sys.stdout.flush()

        self.features_all = torch.Tensor(self.features_all).float()
        self.image_mask_all = torch.Tensor(self.image_mask_all).long()
        self.spatials_all = torch.Tensor(self.spatials_all).float()

    @property
    def num_captions(self):
        return len(self._caption_entries_unique)

    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        for entry in self._caption_entries_unique:
            tokens = self._tokenizer.encode(entry["caption"])
            tokens = tokens[: self._max_seq_length - 2]
            tokens = self._tokenizer.add_special_tokens_single_sentence(tokens)

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < self._max_seq_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (self._max_seq_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), self._max_seq_length)
            entry["token"] = tokens
            entry["input_mask"] = input_mask
            entry["segment_ids"] = segment_ids

    def tensorize(self):
        for entry in self._caption_entries_unique:
            token = torch.from_numpy(np.array(entry["token"])).long()
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"])).long()
            entry["segment_ids"] = segment_ids

    def __getitem__(self, index):

        # we iterate through every caption here.
        image_idx = index
        image_id = self._image_entries[image_idx]

        features_all = self.features_all[image_idx:image_idx+1]
        spatials_all = self.spatials_all[image_idx:image_idx+1]
        image_mask_all = self.image_mask_all[image_idx:image_idx+1]

        captions = []
        input_masks = []
        segment_idss = []
        target_all = torch.zeros(len(self._caption_entries_unique))
        caption_idxs = list(range(len(self._caption_entries_unique)))
        for i, entry in enumerate(self._caption_entries_unique):
            caption = entry["token"]
            input_mask = entry["input_mask"]
            segment_ids = entry["segment_ids"]
            captions.append(caption)
            input_masks.append(input_mask)
            segment_idss.append(segment_ids)
            if entry["class_label"] == self._image_classes[image_id]:
                target_all[i] = 1

        return (
            features_all,
            spatials_all,
            image_mask_all,
            captions,
            input_masks,
            segment_idss,
            target_all,
            caption_idxs,
            image_idx,
        )

    def __len__(self):
        return len(self._image_entries)


class ZeroShotClsDatasetValBatch(Dataset):
    def __init__(
        self,
        task: str,
        dataroot: str,
        annotations_jsonpath: str,
        split: str,
        image_features_reader: ImageFeaturesH5Reader,
        gt_image_features_reader: ImageFeaturesH5Reader,
        tokenizer: BertTokenizer,
        bert_model,
        clean_datasets,
        padding_index: int = 0,
        max_seq_length: int = 20,
        max_region_num: int = 101,
    ):
        # All the keys in `self._entries` would be present in `self._image_features_reader`
        self._image_entries, self._caption_entries, self._caption_entries_unique, self._caption_classes, self._image_classes = \
            _load_annotationsVal(
            annotations_jsonpath, task
        )
        print(f'num images: {len(self._image_entries)}')
        print(f'num captions: {len(self._caption_entries_unique)}')
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer

        self._split = split
        self._padding_index = padding_index
        self._max_region_num = max_region_num
        self._max_seq_length = max_seq_length
        self.num_labels = 1

        # cache file path data/cache/train_ques
        # cap_cache_path = "data/cocoRetreival/cache/val_cap.pkl"
        # if not os.path.exists(cap_cache_path):
        self.tokenize()
        self.tensorize()
        # cPickle.dump(self._entries, open(cap_cache_path, 'wb'))
        # else:
        # print('loading entries from %s' %(cap_cache_path))
        # self._entries = cPickle.load(open(cap_cache_path, "rb"))
        #
        self.features_all = np.zeros((len(self._image_entries), self._max_region_num, 2048))
        self.spatials_all = np.zeros((len(self._image_entries), self._max_region_num, 5))
        self.image_mask_all = np.zeros((len(self._image_entries), self._max_region_num))

        #print("self._image_entries", len(self._image_entries))
        for i, image_id in enumerate(self._image_entries):
            features, num_boxes, boxes, _ = self._image_features_reader[image_id]

            mix_num_boxes = min(int(num_boxes), self._max_region_num)
            mix_boxes_pad = np.zeros((self._max_region_num, 5))
            mix_features_pad = np.zeros((self._max_region_num, 2048))

            image_mask = [1] * (int(mix_num_boxes))
            while len(image_mask) < self._max_region_num:
                image_mask.append(0)

            mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
            mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

            self.features_all[i] = mix_features_pad
            self.image_mask_all[i] = np.array(image_mask)
            self.spatials_all[i] = mix_boxes_pad

            sys.stdout.write("%d/%d\r" % (i, len(self._image_entries)))
            sys.stdout.flush()

        self.features_all = torch.Tensor(self.features_all).float()
        self.image_mask_all = torch.Tensor(self.image_mask_all).long()
        self.spatials_all = torch.Tensor(self.spatials_all).float()

    @property
    def num_captions(self):
        return len(self._caption_entries_unique)

    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        for entry in self._caption_entries_unique:
            tokens = self._tokenizer.encode(entry["caption"])
            tokens = tokens[: self._max_seq_length - 2]
            tokens = self._tokenizer.add_special_tokens_single_sentence(tokens)

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < self._max_seq_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (self._max_seq_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), self._max_seq_length)
            entry["token"] = tokens
            entry["input_mask"] = input_mask
            entry["segment_ids"] = segment_ids

    def tensorize(self):
        for entry in self._caption_entries_unique:
            token = torch.from_numpy(np.array(entry["token"])).long()
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"])).long()
            entry["segment_ids"] = segment_ids

    def __getitem__(self, index):

        caption_idx = index
        image_idx = 0

        image_entries = self._image_entries
        features_all = self.features_all
        spatials_all = self.spatials_all
        image_mask_all = self.image_mask_all

        entry = self._caption_entries[caption_idx]
        caption = entry["token"]
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]

        target_all = torch.zeros(len(self._image_entries))
        for i, image_id in enumerate(image_entries):
            if entry["class_label"] == self._image_classes[image_id]:
                target_all[i] = 1

        return (
            features_all,
            spatials_all,
            image_mask_all,
            caption,
            input_mask,
            segment_ids,
            target_all,
            caption_idx,
            image_idx
        )

    def __len__(self):
        return len(self._caption_entries_unique)
