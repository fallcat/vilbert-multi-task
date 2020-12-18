import json
import numpy as np
import argparse
import jsonlines
from collections import defaultdict
import sys
np.set_printoptions(threshold=sys.maxsize)

def _load_annotationsVal(annotations_jsonpath, task=None):

    with jsonlines.open(annotations_jsonpath) as reader:

        # Build an index which maps image id with a list of caption annotations.
        image_entries = {}
        caption_entries = []
        caption_entries_by_class = defaultdict(list)
        image_classes = {}

        for annotation in reader:
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

parser = argparse.ArgumentParser()

parser.add_argument(
        "--annotations_jsonpath",
        default="/Users/weiqiuyou/Documents/Penn/data/CaltechBird/test.jsonline",
        type=str,
        help="annotations_jsonpath",
    )

parser.add_argument(
        "--results_jsonpath",
        default="results/mul_val_result.json",
        type=str,
        help="results_jsonpath",
    )

args = parser.parse_args()

image_entries, caption_entries, caption_entries_unique, caption_classes, image_classes = \
    _load_annotationsVal(args.annotations_jsonpath)

image_classes = np.array(list(image_classes.values()))
caption_classes = np.array(caption_classes)

# load results
results = json.load(open('results/mul_val_result.json', 'rt'))
score_matrix = np.array(results['score_matrix'])
target_matrix = np.array(results['target_matrix'])

print(sum(score_matrix[:,-1] != 0))

print("---- best 1 ----")
image_classes_pred = caption_classes[score_matrix[:,:-80].argmax(axis=1)]
print("image_classes_pred", image_classes_pred)
print("image_classes", image_classes)
accuracy = sum(image_classes_pred == image_classes) / image_classes.shape[0]
print(f'accuracy: {accuracy}')

print("---- rank ----")
image_classes_pred