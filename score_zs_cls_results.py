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
results = json.load(open(args.results_jsonpath, 'rt'))
score_matrix = np.array(results['score_matrix'])
target_matrix = np.array(results['target_matrix'])

print(sum(score_matrix[:,-1] != 0))

print("---- best 1 ----")
print("caption_classes", caption_classes.shape)
print("score_matrix", score_matrix.shape)
image_classes_pred = caption_classes[score_matrix.argmax(axis=1)]
print("image_classes_pred", len(image_classes_pred))
print("image_classes", len(image_classes))
accuracy = sum(image_classes_pred == image_classes) / image_classes.shape[0]
print(f'accuracy: {accuracy}')

print("---- rank ----")
temp = score_matrix.argsort(axis=1)
ranks = np.empty_like(temp)
for i in range(score_matrix.shape[0]):
    ranks[i, temp[i]] = np.arange(score_matrix.shape[1])

target_ranks = ranks * target_matrix

print("average rank", np.mean(target_ranks[np.nonzero(target_ranks)]))
print("max rank", np.max(target_ranks))
print("min rank", np.min(target_ranks))

avg_ranks = []
curr_c = -1
classes = []
classes_start = []
for i, c in enumerate(caption_classes):
    if c != curr_c:
        avg_ranks.append(np.mean(ranks[:, caption_classes == c], axis=1))
        classes.append(c)
        classes_start.append(i)
        curr_c = c

avg_ranks = np.stack(avg_ranks, axis=1)
print("avg_ranks", avg_ranks.shape)
pred = caption_classes[avg_ranks.argmax(axis=1)]
print("pred == image_classes", image_classes[pred == image_classes])
avg_rank_acc = sum(pred == image_classes) / pred.shape[0]
print(f"max avg rank accuracy: {avg_rank_acc}")
pred = caption_classes[avg_ranks.argmin(axis=1)]
avg_rank_acc = sum(pred == image_classes) / pred.shape[0]
print(f"min avg rank accuracy: {avg_rank_acc}")

print("------ top x any ------")
# print("bottom 10 any ", sum((0 < target_ranks.min(axis=1) <= 10)) / pred.shape[0])
# print("target_ranks.max(axis=1)", target_ranks.max(axis=1))
print("top 1 any ", sum(target_ranks.max(axis=1) >= len(caption_classes) - 1) / pred.shape[0])
print("top 5 any ", sum(target_ranks.max(axis=1) >= len(caption_classes) - 6) / pred.shape[0])
print("top 10 any ", sum(target_ranks.max(axis=1) >= len(caption_classes) - 11) / pred.shape[0])

# print("--------")
# print(image_classes[0])
# print(image_classes[0])
# print(caption_classes.tolist().index(image_classes[0]))
# count = 0
# for i in range(ranks.shape[0]):
#     rank = ranks[i, caption_classes == image_classes[i]]
#     if sum(rank < 200) > 0:
#         print("i", i)
#         print("sum(rank < 200)", sum(rank < 200))
#         print("ranks[:, caption_classes == c]", ranks[i, caption_classes == image_classes[i]])
#         print(np.array(caption_entries_unique)[caption_classes == image_classes[i]])
#         count += 1
#
# print("count", count)
# print("caption_entries_unique", len(caption_entries_unique))
#
# print("-------")
# print("14", np.array(caption_entries_unique)[caption_classes == 14])
# print("166", np.array(caption_entries_unique)[caption_classes == 166])
# print("180", np.array(caption_entries_unique)[caption_classes == 180])
