from hab_detection.dataset import get_image_dataset
import json
import sys
import numpy as np

from hab_detection.constants import (
    device,
    STRUCTURED_FOLDER_PATH_TRAIN,
    STRUCTURED_FOLDER_PATH_TEST,
    ZIP_PATH_TRAIN,
    ZIP_PATH_TEST,
    MODEL_SAVE_BASE_FOLDER,
)
from torch.utils.data import DataLoader
import torch

with open("experiments.json", "r") as f:
    experiments = json.load(f)


if len(sys.argv) < 3:
    print(
        f"run_dataset_statistics.py requires 2 arguments, the dataset type (test or train) and the experiment name."
    )
    exit()

t = sys.argv[1]
if t != "test" and t != "train":
    print("First argument must be either test or train")

experiment_name = sys.argv[2]
if experiment_name not in experiments:
    print(f"Experiment {experiment_name} not in experiments.json")
e = experiments[experiment_name]
cd = e["class_designation"]

print("Getting dataset.")
dataset = get_image_dataset(
    STRUCTURED_FOLDER_PATH_TRAIN if t == "train" else STRUCTURED_FOLDER_PATH_TEST,
    cd,
)

other_dataset = get_image_dataset(
    STRUCTURED_FOLDER_PATH_TRAIN if t == "test" else STRUCTURED_FOLDER_PATH_TEST,
    cd,
)
print(f"Dataset size: {len(dataset)}")

loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
    drop_last=False,
)
other_loader = DataLoader(
    other_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
    drop_last=False,
)
print("Dataset loaded. Calculating labels distribution.")

num_classes = len(cd) if cd is not None else 254
labels_dist = np.zeros(num_classes)
all_dist = np.zeros(254)
premax = 0
postmax = 0
for batch_idx, (_, labels, _, raw_labels) in enumerate(loader):
    mask = labels == -1

    labels_dist_temp = np.bincount(
        labels[~mask].flatten(),
        minlength=num_classes,
    )

    # all_mask = np.logical_or(raw_labels == 254, raw_labels == 255)
    premax = max(premax, torch.max(raw_labels))
    postmax = max(postmax, torch.max(raw_labels[~mask]))
    all_dist_temp = np.bincount(
        raw_labels[~mask].flatten(),
        minlength=254,
    )

    labels_dist = labels_dist + labels_dist_temp
    all_dist = all_dist + all_dist_temp
    if batch_idx % 200 == 0:
        print(f"Batch {batch_idx} - premax {premax}, postmax {postmax}")

premax = 0
postmax = 0
print(f"Dataset size for both: {len(dataset) + len(other_dataset)}")
for batch_idx, (_, labels, _, raw_labels) in enumerate(other_loader):
    mask = labels == -1

    labels_dist_temp = np.bincount(
        labels[~mask].flatten(),
        minlength=num_classes,
    )

    # all_mask = np.logical_or(raw_labels == 254, raw_labels == 255)
    premax = max(premax, torch.max(raw_labels))
    postmax = max(postmax, torch.max(raw_labels[~mask]))
    all_dist_temp = np.bincount(
        raw_labels[~mask].flatten(),
        minlength=254,
    )

    labels_dist = labels_dist + labels_dist_temp
    all_dist = all_dist + all_dist_temp
    if batch_idx % 200 == 0:
        print(f"Batch {batch_idx} - premax {premax}, postmax {postmax}")


print("Labels Count:")
print(labels_dist)
print("Labels Dist:")
print(labels_dist / np.max(labels_dist))
print("Labels Fraction:")
print(labels_dist / np.sum(labels_dist))
print("Weights:")
print(1 / (labels_dist / np.max(labels_dist)))
print("All Pixels:")
print(all_dist)
all_dist = np.where(all_dist == 0, 1, all_dist)
print("All Weights:")
print(1 / (all_dist / np.max(all_dist)))
exit()

print("Calculating mean and std now.")

dataset = get_image_dataset(
    ZIP_PATH_TRAIN if t == "train" else ZIP_PATH_TEST,
    cd,
    transform=False,
)

loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
    drop_last=False,
)

psum = np.zeros((12,))
psum_sq = np.zeros((12,))


# loop through images
counter = 0
count = 0
for (inputs, _, _, _) in loader:

    psum += inputs.sum(axis=[0, 2, 3]).numpy()
    psum_sq += (inputs ** 2).sum(axis=[0, 2, 3]).numpy()
    counter += 1
    count += inputs.shape[0] * inputs.shape[2] * inputs.shape[3]

# mean and STD
mean = psum / count
var = (psum_sq / count) - (mean ** 2)
std = np.sqrt(var)
# output
print("Data stats:")
print(f"- mean: {mean}")
print(f"- std:  {std}")
