from hab_detection.dataset import get_image_dataset
import json
import sys

from hab_detection.constants import (
    device,
    ZIP_PATH_TRAIN,
    ZIP_PATH_TEST,
    MODEL_SAVE_BASE_FOLDER,
)
from torch.utils.data import DataLoader

with open("experiments.json", "r") as f:
    experiments = json.load(f)


if len(sys.argv) < 3:
    print(
        f"run_experiment.py requires 2 arguments, the dataset type (test or train) and the experiment name."
    )
    exit()

t = sys.argv[1]
if t != "test" and t != "train":
    print("First argument must be either test or train")

experiment_name = sys.argv[2]
if experiment_name not in experiments:
    print(f"Experiment {experiment_name} not in experiments.json")
e = experiments[experiment_name]

dataset = get_image_dataset(
    ZIP_PATH_TRAIN if t == "train" else ZIP_PATH_TEST,
    e["class_designation"],
)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=False,
)

num_bins = len(e["class_designation"]) if e["class_designation"] is not None else 253
labels_dist, _ = np.histogram([], bins=num_bins, range=(0, num_bins + 1))
for batch_idx, (inputs, labels, _) in enumerate(test_loader):
    mask = labels == -1

    labels_dist_temp, _ = np.histogram(
        labels[~mask], bins=num_bins, range=(0, num_bins + 1)
    )

    labels_dist = labels_dist + labels_dist_temp

print("Labels Distribution:")
print(labels_dist)
