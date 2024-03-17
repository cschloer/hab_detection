import sys
import torch
import time
import numpy as np
import math
import random
import pickle
import re
import json
from hab_detection.metrics import get_model_performance, get_metric_tracker
from hab_detection.model import load_model
from hab_detection.visualize import visualize
from hab_detection.dataset import get_data, ImageData
from hab_detection.constants import (
    STRUCTURED_FOLDER_PATH_TEST,
    STRUCTURED_FOLDER_PATH_TRAIN,
    MODEL_SAVE_BASE_FOLDER,
)
from torch.utils.data import DataLoader

with open("experiments.json", "r") as f:
    experiments = json.load(f)


if len(sys.argv) != 3:
    print(
        f"run_regional.py requires exactly 2 arguments, the experiment id and the model path."
    )
    exit()

model_file = sys.argv[2]

experiment_name = sys.argv[1]
if experiment_name not in experiments:
    print(f"Experiment {experiment_name} not in experiments.json")

e = experiments[experiment_name]
class_designation = e["class_designation"]

model = load_model(e["model_architecture"], model_file, None, class_designation)

features, labels, _ = get_data(
    # STRUCTURED_FOLDER_PATH_TRAIN,
    STRUCTURED_FOLDER_PATH_TEST,
)

results = {}
count = 0
used_pixels = {}
random.seed(42)
tracker = get_metric_tracker(class_designation)
for i in range(75):
    x = None
    y = None
    while f"{x}-{y}" in used_pixels or x is None:
        x = math.floor(random.random() * 64)
        y = math.floor(random.random() * 64)
    used_pixels[f"{x}-{y}"] = True

    def set_random(image):
        pixel = image[:, x, y]

        image[:, :, :] = torch.from_numpy(
            np.full(
                (64, 64, 12),
                pixel,
            ).reshape((12, 64, 64))
        )
        return image

    dataset = ImageData(
        features,
        labels,
        class_designation,
        zip_path=None,
        randomize=False,
        transform=True,
        in_memory=False,
        transform_image_func=set_random,
    )
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    start = time.time()
    get_model_performance(
        model,
        loader,
        class_designation,
        num_batches=-1,
        calculate_2d_hist=False,
        calculate_statistics=True,
        pixel_mode=(
            x,
            y,
        ),
        tracker=tracker,
    )
    print(f"Running statistics for round {i} took {time.time() - start}")
    print(tracker.compute_all())
    print("----------------")

results = tracker.compute_all()
avg_acc = np.mean(results["MulticlassAccuracy"].cpu().numpy())
print(f"Average accuracy: {avg_acc}")
for k in results.keys():
    results[k] = results[k].cpu().numpy().tolist()

model_save_folder = f"{MODEL_SAVE_BASE_FOLDER}/{experiment_name}"
image_save_folder = f"{model_save_folder}/visualize/test"
# filename = f"{image_save_folder}/band_zero_results.json"
filename = f"{image_save_folder}/spatial_random_results.json"
with open(filename, "w") as f:
    json.dump(results, f)