import sys
import torch
import numpy as np
import pickle
import re
import json
from hab_detection.metrics import get_model_performance
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
for band in range(12):

    def set_zero(image):
        image[band, :, :] = 0
        return image

    def set_random(image):
        image[band, :, :] = torch.from_numpy(
            np.random.normal(
                0,
                1,
                (
                    64,
                    64,
                ),
            )
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
    _, metrics, _ = get_model_performance(
        model,
        loader,
        class_designation,
        num_batches=-1,
        calculate_2d_hist=True,
        calculate_statistics=True,
    )
    print(f"Running statistics for band index {band}")
    print(metrics)
    results[band] = metrics
    print("----------------")

for band in results.keys():
    avg_acc = np.mean(results[band]["MulticlassAccuracy"].cpu().numpy())
    print(f"Band {band} average accuracy: {avg_acc}")
    for k in results[band].keys():
        results[band][k] = results[band][k].cpu().numpy().tolist()

model_save_folder = f"{MODEL_SAVE_BASE_FOLDER}/{experiment_name}"
image_save_folder = f"{model_save_folder}/visualize/test"
# filename = f"{image_save_folder}/band_zero_results.json"
filename = f"{image_save_folder}/band_random_results.json"
with open(filename, "w") as f:
    json.dump(results, f)
