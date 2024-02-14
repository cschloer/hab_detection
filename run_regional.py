import sys
import re
import json
from hab_detection.metrics import get_model_performance
from hab_detection.model import load_model
from hab_detection.visualize import visualize
from hab_detection.dataset import get_data, ImageData
from hab_detection.constants import (
    STRUCTURED_FOLDER_PATH_TEST,
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
    STRUCTURED_FOLDER_PATH_TEST,
)
print(features[0])
results = {}
for i in range(len(features)):
    feature = features[i]
    match = re.findall(
        ".*\/dataset_test_structured\/(\d_\d)\/.*",
        feature,
        re.IGNORECASE,
    )
    region = match[0]
    if region not in results:
        results[region] = {"features": [], "labels": []}
    results[region]["features"].append(feature)
    results[region]["labels"].append(labels[i])

region_statistics = {}
for region in results.keys():
    region_statistics[region] = {}
    features = results[region]
    print(
        f"Running statistics for region {region}, # of images {len(results[region]['features'])}"
    )
    dataset = ImageData(
        results[region]["features"],
        results[region]["labels"],
        class_designation,
        zip_path=None,
        randomize=False,
        transform=True,
        in_memory=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    _, metrics, hist_2d = get_model_performance(
        model,
        loader,
        class_designation,
        num_batches=-1,
        calculate_2d_hist=True,
        calculate_statistics=True,
    )
    print(metrics)
    print("----------------")
