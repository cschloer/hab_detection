import sys
import json
from hab_detection.visualize import visualize
from hab_detection.dataset import get_data

with open("experiments.json", "r") as f:
    experiments = json.load(f)


if len(sys.argv) < 2:
    print(f"run_regional.py requires atleast 1 argument, the experiment id.")
    exit()
if len(sys.argv) > 4:
    print(
        "run_regional.py takes maximally 3 arguments, the experiment id, the dataset type, and the filename/path of the model to preload ."
    )
    exit()

e = sys.argv[1]
dataset_type = "test"
if len(sys.argv) >= 3:
    dataset_type = sys.argv[2]

model_file = "epoch_recent.pt"
if len(sys.argv) == 4:
    model_file = sys.argv[3]

experiment_name = sys.argv[1]
if experiment_name not in experiments:
    print(f"Experiment {experiment_name} not in experiments.json")

e = experiments[experiment_name]
class_designation = e["class_designation"]
features, labels, _ = get_data(
    STRUCTURED_FOLDER_PATH_TEST,
)
print(features[0])
