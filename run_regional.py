import sys
import json
from hab_detection.visualize import visualize
from hab_detection.dataset import get_data
from hab_detection.constants import (
    STRUCTURED_FOLDER_PATH_TEST,
)

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
features, labels, _ = get_data(
    STRUCTURED_FOLDER_PATH_TEST,
)
print(features[0])
"dataset_test_structured"
match = re.findall(
    ".*\/dataset_test_structured\/(\d_\d)\/.*",
    features[0],
    re.IGNORECASE,
)
print(match[0])
