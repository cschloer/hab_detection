import sys
import json
from hab_detection.visualize import visualize

with open("experiments.json", "r") as f:
    experiments = json.load(f)

# First argument is experiment id, second argument is the epoch start

if len(sys.argv) < 2:
    print(f"run_visualize.py requires atleast 1 argument, the experiment id.")
    exit()
if len(sys.argv) > 4:
    print(
        "run_visualize.py takes maximally 3 arguments, the experiment id, the start epoch, and the filename/path of the model to preload ."
    )
    exit()

e = sys.argv[1]
epoch = 0
if len(sys.argv) >= 3:
    epoch_str = sys.argv[2]
    if not epoch_str.isdigit():
        print("The second argument must be an integer")
        exit()
    epoch = int(epoch_str)

model_file = None
if len(sys.argv) == 4:
    model_file = sys.argv[3]

experiment_name = sys.argv[1]
if experiment_name not in experiments:
    print(f"Experiment {experiment_name} not in experiments.json")

e = experiments[experiment_name]

visualize(
    e["name"],
    e["class_designation"],
    epoch,
    model_file,
)
