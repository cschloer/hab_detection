import sys
import json
from hab_detection.train import train

with open("experiments.json", "r") as f:
    experiments = json.load(f)

# First argument is experiment id, second argument is the epoch start

if len(sys.argv) < 2:
    print("run_experiment.py requires atleast 1 argument, the experiment id.")
    exit()
if len(sys.argv) > 4:
    print(
        "run_experiment.py takes maximally 2 arguments, the experiment id and the start epoch."
    )
    exit()

e = sys.argv[1]
epoch_start = 0
if len(sys.argv) >= 3:
    epoch_start_str = sys.argv[2]
    if not epoch_start_str.isdigit():
        print("The second argument must be an integer")
        exit()
    epoch_start = int(epoch_start_str)

model_file = None
if len(sys.argv) == 4:
    model_file = sys.argv[3]


train(
    experiments[e]["name"],
    experiments[e]["batch_size"],
    epoch_start=epoch_start,
    model_file=model_file,
)
