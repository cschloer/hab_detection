import sys
import json
from hab_detection.train import train


def get_input_experiment(args, script_name):
    with open("experiments.json", "r") as f:
        experiments = json.load(f)

    # First argument is experiment id, second argument is the epoch start

    if len(args) < 2:
        print(f"{script_name} requires atleast 1 argument, the experiment id.")
        exit()
    if len(args) > 4:
        print(
            "{script_name} takes maximally 3 arguments, the experiment id, the start epoch, and the filename/path of the model to preload ."
        )
        exit()

    e = args[1]
    epoch_start = 0
    if len(args) >= 3:
        epoch_start_str = args[2]
        if not epoch_start_str.isdigit():
            print("The second argument must be an integer")
            exit()
        epoch_start = int(epoch_start_str)

    model_file = None
    if len(args) == 4:
        model_file = args[3]

    experiment_name = args[1]
    if experiment_name not in experiments:
        print(f"Experiment {experiment_name} not in experiments.json")

    e = experiments[experiment_name]
    return e, epoch_start, model_file


e, epoch_start, model_file = get_input_experiment(sys.argv, "run_experiment.py")


train(
    e["name"],
    e["batch_size"],
    e["class_designation"],
    e["class_weights"],
    e["model_architecture"],
    epoch_start=epoch_start,
    model_file=model_file,
)
