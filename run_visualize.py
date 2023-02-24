import sys
import json
from hab_detection.visualize import visualize
from .run_experiment import get_input_experiment

e, _, model_file = get_input_experiment(sys.argv, "run_visualize")


visualize(
    e["name"],
    e["batch_size"],
    e["class_designation"],
    model_file=model_file,
)
