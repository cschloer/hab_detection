import sys
import json
from hab_detection.visualize import visualize
from .run_experiment import get_input_experiment

e, epoch, model_file = get_input_experiment(sys.argv, "run_visualize")


visualize(
    e["name"],
    e["class_designation"],
    epoch,
    model_file,
)
