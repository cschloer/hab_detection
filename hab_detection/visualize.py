from torch.utils.data import DataLoader
import torch
from .constants import (
    device,
    ZIP_PATH_TRAIN,
    ZIP_PATH_TEST,
    MODEL_SAVE_BASE_FOLDER,
)
from .metrics import get_metric_tracker


def visualize(
    experiment_name,
    class_designation,
    epoch,
    model_file,
    dataset_type="test",
):
    # Set the config to print to stdout
    set_config("test")

    image_save_folder = f"{MODEL_SAVE_BASE_FOLDER}/{experiment_name}/visualize/{epoch}"
    os.makedirs(image_save_folder, exist_ok=True)

    log(f"Loading the dataset")
    if dataset_type == "test":
        dataset = get_image_dataset(ZIP_PATH_TEST, class_designation)
    elif dataset_type == "train":
        dataset = get_image_dataset(ZIP_PATH_TRAIN, class_designation)
    else:
        raise Exception(f"Unknown dataset type {dataset_type}")
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    log(f"Loading the model")
    model = load_model(
        model_architecture, model_file, model_save_folder, class_designation
    )

    _, metrics = get_model_performance(model, loader, class_designation, num_batches=10)
    log(pprint.pformat(metrics))
