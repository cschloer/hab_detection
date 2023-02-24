from torchmetrics import Accuracy, ConfusionMatrix, StatScores
from torch.utils.data import DataLoader
import torch
from .constants import (
    device,
    ZIP_PATH_TRAIN,
    ZIP_PATH_TEST,
    MODEL_SAVE_BASE_FOLDER,
)


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

    accuracy = Accuracy(
        task="multiclass",
        num_classes=len(class_designation),
        ignore_index=-1,
    ).to(device)
    cm = ConfusionMatrix(
        task="multiclass",
        num_classes=len(class_designation),
        ignore_index=-1,
    ).to(device)
    ss = ConfusionMatrix(
        task="multiclass",
        num_classes=len(class_designation),
        ignore_index=-1,
    ).to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, labels, _) in enumerate(loader):
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device)

            preds = model(inputs)["out"]  # make prediction
