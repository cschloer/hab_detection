from .constants import device, LEARNING_RATE, MODEL_SAVE_BASE_FOLDER

from torchvision import models
import torch
import numpy as np


def load_model(
    model_architecture,
    # model_file is None if we shouldn't load from a previous model.
    # Otherwise path starting with "/" or filename in model_folder starting without /
    model_file,
    model_folder,
    class_designation,
):
    if model_architecture == "resnet50":
        # Load resnet50 segmentation model
        model = models.segmentation.deeplabv3_resnet50(weights=None)

        # Fix BatchNorm2D Bug
        model.backbone.bn1 = torch.nn.BatchNorm2d(64, track_running_stats=False)
        model.classifier[2] = torch.nn.BatchNorm2d(256, track_running_stats=False)

        # Chance first layer to accept 12 input bands (12 bands)
        model.backbone.conv1 = torch.nn.Conv2d(
            12, 64, kernel_size=(7, 7), stride=(2, 2)
        )

        if class_designation is not None:
            # It's a class based problem
            num_classes = len(class_designation)
            model.classifier[4] = torch.nn.Conv2d(
                256,
                num_classes,
                kernel_size=(1, 1),
                stride=(1, 1),
            )

        else:
            # It's a regression problem

            # Change final layer to 1 continuous output
            model.classifier[4] = torch.nn.Sequential(
                torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)),
                torch.nn.Sigmoid(),
            )
        model = model.to(device)

        if model_file is not None:
            # An entire path was passed in
            if model_file.startswith("/"):
                model.load_state_dict(torch.load(model_file, map_location=device))
            else:
                model_file_path = f"{model_folder}/{model_file}"
                model.load_state_dict(torch.load(model_file_path, map_location=device))
    else:
        raise Exception(f"Unknown model architecture {model_architecture}")
    return model


def mask_and_flatten_output(pred, label, flatten=True):
    # Missing data are transformed to -1
    label_masked = torch.where(label == -1, np.nan, label)

    pred_masked = torch.where(torch.isnan(label_masked), np.nan, pred)

    if flatten:
        label_flattened = label_masked.flatten()
        label_final = label_flattened[~torch.isnan(label_flattened)]

        pred_flattened = pred_masked.flatten()
        pred_final = pred_flattened[~torch.isnan(pred_flattened)]

        return pred_final, label_final
    return pred_masked, label_masked


def mse_loss_with_nans(pred, label):
    # Custom loss function that changes removes any pixels that are 254 or 255 from consideration

    pred, label = mask_and_flatten_output(pred, label)
    # Multiply the flattened prediction by 253 to put it in the same range as the target
    pred = pred * 253

    loss = torch.mean((pred - label) ** 2)
    return loss


def get_criterion(class_designation, class_weights):
    if class_designation is None:
        return mse_loss_with_nans
    else:
        lf = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights)
        return lf


def get_optimizer(model):
    # Create adam optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=LEARNING_RATE,
        # weight_decay=weight_decay,
    )
    return optimizer
