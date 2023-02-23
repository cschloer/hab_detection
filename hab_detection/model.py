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
        model = models.segmentation.deeplabv3_resnet50(pretrained=False)

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
                model.load_state_dict(torch.load(model_file, map_location=device))
    else:
        raise Exception(f"Unknown model architecture {model_architecture}")
    return model


def mask_and_flatten_output(pred, label, flatten=True):
    # Missing data are 254 and 255

    label_masked_land = torch.where(label == 254, np.nan, label)
    label_masked_all = torch.where(label_masked_land == 255, np.nan, label_masked_land)

    pred_masked_all = torch.where(torch.isnan(label_masked_all), np.nan, pred)

    if flatten:
        label_flattened = label_masked_all.flatten()
        label_final = label_flattened[~torch.isnan(label_flattened)]

        pred_flattened = pred_masked_all.flatten()
        # Multiply the flattened input by 253 to put it in the same range as the target
        pred_final = pred_flattened[~torch.isnan(pred_flattened)] * 253

        return pred_final, label_final
    return pred_masked_all * 253, label_masked_all


def mse_loss_with_nans(pred, label):
    # Custom loss function that changes removes any pixels that are 254 or 255 from consideration

    pred_final, label_final = mask_and_flatten_output(pred, label)

    loss = torch.mean((pred_final - label_final) ** 2)
    return loss


def get_criterion():
    return mse_loss_with_nans


def get_optimizer(model):
    # Create adam optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=LEARNING_RATE,
        # weight_decay=weight_decay,
    )
    return optimizer
