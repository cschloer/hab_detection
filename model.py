from constants import device, LEARNING_RATE

from torchvision import models
import torch
import numpy as np


def load_model(model_path):
    model = models.segmentation.deeplabv3_resnet50(pretrained=False)
    # Chance first layer to accept 12 input bands (12 bands)
    model.backbone.conv1 = torch.nn.Conv2d(12, 64, kernel_size=(7, 7), stride=(2, 2))
    # Change final layer to 1 continuous output
    model.classifier[4] = torch.nn.Sequential(
        torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)),
        torch.nn.Sigmoid(),
    )
    model = model.to(device)

    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    print(model)
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
