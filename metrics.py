import numpy as np
import math
import torch
from helpers import log
from constants import device
from model import mse_loss_with_nans


def get_model_performance(model, loader):
    # model_cpu = model.cpu()
    model.eval()

    all_preds = torch.tensor([])
    all_labels = torch.tensor([])
    counter = 0
    sum = 0
    sum_gradient = 0
    sum_from_loss_func = 0
    sum_without_hab = 0
    sum_average_distance = 0
    sum_average_distance_without_hab = 0
    for batch_idx, (inputs, labels, _) in enumerate(loader):
        counter += 1
        # print(f"{batch_idx + 1} / {len(loader)}")
        # if counter > 10:
        #  break
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        preds = model(inputs)["out"]  # make prediction
        labels_masked_land = torch.where(labels == 254, np.nan, labels)
        labels_masked_all = torch.where(
            labels_masked_land == 255, np.nan, labels_masked_land
        )
        labels_flattened = labels_masked_all.flatten()
        pixel_labels = labels_flattened[~torch.isnan(labels_flattened)].cpu().detach()
        pixel_labels_nonzero = pixel_labels[~(pixel_labels == 0)].cpu().detach()

        preds_flattened = preds.flatten()
        pixel_preds = (
            (preds_flattened[~torch.isnan(labels_flattened)] * 253).cpu().detach()
        )
        pixel_preds_nonzero = pixel_preds[~(pixel_labels == 0)].cpu().detach()

        all_preds = torch.cat(
            (
                all_preds,
                pixel_preds,
            ),
            0,
        )
        all_labels = torch.cat(
            (
                all_labels,
                pixel_labels,
            ),
            0,
        )
        loss = ((pixel_labels - pixel_preds) ** 2).mean(axis=0)
        loss_without_hab = ((pixel_labels_nonzero - pixel_preds_nonzero) ** 2).mean(
            axis=0
        )

        loss_from_loss_func = mse_loss_with_nans(preds, labels).cpu().detach()
        average_distance = np.abs(pixel_labels - pixel_preds).mean()
        average_distance_without_hab = np.abs(
            pixel_labels_nonzero - pixel_preds_nonzero
        ).mean()

        sum += loss.item()
        loss.backward()
        sum_gradient += loss.item()
        sum_from_loss_func += loss_from_loss_func.item()
        sum_without_hab += loss_without_hab.item()
        sum_average_distance += average_distance.item()
        sum_average_distance_without_hab += average_distance_without_hab.item()

        # break
        del inputs
        del labels
        del preds
        del labels_masked_land
        del labels_masked_all
        del labels_flattened
        del pixel_labels
        del pixel_labels_nonzero
        del preds_flattened
        del pixel_preds
        del pixel_preds_nonzero

    mses = ((all_labels - all_preds) ** 2).mean(axis=0)
    log(f"Averaged gradient: {math.sqrt(sum_gradient / (batch_idx + 1))}")
    log(f"Mean squared error: {mses} : {np.sqrt(mses)}")
    log(f"Averaged MSE: {math.sqrt(sum / (batch_idx + 1))}")
    log(
        f"Averaged MSE from loss function: {math.sqrt(sum_from_loss_func / (batch_idx + 1))}"
    )
    log(f"Averaged MSE without no HAB: {math.sqrt(sum_without_hab / (batch_idx + 1))}")
    log(f"Averaged distance from correct: {(sum_average_distance / (batch_idx + 1))}")
    log(
        f"Averaged distance from correct without no HAB: {(sum_average_distance_without_hab / (batch_idx + 1))}"
    )
