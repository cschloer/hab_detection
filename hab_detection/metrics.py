import numpy as np
import math
import torch

from torchmetrics import (
    MetricTracker,
    MetricCollection,
    MeanSquaredError,
    Precision,
    Accuracy,
    ConfusionMatrix,
    Recall,
    Specificity,
)
from .helpers import log
from .constants import device
from .model import get_criterion


def get_metric_tracker(class_designation):
    if class_designation is None:
        tracker = MetricTracker(
            MetricCollection(
                [
                    MeanSquaredError(squared=False),
                ]
            ),
        )
    else:
        tracker = MetricTracker(
            MetricCollection(
                [
                    Accuracy(
                        task="multiclass",
                        num_classes=len(class_designation),
                        ignore_index=-1,
                        average="macro",
                    ).to(device),
                    ConfusionMatrix(
                        task="multiclass",
                        num_classes=len(class_designation),
                        ignore_index=-1,
                    ).to(device),
                    Precision(
                        task="multiclass",
                        num_classes=len(class_designation),
                        ignore_index=-1,
                        average="macro",
                    ).to(device),
                    Recall(
                        task="multiclass",
                        num_classes=len(class_designation),
                        ignore_index=-1,
                        average="macro",
                    ).to(device),
                    Specificity(
                        task="multiclass",
                        num_classes=len(class_designation),
                        ignore_index=-1,
                        average="macro",
                    ).to(device),
                ]
            )
        ).to(device)
    tracker.increment()
    return tracker


def get_model_performance(
    model,
    loader,
    class_designation,
    class_weights=None,
    num_batches=-1,
    calculate_2d_hist=False,
):
    # model_cpu = model.cpu()
    tracker = get_metric_tracker(class_designation)
    hist_2d = (
        np.zeros(
            (
                len(class_designation),
                254,
            )
        )
        if class_designation is not None
        else None
    )
    with torch.no_grad():
        model.eval()

        total_loss = 0
        counter = 0
        for batch_idx, (inputs, labels, _, raw_labels) in enumerate(loader):
            # print(f"{batch_idx + 1} / {len(loader)}")
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device)
            preds = model(inputs)  # make prediction

            # Only calculate the loss if its regression or class weights is passed in
            if class_designation is None or class_weights is not None:
                criterion = get_criterion(class_designation, class_weights)
                loss = criterion(preds, labels)  # Calculate cross entropy loss
                total_loss += loss.item()

            mask = labels == -1
            preds = preds[~mask]
            labels = labels[~mask]

            tracker.update(preds, labels)

            if class_designation is not None and calculate_2d_hist:
                raw_labels = raw_labels[~mask]
                h, _ = np.histogramdd(
                    np.array([preds, raw_labels]).T,
                    bins=[len(class_designation), 254],
                )
                print(h.shape, hist_2d.shape)
                hist_2d += h

            counter += 1
            if num_batches >= 0 and counter >= num_batches:
                break

    return total_loss / batch_idx, tracker.compute_all(), hist_2d
