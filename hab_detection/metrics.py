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
from .model import mse_loss_with_nans


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
                    ).to(device),
                    Recall(
                        task="multiclass",
                        num_classes=len(class_designation),
                        ignore_index=-1,
                    ).to(device),
                    Specificity(
                        task="multiclass",
                        num_classes=len(class_designation),
                        ignore_index=-1,
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
):
    # model_cpu = model.cpu()
    tracker = get_metric_tracker(class_designation)
    with torch.no_grad():
        model.eval()

        total_loss = 0
        for batch_idx, (inputs, labels, _) in enumerate(loader):
            # print(f"{batch_idx + 1} / {len(loader)}")
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device)
            preds = model(inputs)["out"]  # make prediction

            # Only calculate the loss if its regression or class weights is passed in
            if class_designation is None or class_weights is not None:
                criterion = get_criterion(class_designation, class_weights)
                loss = criterion(preds, labels)  # Calculate cross entropy loss
                total_loss += loss.item()

            if class_designation is None:
                # Mask the unused values for metrics
                preds = preds[~labels == -1]
                labels = labels[~labels == -1]

            tracker.update(preds, labels)

            counter += 1
            if num_batches >= 0 and counter >= num_batches:
                break

    return total_loss / batch_idx, tracker.compute_all()
