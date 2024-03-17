import numpy as np
import pprint
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
from .dataset import get_weighted_all_dist


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
                        average="none",
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
                        average="none",
                    ).to(device),
                    Recall(
                        task="multiclass",
                        num_classes=len(class_designation),
                        ignore_index=-1,
                        average="none",
                    ).to(device),
                    Specificity(
                        task="multiclass",
                        num_classes=len(class_designation),
                        ignore_index=-1,
                        average="none",
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
    # If we only want the loss
    calculate_statistics=True,
    # If we only evaluate a single pixel (when checking geospatial influence)
    pixel_mode=None,
):
    # model_cpu = model.cpu()
    if calculate_statistics:
        tracker = get_metric_tracker(class_designation)
        tracker2 = get_metric_tracker(class_designation)
    if calculate_2d_hist:
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
    weighted_all_dist = get_weighted_all_dist(class_designation)
    batch_idx = -1
    with torch.no_grad():
        model.eval()

        total_loss = 0
        counter = 0
        for batch_idx, (inputs, labels, raw_input_, raw_labels) in enumerate(loader):
            # print(f"{batch_idx + 1} / {len(loader)}")
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device)
            # TODO PUT BACK AFTER VUISUALIZATION TESTING
            """
            if batch_idx < 2:
                print("MEAN VARIANCE IN BATCH")
                print(torch.mean(inputs, (0, 2, 3)))
                print(torch.var(inputs, (0, 2, 3)))
                print(
                    "ratio after mask",
                    round(
                        torch.numel(
                            inputs[
                                ~(
                                    torch.broadcast_to(
                                        torch.unsqueeze(labels, 1) == -1, inputs.shape
                                    )
                                )
                            ]
                        )
                        / torch.numel(inputs),
                        2,
                    ),
                )
                print("masked values")
                print(
                    torch.mean(
                        inputs[
                            ~(
                                torch.broadcast_to(
                                    torch.unsqueeze(labels, 1) == -1, inputs.shape
                                )
                            )
                        ]
                    ),
                )
                print(
                    torch.var(
                        inputs[
                            ~(
                                torch.broadcast_to(
                                    torch.unsqueeze(labels, 1) == -1, inputs.shape
                                )
                            )
                        ]
                    ),
                )
                print("pre normalization")
                print(torch.mean(raw_input_))
                print(torch.var(raw_input_))
                print("___________________________________")
            """

            preds = model(inputs)  # make prediction
            if isinstance(preds, dict):
                preds = preds["out"]

            # Only calculate the loss if its regression or class weights is passed in
            if class_designation is None or class_weights is not None:
                criterion = get_criterion(class_designation, class_weights)
                raw_loss = criterion(preds, labels)  # Calculate cross entropy loss
                pixel_weights = torch.from_numpy(weighted_all_dist[raw_labels]).to(
                    device
                )

                weighted_loss = raw_loss * pixel_weights
                loss = torch.mean(torch.sum(weighted_loss.flatten(start_dim=1), axis=0))

                total_loss += loss.item()

            if calculate_statistics:
                if class_designation is None:
                    # Mask the unused values for metrics
                    preds = preds[~(labels == -1)]
                    labels = labels[~(labels == -1)]

                    tracker.update(preds, labels)
                    tracker2.update(
                        torch.argmax(preds, dim=1, keepdim=False)[~(labels == -1)],
                        labels[~(labels == -1)],
                    )
                else:
                    if pixel_mode is not None:
                        x, y = pixel_mode
                        preds = preds[:, :, x, y]
                        labels = labels[:, x, y]
                    tracker.update(preds, labels)

                if class_designation is not None and calculate_2d_hist:
                    # Reverse the 1 hot encoding on preds and bring everything to np
                    preds = torch.argmax(preds, dim=1, keepdim=False).cpu().numpy()
                    labels = labels.cpu().numpy()

                    mask = labels == -1
                    preds = preds[~mask]
                    labels = labels[~mask]
                    raw_labels = np.squeeze(raw_labels.numpy())
                    raw_labels = raw_labels[~mask]

                    preds_flat = preds.flatten()
                    raw_labels_flat = raw_labels.flatten()
                    histogram_input = np.column_stack((preds_flat, raw_labels_flat))
                    # hist_2d += h
                    for g in histogram_input:
                        hist_2d[g[0], g[1]] += 1

            counter += 1
            if num_batches >= 0 and counter >= num_batches:
                break

    if calculate_statistics:
        log(f"\n{pprint.pformat(tracker.compute_all())}")

    return (
        total_loss / (batch_idx + 1),
        None if not calculate_statistics else tracker.compute_all(),
        None if not calculate_2d_hist else hist_2d,
    )
