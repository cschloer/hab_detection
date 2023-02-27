import os
from torch.utils.data import DataLoader
import torch
import numpy as np
import math
import pprint

from .constants import (
    device,
    ZIP_PATH_TRAIN,
    ZIP_PATH_TEST,
    MODEL_SAVE_BASE_FOLDER,
)
from .helpers import log, set_config
from .model import load_model, get_criterion, get_optimizer
from .dataset import get_image_dataset
from .metrics import get_model_performance, get_metric_tracker


def train(
    experiment_name,
    batch_size,
    # None for regression, a list of integers ending in 254 for class
    class_designation,
    class_weights,
    model_architecture,
    epoch_start=0,
    model_file=None,
):
    model_save_folder = f"{MODEL_SAVE_BASE_FOLDER}/{experiment_name}"
    os.makedirs(model_save_folder, exist_ok=True)
    set_config(experiment_name)
    try:
        log(
            f'Starting with model save folder "{model_save_folder}", training batch size "{batch_size}"'
        )
        log(f"Loading datasets...")

        train_dataset = get_image_dataset(ZIP_PATH_TRAIN, class_designation)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        test_dataset = get_image_dataset(ZIP_PATH_TEST, class_designation)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
        )
        log(f"Done loading datasets. Getting the model.")

        model = load_model(
            model_architecture, model_file, model_save_folder, class_designation
        )
        optimizer = get_optimizer(model)
        criterion = get_criterion(class_designation, class_weights)

        train_tracker = get_metric_tracker(class_designation)
        for epoch in range(epoch_start, 1000):  # Training loop

            log(f"Starting Epoch {epoch + 1}!")
            running_loss = 0
            total_loss = 0
            try:
                for batch_idx, (inputs, labels, _) in enumerate(train_loader):
                    model.train()
                    inputs = inputs.to(device, dtype=torch.float)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    preds = model(inputs)  # make prediction
                    loss = criterion(preds, labels)  # Calculate cross entropy loss
                    loss.backward()  # Backpropogate loss
                    optimizer.step()  # Apply gradient descent change to weight

                    running_loss += loss.item()
                    total_loss += loss.item()

                    if class_designation is None:
                        # Mask the unused values for metrics
                        preds = preds[~labels == -1]
                        labels = labels[~labels == -1]

                    train_tracker.update(preds, labels)

                    NUM_BATCHES = 100
                    if (
                        batch_idx % NUM_BATCHES == NUM_BATCHES - 1
                    ):  # print every 99 mini-batches
                        avg_loss = math.sqrt(running_loss / NUM_BATCHES)
                        log(
                            f"[{epoch + 1}, {batch_idx + 1:5d}] average loss: {avg_loss :.3f}"
                        )
                        running_loss = 0.0

                torch.save(model.state_dict(), f"{model_save_folder}/epoch_recent.pt")

                log(f"Epoch {epoch + 1} train loss: {total_loss / (batch_idx + 1)}")
                test_loss, test_metrics = get_model_performance(
                    model,
                    test_loader,
                    class_designation,
                    class_weights=class_weights,
                )
                log(f"Epoch {epoch + 1} test loss: {test_loss}")
                # Print out performance metrics
                log(f"Train statistics:")
                log(f"\n{pprint.pformat(train_tracker.compute_all())}")
                log(f"Test statistics:")
                log(f"\n{pprint.pformat(test_metrics)}")

                if epoch % 5 == 4 or epoch == 0:
                    torch.save(
                        model.state_dict(), f"{model_save_folder}/epoch_{epoch + 1}.pt"
                    )
                train_tracker.reset()

            finally:
                try:
                    # Clear GPU cache in case it crashes so it can run again
                    del inputs
                    del labels
                    del preds
                    torch.cuda.empty_cache()
                except:
                    pass

    except Exception as e:
        log(str(e))
        raise e
