import os
from torch.utils.data import DataLoader
import torch
import numpy as np
import math
import pprint
import psutil

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


def train_wrapper(
    experiment_name,
    batch_size,
    # None for regression, a list of integers ending in 254 for class
    class_designation,
    class_weights,
    model_architecture,
    randomize,
    learning_rate,
    epoch_start=0,
    model_file=None,
    weight_decay=0,
    epoch_limit=-1,
    track_statistics=False,
    log_progress=True,
    save_progress=True,
    subset_train=None,
    subset_test=None,
):
    if log_progress:
        model_save_folder = f"{MODEL_SAVE_BASE_FOLDER}/{experiment_name}"
        os.makedirs(model_save_folder, exist_ok=True)
        set_config(experiment_name)
        log(f"Loading datasets...")

    train_dataset = get_image_dataset(
        ZIP_PATH_TRAIN,
        class_designation,
        randomize=randomize,
        subset=subset_train,
        in_memory=False,
        use_unzipped=False,
    )
    test_dataset = get_image_dataset(
        ZIP_PATH_TEST,
        class_designation,
        subset=subset_test,
        in_memory=False,
        use_unzipped=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
    )

    if log_progress:
        log(f"Done loading datasets. Getting the model.")

    return train(
        train_loader,
        test_loader,
        experiment_name,
        batch_size,
        # None for regression, a list of integers ending in 254 for class
        class_designation,
        class_weights,
        model_architecture,
        randomize,
        learning_rate,
        epoch_start=epoch_start,
        model_file=model_file,
        weight_decay=weight_decay,
        epoch_limit=epoch_limit,
        track_statistics=track_statistics,
        log_progress=log_progress,
        save_progress=save_progress,
    )


def train(
    train_loader,
    test_loader,
    experiment_name,
    batch_size,
    # None for regression, a list of integers ending in 254 for class
    class_designation,
    class_weights,
    model_architecture,
    randomize,
    learning_rate,
    epoch_start=0,
    model_file=None,
    weight_decay=0,
    epoch_limit=-1,
    track_statistics=False,
    log_progress=True,
    save_progress=True,
):
    model_save_folder = f"{MODEL_SAVE_BASE_FOLDER}/{experiment_name}"
    try:
        if log_progress:
            log(
                f'Starting with model save folder "{model_save_folder}", training batch size "{batch_size}"'
            )

        model = load_model(
            model_architecture, model_file, model_save_folder, class_designation
        )
        if log_progress:
            print(model)
        optimizer = get_optimizer(model, learning_rate, weight_decay=weight_decay)
        criterion = get_criterion(class_designation, class_weights)

        if track_statistics:
            train_tracker = get_metric_tracker(class_designation)
        for epoch in range(epoch_start, 1000):  # Training loop

            if log_progress:
                log(f"Starting Epoch {epoch + 1}!")
            running_loss = 0
            total_loss = 0
            loss_list = []
            try:
                for batch_idx, (inputs, labels, _, _) in enumerate(train_loader):
                    model.train()
                    inputs = inputs.to(device, dtype=torch.float, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)


                    preds = model(inputs)  # make prediction
                    if isinstance(preds, dict):
                        preds = preds["out"]
                    loss = criterion(preds, labels)  # Calculate cross entropy loss

                    optimizer.zero_grad()
                    loss.backward()  # Backpropogate loss
                    optimizer.step()  # Apply gradient descent change to weight

                    running_loss += loss.item()
                    total_loss += loss.item()
                    loss_list.append(loss.item())

                    if track_statistics:
                        if class_designation is None:
                            # Mask the unused values for metrics
                            preds = preds[~labels == -1]
                            labels = labels[~labels == -1]

                        train_tracker.update(preds, labels)

                    exit()
                    NUM_BATCHES = 100
                    if (
                        batch_idx % NUM_BATCHES == NUM_BATCHES - 1
                    ):  # print every 99 mini-batches
                        avg_loss = running_loss / NUM_BATCHES
                        if log_progress:
                            log
                            log(
                                f"[{epoch + 1}, {batch_idx + 1:5d}] average loss: {avg_loss :.3f} -- using {round(psutil.Process(os.getpid()).memory_info().rss / (1<<30), 2)} GB"
                            )
                        running_loss = 0.0

                if save_progress:
                    torch.save(
                        model.state_dict(), f"{model_save_folder}/epoch_recent.pt"
                    )
                    if epoch % 5 == 4 or epoch == 0:
                        torch.save(
                            model.state_dict(),
                            f"{model_save_folder}/epoch_{epoch + 1}.pt",
                        )

                if log_progress:
                    log(f"Epoch {epoch + 1} train loss: {total_loss / (batch_idx + 1)} ({np.mean(loss_list)}))")
                    log(f"Epoch {epoch + 1} train loss std: {np.std(loss_list)}")
                    test_loss, _, _ = get_model_performance(
                        model,
                        test_loader,
                        class_designation,
                        class_weights=class_weights,
                        calculate_statistics=False,
                    )
                    log(f"Epoch {epoch + 1} test loss: {test_loss}")
                    # Print out performance metrics
                    if track_statistics:
                        log(f"Train statistics:")
                        log(f"\n{pprint.pformat(train_tracker.compute_all())}")
                        log(f"Test statistics:")
                        log(f"\n{pprint.pformat(test_metrics)}")
                    if track_statistics:
                        train_tracker.reset()

                if epoch_limit >= 0 and epoch + 1 >= epoch_limit:
                    return model
                loss_list = []

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
