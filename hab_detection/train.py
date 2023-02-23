import os
from torch.utils.data import DataLoader
import torch
from torchmetrics import Accuracy, ConfusionMatrix
import numpy as np
import math

from .constants import (
    device,
    ZIP_PATH_TRAIN,
    ZIP_PATH_TEST,
    MODEL_SAVE_BASE_FOLDER,
)
from .helpers import log, set_config
from .model import load_model, get_criterion, get_optimizer
from .dataset import get_image_dataset
from .metrics import get_model_performance


def train(
    experiment_name,
    batch_size,
    # None for regression, a list of integers ending in 254 for class
    class_designation,
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
        criterion = get_criterion(class_designation)

        if class_designation is not None:
            train_accuracy = Accuracy(
                task="multiclass",
                num_classes=len(class_designation),
                ignore_index=-1,
            ).to(device)
            train_cm = ConfusionMatrix(
                task="multiclass",
                num_classes=len(class_designation),
                ignore_index=-1,
            ).to(device)
        for epoch in range(epoch_start, 1000):  # Training loop

            log(f"Starting Epoch {epoch + 1}!")
            running_accuracy = 0
            running_loss = 0
            try:
                for batch_idx, (inputs, labels, _) in enumerate(train_loader):
                    model.train()
                    inputs = inputs.to(device, dtype=torch.float)
                    labels = labels.to(device)
                    log(f"INPUTS SHAPE {inputs.shape}")
                    log(f"LABELS SHAPE {labels.shape}")

                    optimizer.zero_grad()
                    outputs = model(inputs)["out"]  # make prediction
                    log(f"OUTPUTS SHAPE {outputs.shape}")
                    loss = criterion(outputs, labels)  # Calculate cross entropy loss
                    loss.backward()  # Backpropogate loss
                    optimizer.step()  # Apply gradient descent change to weight
                    running_loss += loss.item()

                    if class_designation is not None:
                        batch_accuracy = train_accuracy(outputs, labels)
                        running_accuracy += batch_accuracy
                        train_cm.update(outputs, labels)

                    NUM_BATCHES = 100
                    if (
                        batch_idx % NUM_BATCHES == NUM_BATCHES - 1
                    ):  # print every 99 mini-batches
                        avg_loss = math.sqrt(running_loss / NUM_BATCHES)
                        if class_designation is not None:
                            avg_accuracy = math.sqrt(running_accuracy / NUM_BATCHES)
                            log(
                                f"[{epoch + 1}, {batch_idx + 1:5d}] avg loss: {avg_loss :.3f}, avg accuracy: {avg_accuracy :.3f}"
                            )
                        else:
                            log(
                                f"[{epoch + 1}, {batch_idx + 1:5d}] avg loss: {avg_loss :.3f}"
                            )
                        running_loss = 0.0
                        running_accuracy = 0.0

                torch.save(model.state_dict(), f"{model_save_folder}/epoch_recent.pt")

                # Print out performance metrics
                if epoch % 5 == 4 or epoch == 0:
                    torch.save(
                        model.state_dict(), f"{model_save_folder}/epoch_{epoch + 1}.pt"
                    )
                    log("Getting actual test model performance")
                    get_model_performance(model, test_loader, class_designation)
                else:
                    log("Getting subset test model performance")
                    get_model_performance(
                        model, test_loader, class_designation, num_batches=10
                    )
                if class_designation is not None:
                    log(f"Epoch {epoch + 1} train accuracy: {train_accuracy.compute()}")
                    log(
                        f"Epoch {epoch + 1} train confusion matrix: {train_cm.compute()}"
                    )
                    train_accuracy.reset()
                    train_cm.reset()
                else:
                    log("Getting subset train model performance")
                    get_model_performance(model, train_loader, num_batches=10)

            finally:
                try:
                    # Clear GPU cache in case it crashes so it can run again
                    del inputs
                    del labels
                    del outputs
                    torch.cuda.empty_cache()
                except:
                    pass

    except Exception as e:
        log(str(e))
        raise e
