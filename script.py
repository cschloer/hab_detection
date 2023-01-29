import os
from torch.utils.data import DataLoader
import torch
import numpy as np
import math

from constants import (
    device,
    ZIP_PATH_TRAIN,
    ZIP_PATH_TEST,
    TRAINING_BATCH_SIZE,
    TEST_BATCH_SIZE,
    MODEL_SAVE_FOLDER,
    MODEL_LOAD_PATH,
)
from helpers import log
from model import load_model, get_criterion, get_optimizer
from dataset import get_image_dataset
from metrics import get_model_performance

try:
    log(
        f'Starting with model save folder "{MODEL_SAVE_FOLDER}", training batch size "{TRAINING_BATCH_SIZE}"'
    )
    log(f"Loading datasets...")

    train_dataset = get_image_dataset(ZIP_PATH_TRAIN)
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    test_dataset = get_image_dataset(ZIP_PATH_TEST)
    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    log(f"Done loading datasets. Getting the model.")

    model = load_model(MODEL_LOAD_PATH)
    optimizer = get_optimizer(model)
    criterion = get_criterion()

    for epoch in range(0, 1000):  # Training loop

        log(f"Starting Epoch {epoch + 1}!")
        running_loss = 0.0
        try:
            for batch_idx, (inputs, labels, _) in enumerate(train_loader):
                model.train()
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)["out"]  # make prediction
                loss = criterion(outputs, labels)  # Calculate cross entropy loss
                loss.backward()  # Backpropogate loss
                optimizer.step()  # Apply gradient descent change to weight
                running_loss += loss.item()
                if batch_idx % 25 == 24:  # print every 50 mini-batches
                    log(
                        f"[{epoch + 1}, {batch_idx + 1:5d}] average loss: {math.sqrt(running_loss / 25) :.3f}"
                    )
                    running_loss = 0.0

            os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
            torch.save(model.state_dict(), f"{MODEL_SAVE_FOLDER}/epoch_recent.pt")

            if epoch % 5 == 4 or epoch == 0:
                torch.save(
                    model.state_dict(), f"{MODEL_SAVE_FOLDER}/epoch_{epoch + 1}.pt"
                )
                get_model_performance(model, test_loader)
        finally:
            try:
                # Clear GPU cache in case it crashes so it can run again
                del inputs
                del labels
                del outputs
                gc.collect()
                torch.cuda.empty_cache()
            except:
                pass

except Exception as e:
    log(str(e))
    raise e
