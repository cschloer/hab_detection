import numpy as np
import math
import torch

from torchmetrics import Accuracy, ConfusionMatrix
from .helpers import log
from .constants import device
from .model import mse_loss_with_nans


def get_model_performance(
    model, loader, class_designation, num_batches=-1, additional_str=""
):
    # model_cpu = model.cpu()
    with torch.no_grad():
        model.eval()

        if class_designation is None:
            # It's a regression

            all_preds = torch.tensor([])
            all_labels = torch.tensor([])
            counter = 0
            sum = 0
            for batch_idx, (inputs, labels, _) in enumerate(loader):
                # print(f"{batch_idx + 1} / {len(loader)}")
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device)
                preds = model(inputs)["out"]  # make prediction

                loss = mse_loss_with_nans(preds, labels).cpu().detach()
                sum += loss.item()

                counter += 1
                if num_batches >= 0 and counter >= num_batches:
                    break

                # break
                del inputs
                del labels
                del preds

            log(f"{additional_str}MSE: {math.sqrt(sum / (batch_idx + 1))}")
        else:
            total_correct = 0
            total = 0
            cm = ConfusionMatrix(
                task="multiclass",
                num_classes=len(class_designation),
                ignore_index=-1,
            ).to(device)
            accuracy = Accuracy(
                task="multiclass",
                num_classes=len(class_designation),
                ignore_index=-1,
            ).to(device)
            for batch_idx, (inputs, labels, _) in enumerate(loader):
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device)
                preds = model(inputs)["out"]  # make prediction

                cm.update(preds, labels)
                accuracy.update(preds, labels)
            log(f"{additional_str}accuracy: {accuracy.compute()}")
            log(f"{additional_str}confusion matrix:\n{cm.compute()}")
