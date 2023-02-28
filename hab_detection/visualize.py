import os
import re
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import pprint
from .constants import (
    device,
    cyan_colormap,
    LOG_NAME,
    ZIP_PATH_TRAIN,
    ZIP_PATH_TEST,
    MODEL_SAVE_BASE_FOLDER,
)
from .metrics import get_metric_tracker, get_model_performance
from .helpers import log, set_config
from .dataset import get_image_dataset
from .model import load_model


def save_plot(image_save_folder, filename):
    plt.savefig(f"{image_save_folder}/{filename}.png")


def visualize(
    experiment_name,
    class_designation,
    model_architecture,
    model_file,
    epoch,
    dataset_type="test",
):
    # Set the config to print to stdout
    set_config("test")

    model_save_folder = f"{MODEL_SAVE_BASE_FOLDER}/{experiment_name}"
    image_save_folder = f"{model_save_folder}/visualize/{epoch}"
    log_file = f"{model_save_folder}/{LOG_NAME}"

    os.makedirs(image_save_folder, exist_ok=True)

    test_loss = []
    train_loss = []
    cur_epoch = None
    with open(log_file, "r") as f:
        for line in f.readlines():
            result = re.search(r"^.*Epoch (\d*) ([a-z]*) loss: (\d*[.]?\d*)", line)
            if result is not None:
                epoch, t, loss = result.groups()
                if cur_epoch is None:
                    cur_epoch = epoch
                else:
                    # Assert that train and test don't get misaligned
                    assert cur_epoch == epoch
                    cur_epoch = None
                if t == "test":
                    test_loss.append(float(loss))
                elif t == "train":
                    train_loss.append(float(loss))
                else:
                    raise Exception(f"Found unknown type {t} in log")

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(test_loss, color="b", label="Test")
    ax.plot(train_loss, color="r", label="Train")
    ax.set(xlabel="Epoch", ylabel="Loss")

    plt.legend()
    plt.title("Loss across training epochs")

    # plt.show()
    save_plot(image_save_folder, "loss")

    log(f"Loading the dataset")
    if dataset_type == "test":
        dataset = get_image_dataset(ZIP_PATH_TEST, class_designation)
    elif dataset_type == "train":
        dataset = get_image_dataset(ZIP_PATH_TRAIN, class_designation)
    else:
        raise Exception(f"Unknown dataset type {dataset_type}")
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    log(f"Loading the model")
    model = load_model(
        model_architecture, model_file, model_save_folder, class_designation
    )

    _, metrics, hist_2d = get_model_performance(
        model,
        loader,
        class_designation,
        num_batches=-1,
        calculate_2d_hist=True,
    )
    log(f"\n{pprint.pformat(metrics)}")

    """ Confusion Matrix """
    cm = np.squeeze(metrics["MulticlassConfusionMatrix"].cpu().numpy())
    cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    vmin = np.min(cmn)
    vmax = np.max(cmn)
    off_diag_mask = np.eye(*cmn.shape, dtype=bool)

    fig = plt.figure()
    gs0 = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[20, 2], hspace=0.05)
    gs00 = matplotlib.gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs0[1], hspace=0
    )

    ax = fig.add_subplot(gs0[0])
    cax1 = fig.add_subplot(gs00[0])
    # cax2 = fig.add_subplot(gs00[1])

    class_names = [
        f"{0 if i == 0 else class_designation[i-1]} - {a - 1}"
        for i, a in enumerate(class_designation)
    ]
    """
    sns.heatmap(
        cmn,
        annot=True,
        mask=~off_diag_mask,
        cmap="Blues",
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar_ax=cax2,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    """
    ax = sns.heatmap(
        cmn,
        annot=True,
        # mask=off_diag_mask,
        cmap="OrRd",
        vmin=0.0,
        vmax=1.0,
        ax=ax,
        cbar_ax=cax1,
        # cbar_kws=dict(ticks=[]),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    """
    cf_disp = ConfusionMatrixDisplay(cm)
    cf_disp.plot()
    """
    save_plot(image_save_folder, "confusion_matrix")

    if hist_2d is not None:
        fig, axs = plt.subplots(1, 1, figsize=(12, 8))
        sums = hist_2d.astype("float").sum(axis=0) + 1
        for i in range(len(class_designation)):
            axs.plot(
                hist_2d[i] / sums,
                label=f"Class {i + 1}",
                color=cyan_colormap[class_designation[i] - 1] / 255,
            )
            plt.axvline(x=class_designation[i], color="black", alpha=0.5)

        plt.legend()
        plt.autoscale(enable=True, axis="x", tight=True)

        save_plot(image_save_folder, "class_preds")
