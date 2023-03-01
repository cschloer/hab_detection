import os
import re
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
from torch.utils.data import DataLoader
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
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
    FULL_IMAGE_1_INPUT,
    FULL_IMAGE_1_LABEL,
)

from .metrics import get_metric_tracker, get_model_performance
from .helpers import log, set_config
from .dataset import get_image_dataset
from .model import load_model


def save_plot(image_save_folder, filename):
    plt.savefig(f"{image_save_folder}/{filename}.png")


def normalize_sen2(red, green, blue):
    def normalize(arr):
        """Function to normalize an input array to 0-1"""
        arr_min = arr.min()
        arr_max = arr.max()
        return (arr - arr_min) / (arr_max - arr_min)

    img = np.dstack((normalize(red), normalize(green), normalize(blue)))

    # Increase contrast
    pixvals = img
    minval = np.percentile(pixvals, 5)
    maxval = np.percentile(pixvals, 95)
    pixvals = np.clip(pixvals, minval, maxval)
    pixvals = ((pixvals - minval) / (maxval - minval)) * 1
    Image.fromarray(pixvals.astype(np.uint8))
    return pixvals


def visualize_full_image(
    model, dataset, class_designation, input_path, label_path, image_save_folder
):
    model.eval()
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    sen2_np = np.pad(
        np.load(FULL_IMAGE_1_INPUT).astype(np.float32),
        ((0, 0), (0, 6), (0, 2)),
    )
    sen2_img = normalize_sen2(sen2_np[1, :, :], sen2_np[2, :, :], sen2_np[3, :, :])
    ax = axs[1, 0]
    ax.set_title("Actual image")
    ax.imshow(sen2_img)
    ax.axis("off")

    cyan_np = np.load(label_path)
    cyan_reshaped = np.pad(
        cyan_np.reshape(cyan_np.shape[1], cyan_np.shape[2]), ((0, 6), (0, 2))
    )
    print("SHAPE AFTER PAD", cyan_reshaped.shape)

    custom_colormap = np.copy(cyan_colormap)
    prev_val = 0
    for i, c in enumerate(class_designation):
        cur_color = cyan_colormap[c if i != 0 else 0]
        print("CUR COLOR", cur_color)
        for j in range(c - prev_val):
            custom_colormap[prev_val + j] = cur_color
        prev_val = c

    print("UNQUE CUSTOM COLORMAP", np.unique(custom_colormap))
    print(custom_colormap)
    cyan_image = custom_colormap[cyan_reshaped]
    ax = axs[0, 0]
    ax.set_title("Actual HAB")
    ax.imshow(cyan_image)
    ax.axis("off")

    print("sen2 shape", sen2_np.transpose(1, 2, 0).shape)

    transformed_sen2 = dataset.transform_input(
        sen2_np.transpose(1, 2, 0) / 10000,
    )
    print("sen2 shape", transformed_sen2)

    transformed_sen2 = transformed_sen2.to(device, dtype=torch.float)
    pred = model(torch.unsqueeze(transformed_sen2, axis=0))  # make prediction
    pred = pred.cpu().detach()
    print("PRED SHAPE", pred.shape)

    pred = torch.argmax(pred, dim=1, keepdim=False).cpu().numpy()
    print("PRED HSAPE AFTER", pred.shape)

    save_plot(image_save_folder, "winnebago")


def visualize(
    experiment_name,
    class_designation,
    model_architecture,
    model_file,
    dataset_type="test",
):
    # Set the config to print to stdout
    set_config("test")

    model_save_folder = f"{MODEL_SAVE_BASE_FOLDER}/{experiment_name}"
    image_save_folder = f"{model_save_folder}/visualize/{dataset_type}"
    log_file = f"{model_save_folder}/{LOG_NAME}"

    os.makedirs(image_save_folder, exist_ok=True)
    log(f"Loading the model")

    model = load_model(
        model_architecture, model_file, model_save_folder, class_designation
    )

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

    visualize_full_image(
        model,
        dataset,
        class_designation,
        FULL_IMAGE_1_INPUT,
        FULL_IMAGE_1_LABEL,
        image_save_folder,
    )

    return

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
        rectangles = {}
        ranges = [
            (
                0 if i == 0 else class_designation[i - 1],
                class_designation[i],
            )
            for i in range(len(class_designation))
        ]

        for i in range(len(class_designation)):
            floor = ranges[i][0]
            ceil = ranges[i][1]

            rectangles[f"{floor} - {ceil -1}"] = mpatch.Rectangle(
                (floor, 0.95),
                ceil - floor,
                0.05,
                color=cyan_colormap[ceil - 1 if floor != 0 else 0] / 255,
            )

            normalized = hist_2d[i] / sums
            axs.plot(
                normalized,
                label=f"Class {i + 1}",
                color=cyan_colormap[ceil - 1 if floor != 0 else 0] / 255,
                alpha=0.3,
                linewidth=2.0,
            )
            # Plot thicker line inside of range where it is correct
            axs.plot(
                range(floor, ceil),
                normalized[floor:ceil],
                color=cyan_colormap[ceil - 1 if floor != 0 else 0] / 255,
                linewidth=2.0,
            )
            plt.axvline(x=class_designation[i], color="black", alpha=0.5)

        # Plot rectangles
        for r in rectangles:
            rect = rectangles[r]
            axs.add_artist(rect)
            rx, ry = rect.get_xy()
            cx = rx + rect.get_width() / 2.0
            cy = ry + rect.get_height() / 2.0

            axs.annotate(
                r,
                (cx, cy),
                color="white",
                weight="bold",
                fontsize=10,
                ha="center",
                va="center",
            )

        plt.autoscale(enable=True, axis="x", tight=True)
        axs.set_ylim(0.0, 1.0)
        axs.set_title("Classification of Pixels vs Actual HAB Index")
        plt.xlabel("Actual HAB Index")
        plt.ylabel("Fraction Classified")

        save_plot(image_save_folder, "class_preds")
